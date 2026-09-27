from __future__ import annotations
import csv
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from .common import atomic_parquet,connect,read_json,sha,write_json
from .training import load_model,shards


def deployment(cfg):
    root=cfg.work/'models';d=read_json(root/'deployment.json')
    if d['config']!=cfg.fingerprint or d['model_sha256']!=sha(root/'matcher.txt') or d['selection_sha256']!=sha(root/'selection.json'):raise ValueError('Deployment approval is stale')
    return d


def predict(cfg):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d=deployment(cfg);model=load_model(cfg);start=time.monotonic();count=0
    for s in shards(cfg,'test'):
        out=cfg.work/'predictions'/s.parent.name/s.name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():
            if read_json(out/'complete.json')['deployment']!=d:raise ValueError('Prediction shard uses another model')
            continue
        p=pq.ParquetFile(s/'pairs.parquet').read().to_pylist();q=pq.ParquetFile(s/'queries.parquet').read().to_pylist()
        x=np.load(s/'X.npy',mmap_mode='r')
        if len(p)!=len(x):raise ValueError('Feature/metadata mismatch')
        if any(r['candidate_rank']>d['candidate_budget'] for r in p):raise ValueError('Model would score pairs outside submitted budget')
        prob=model.predict(x,num_threads=cfg['matcher']['num_threads']) if len(x) else np.empty(0)
        grouped={r['source1_entity_id']:[] for r in q}
        for r,z in zip(p,prob):grouped[r['source1_entity_id']].append((r['candidate_rank'],r['target_entity_id'],float(z)))
        outrows=[]
        for r in q:
            items=sorted(grouped[r['source1_entity_id']]);cand=[t for _,t,z in items];sc=[z for _,t,z in items]
            outrows.append({'source1_entity_id':r['source1_entity_id'],'country':r['country'],'candidate_entity_ids':cand,'probabilities':sc,'matched_entity_ids':[t for (_,t,z) in items if z>=d['threshold']]})
        schema=pa.schema([('source1_entity_id',pa.string()),('country',pa.string()),('candidate_entity_ids',pa.list_(pa.string())),('probabilities',pa.list_(pa.float64())),('matched_entity_ids',pa.list_(pa.string()))])
        atomic_parquet(out/'predictions.parquet',pa.Table.from_pylist(outrows,schema=schema))
        write_json(out/'complete.json',{'deployment':d,'queries':len(q),'pairs_scored':len(p)})
        count+=len(q);print('Predicted',count,'queries; elapsed',round(time.monotonic()-start),'s',flush=True)


def export(cfg):
    import pyarrow.parquet as pq
    d=deployment(cfg);out=cfg.work/'output';out.mkdir(parents=True,exist_ok=True)
    candpath=out/'candidate_pairs.tsv';matchpath=out/'matching_results.tsv'
    counts=[];all_seen=set();total=0
    with (out/'candidate_pairs.tsv.tmp').open('w',newline='') as cf,(out/'matching_results.tsv.tmp').open('w',newline='') as mf:
        cw=csv.writer(cf,delimiter='\t',lineterminator='\n');mw=csv.writer(mf,delimiter='\t',lineterminator='\n')
        cw.writerow(['source1_entity_id','candidate_entity_ids']);mw.writerow(['source1_entity_id','matched_entity_ids'])
        for country in cfg.countries('test'):
            root=cfg.country_dir('test',country);qc=connect(root/'queries.sqlite',True);tc=connect(root/'targets.sqlite',True)
            expected=qc.execute('SELECT count(*) FROM records').fetchone()[0];n=0
            jobs=sorted((cfg.work/'pairs/test'/root.name).glob('[0-9]*'))
            for job in jobs:
                p=cfg.work/'predictions'/root.name/job.name
                if not(p/'complete.json').exists() or read_json(p/'complete.json')['deployment']!=d:raise ValueError(f'Incomplete prediction {p}')
                for r in pq.ParquetFile(p/'predictions.parquet').read().to_pylist():
                    qid=r['source1_entity_id'];cand=r['candidate_entity_ids'];matches=r['matched_entity_ids'];prob=r['probabilities']
                    if qid in all_seen:raise ValueError('Duplicate S1 output')
                    all_seen.add(qid)
                    if not qc.execute('SELECT 1 FROM records WHERE entity_id=?',(qid,)).fetchone():raise ValueError('Unknown test S1')
                    if len(cand)!=len(set(cand)) or len(cand)!=len(prob) or len(cand)>d['candidate_budget']:raise ValueError('Invalid candidate list')
                    if any(not t.startswith(('S2-','S3-')) for t in cand):raise ValueError('Invalid target prefix')
                    if cand:
                        got=tc.execute('SELECT count(*) FROM records WHERE entity_id IN ('+','.join('?' for _ in cand)+')',cand).fetchone()[0]
                        if got!=len(cand):raise ValueError('Unknown candidate')
                    if matches!=[t for t,z in zip(cand,prob) if z>=d['threshold']]:raise ValueError('Matches do not equal scored/thresholded candidates')
                    cw.writerow([qid,','.join(cand)]);mw.writerow([qid,','.join(matches)])
                    counts.append(len(cand));n+=1
            if n!=expected:raise ValueError(f'S1 coverage mismatch {country}: {n}/{expected}')
            total+=n;qc.close();tc.close()
    (out/'candidate_pairs.tsv.tmp').replace(candpath);(out/'matching_results.tsv.tmp').replace(matchpath)
    report={'status':'PASS','entities':total,'candidate_count':sum(counts),'mean_candidates':float(np.mean(counts)),'p95_candidates':float(np.percentile(counts,95)),'max_candidates':max(counts,default=0),'candidate_file_sha256':sha(candpath),'matching_file_sha256':sha(matchpath),'deployment':d}
    write_json(out/'validation_report.json',report);print(json.dumps(report,indent=2))


def package(cfg):
    out=cfg.work/'output';report=read_json(out/'validation_report.json')
    if report['candidate_file_sha256']!=sha(out/'candidate_pairs.tsv') or report['matching_file_sha256']!=sha(out/'matching_results.tsv'):raise ValueError('Output changed since validation')
    target=cfg.work/'gururaj_submission.zip'
    method='# Methodology\n\nAll nine V4 routes, full-split TF-IDF, native exact positive-score WAND search, E5 GPU IVF-Flat (approximate, no PQ), 48-feature LightGBM.\n\n'
    method+='Candidate cap is selected before final matcher scoring; all scored test pairs are in candidate_pairs.tsv. No external business lookup.\n\n'
    method+='Original raw/preprocessed challenge data only. France is processed as a separate dynamically discovered country without labeled France training.\n\n'
    for name in ['fit.json','selection.json','audit.json']:
        method+='## '+name+'\n```json\n'+(cfg.work/'models'/name).read_text()+'\n```\n'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in ['candidate_pairs.tsv','matching_results.tsv']:z.write(out/name,'output/'+name)
        for p in cfg.root.rglob('*'):
            if not p.is_file() or any(x in {'.git','__pycache__','data','logs','work','.venv'} or x.endswith('.egg-info') for x in p.relative_to(cfg.root).parts):continue
            if p.suffix in {'.so','.pyc','.npy','.parquet','.zip','.pem'}:continue
            z.write(p,'code/business_entity_resolution/'+str(p.relative_to(cfg.root)))
        for name in ['matcher.txt','fit.json','selection.json','audit.json','deployment.json']:
            z.write(cfg.work/'models'/name,'code/business_entity_resolution/artifacts/'+name)
        z.write(cfg.work/'assets/model.json','code/business_entity_resolution/artifacts/e5_revision.json')
        for name in ['queries.parquet','complete.json']:
            z.write(cfg.work/'samples'/name,'code/business_entity_resolution/artifacts/samples/'+name)
        z.writestr('Documentation_template.md',method)
    print('Submission package:',target)
