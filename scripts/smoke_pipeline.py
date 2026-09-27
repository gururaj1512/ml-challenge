"""Real Arrow/SQLite/lexical/CUDA-FAISS/LightGBM/export integration on synthetic rows.
E5 vectors are deterministic fixtures; this does NOT claim to test real E5 encoding.
No challenge files are read or changed.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import numpy as np
import pyarrow as pa
from erfast.common import Config,atomic_parquet,connect,write_json
from erfast.text import record_from_row
from erfast import data,lexical,embeddings,pairs,training,output


def main():
    root=Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix='er-pipeline-smoke-') as td:
        td=Path(td);c=json.loads((root/'config.json').read_text())
        c.update(processed_dir=str(td/'processed'),raw_dir=str(td/'raw'),work_dir=str(td/'work'),minimum_free_gib=0,cpu_threads=2,lexical_workers=1,feature_workers=1,read_batch_rows=100,query_batch_rows=20,feature_shard_rows=25)
        c['exclude_sample_globs']=[];c['sample']={'fit':60,'early':20,'calibrate':20,'audit':20}
        c['embedding']['gpu_ids']=[0];c['matcher']['num_boost_round']=40;c['matcher']['early_stopping_rounds']=5;c['matcher']['num_threads']=2;c['retrieval']['candidate_budgets']=[16,24]
        p=td/'config.json';write_json(p,c);cfg=Config(p);manifest=[];truth=[]
        for split,cs in [('train',['US','India']),('test',['US','India','France'])]:
            for source in ['source1','source2','source3']:
                total=0;counts={}
                for country in cs:
                    rs=[]
                    for i in range(100 if split=='train' else 20):
                        if source!='source1' and i%10==0:continue
                        tag=f'{split}-{country}-{i:04d}';eid={'source1':'S1-','source2':'S2-','source3':'S3-'}[source]+tag
                        r=record_from_row({'entity_id':eid,'business_name':f'Clinic {country} {i:04d} Private Limited','business_address':f'{i+100} Park Lane, City {i%5}','country':country})
                        rs.append(r)
                        if split=='train' and source=='source1':
                            ids=[] if i%10==0 else ['S2-'+tag,'S3-'+tag]
                            truth.append({'source1_entity_id':eid,'matched_entity_ids':','.join(ids)})
                    atomic_parquet(cfg.processed/split/source/f'country={country}'/'part-0.parquet',pa.Table.from_pylist(rs))
                    total+=len(rs);counts[country]=len(rs)
                manifest.append({'file':f'{split}_{source}.tsv','split':split,'source':source,'rows':total,'countries':counts})
        atomic_parquet(cfg.processed/'train/ground_truth/part-0.parquet',pa.Table.from_pylist(truth))
        manifest.append({'file':'train_ground_truth.tsv','split':'train','source':'ground_truth','rows':len(truth)})
        write_json(cfg.processed/'_manifest.json',{'files':manifest})
        data.prepare(cfg)
        for split in ['train','test']:
            lexical.build(cfg,split)
            for country in cfg.countries(split):
                folder=cfg.country_dir(split,country)
                for kind in ['targets','queries']:
                    db=connect(folder/f'{kind}.sqlite',True);names=[r[0] for r in db.execute('SELECT normalized_name FROM records ORDER BY rid')];db.close()
                    es=[]
                    for name in names:
                        seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16);e=np.random.default_rng(seed).normal(size=384).astype(np.float32);es.append(e/np.linalg.norm(e))
                    np.save(folder/f'{kind}_e5.npy',np.asarray(es,dtype=np.float32));write_json(folder/f'{kind}_e5.complete.json',{'fixture_only':True})
            embeddings.ann(cfg,split)
        pairs.mine(cfg,'train');training.fit(cfg);training.calibrate(cfg);training.audit(cfg);training.accept(cfg)
        pairs.mine(cfg,'test');output.predict(cfg);output.export(cfg)
        report=json.loads((cfg.work/'output/validation_report.json').read_text())
        assert report['status']=='PASS' and report['entities']==60,report
        print('PASS: synthetic all-nine-route pipeline, independent splits, 48 features, model, budget calibration, audit and both TSVs.')


if __name__=='__main__':main()
