from __future__ import annotations
import glob
import heapq
import json
import random
import shutil
import sqlite3
import time
from pathlib import Path
from .common import Config,atomic_parquet,bind_run,check_disk,connect,digest,lock,read_json,rows_from_parquet,write_json,sha
from .text import record_from_row

COLS=['entity_id','business_name','business_address','country','normalized_name','stripped_name','accent_name','normalized_address']


def processed_files(cfg,split,source):
    files=sorted((cfg.processed/split/source).rglob('*.parquet'))
    if not files:raise FileNotFoundError(f'Missing processed {split}/{source}. Re-download processed/v2, do not silently use partial data.')
    return files


def inventory(cfg):
    import pyarrow.parquet as pq
    m=read_json(cfg.processed/'_manifest.json')
    found={}
    for item in m['files']:
        files=processed_files(cfg,item['split'],item['source'])
        n=sum(pq.ParquetFile(p).metadata.num_rows for p in files)
        if n!=item['rows']:raise ValueError(f'Row count mismatch for {item["file"]}: {n} vs {item["rows"]}')
        found[item['file']]={'rows':n,'files':[{'path':str(p.relative_to(cfg.processed)),'bytes':p.stat().st_size,'rows':pq.ParquetFile(p).metadata.num_rows} for p in files]}
    return found,m


def sample(cfg):
    """Global reservoir sample of S1 labels, then four disjoint randomized roles."""
    import pyarrow as pa
    bind_run(cfg);out=cfg.work/'samples'
    if (out/'complete.json').exists():return
    reproduced=cfg.root/'artifacts/samples/queries.parquet'
    if reproduced.exists():
        saved=read_json(cfg.root/'artifacts/samples/complete.json')
        if saved['config']!=cfg.fingerprint or sha(reproduced)!=saved['sample_sha256']:
            raise ValueError('Packaged training sample is inconsistent with configuration/hash')
        out.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(reproduced,out/'queries.parquet')
        write_json(out/'complete.json',saved)
        print('Reusing exact packaged disjoint sample for reproduction.',flush=True)
        return
    exclude=set();used=[]
    for pattern in cfg['exclude_sample_globs']:
        for p in glob.glob(str(cfg.root/pattern)):
            import pyarrow.parquet as pq
            names=pq.ParquetFile(p).schema_arrow.names
            col='entity_id' if 'entity_id' in names else 'source1_entity_id' if 'source1_entity_id' in names else None
            if not col:continue
            for b in rows_from_parquet(p,[col]):exclude.update(b[col])
            used.append(p)
    count=sum(cfg['sample'].values());rng=random.Random(cfg['seed']);selected=[];seen=0
    for p in processed_files(cfg,'train','ground_truth'):
        for b in rows_from_parquet(p,['source1_entity_id','matched_entity_ids']):
            for q,truth in zip(b['source1_entity_id'],b['matched_entity_ids']):
                if q in exclude:continue
                seen+=1;t=[x.strip() for x in (truth or '').split(',') if x.strip()]
                if len(t)!=len(set(t)) or not all(x.startswith(('S2-','S3-')) for x in t):raise ValueError('Invalid ground-truth target list')
                item=(q,t)
                if len(selected)<count:selected.append(item)
                else:
                    j=rng.randrange(seen)
                    if j<count:selected[j]=item
    if len(selected)!=count:raise ValueError('Not enough eligible S1 entities for configured sample')
    rng.shuffle(selected);lookup={};offset=0
    for role,n in cfg['sample'].items():
        for q,t in selected[offset:offset+n]:lookup[q]={'role':role,'truth':t}
        offset+=n
    result=[]
    for p in processed_files(cfg,'train','source1'):
        for b in rows_from_parquet(p,COLS):
            for i,q in enumerate(b['entity_id']):
                if q in lookup:
                    row=record_from_row({k:v[i] for k,v in b.items()});row.update(lookup.pop(q));result.append(row)
    if lookup:raise ValueError('Sampled S1 IDs not found')
    result.sort(key=lambda x:x['entity_id'])
    if len({r['entity_id'] for r in result})!=len(result):raise ValueError('Duplicate sample IDs')
    atomic_parquet(out/'queries.parquet',pa.Table.from_pylist(result))
    write_json(out/'complete.json',{'config':cfg.fingerprint,'entities':len(result),'role_counts':cfg['sample'],'excluded_count':len(exclude),'exclusion_files':used,'sample_sha256':sha(out/'queries.parquet')})
    print('Fixed sample ready:',cfg['sample'],flush=True)


def _build_one(args):
    config_path,split,country=args;cfg=Config(config_path)
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder=cfg.country_dir(split,country);folder.mkdir(parents=True,exist_ok=True)
    with lock(folder/'build.lock'):
        if (folder/'complete.json').exists():return
        check_disk(cfg)
        schema=pa.schema([pa.field('rid',pa.int64())]+[pa.field(c,pa.string()) for c in COLS]+[pa.field('role',pa.string()),pa.field('truth',pa.list_(pa.string()))])
        counts={}
        for kind,sources in [('targets',['source2','source3']),('queries',['source1'])]:
            db=folder/f'{kind}.sqlite';tmpdb=folder/f'{kind}.building.sqlite'
            if tmpdb.exists():tmpdb.unlink() # only incomplete staging output in this fresh run
            con=connect(tmpdb)
            con.execute('CREATE TABLE records(rid INTEGER PRIMARY KEY,'+','.join(c+' TEXT NOT NULL' for c in COLS)+',role TEXT NOT NULL,truth TEXT NOT NULL)')
            out=folder/f'{kind}.parquet';tmp=folder/f'{kind}.building.parquet';rid=0
            paths=[cfg.work/'samples/queries.parquet'] if split=='train' and kind=='queries' else [p for source in sources for p in processed_files(cfg,split,source)]
            with pq.ParquetWriter(tmp,schema,compression='zstd') as writer:
                for p in paths:
                    columns=COLS+(['role','truth'] if split=='train' and kind=='queries' else [])
                    for b in rows_from_parquet(p,columns,cfg['read_batch_rows']):
                        rows=[]
                        for i,c in enumerate(b['country']):
                            if (c or '').strip()!=country:continue
                            r=record_from_row({k:v[i] for k,v in b.items()});r['rid']=rid;rid+=1
                            expected=('S2-','S3-') if kind=='targets' else ('S1-',)
                            if not r['entity_id'].startswith(expected):raise ValueError('Wrong ID source in processed partition')
                            r['role']=b['role'][i] if 'role'in b else 'test'
                            r['truth']=b['truth'][i] if 'truth'in b else []
                            rows.append(r)
                        if not rows:continue
                        writer.write_table(pa.Table.from_pylist(rows,schema=schema))
                        con.executemany('INSERT INTO records VALUES('+','.join('?' for _ in range(len(COLS)+3))+')',[(r['rid'],*[r[c] for c in COLS],r['role'],json.dumps(r['truth'])) for r in rows])
                        con.commit()
            con.execute('CREATE UNIQUE INDEX id_idx ON records(entity_id)')
            con.execute('CREATE INDEX name_idx ON records(normalized_name)');con.commit();con.close()
            tmp.replace(out);tmpdb.replace(db);counts[kind]=rid
        
        manifest=read_json(cfg.processed/'_manifest.json')
        expected=sum(f.get('countries',{}).get(country,0) for f in manifest['files'] if f['split']==split and f['source'] in ('source2','source3'))
        if counts['targets']!=expected:raise ValueError('Country target store disagrees with processed manifest')
        if split=='test':
            expected_q=sum(f.get('countries',{}).get(country,0) for f in manifest['files'] if f['split']==split and f['source']=='source1')
            if counts['queries']!=expected_q:raise ValueError('Country test query store disagrees with manifest')
        write_json(folder/'complete.json',{'country':country,'split':split,**counts})
        print('Corpus ready',split,country,counts,flush=True)


def prepare(cfg):
    import concurrent.futures as cf
    bind_run(cfg);inv,manifest=inventory(cfg)
    p=cfg.work/'input_manifest.json'
    if p.exists() and read_json(p)!=inv:raise ValueError('Processed input inventory changed in an existing run')
    write_json(p,inv);sample(cfg)
    tasks=[]
    for split in ['train','test']:
        countries=sorted({c for item in manifest['files'] if item['split']==split and item['source']=='source1' for c in item['countries']})
        if ''in countries:raise ValueError('Missing-country policy requires review; not silently dropped')
        write_json(cfg.work/f'{split}.json',{'countries':countries})
        tasks.extend((str(cfg.path),split,c) for c in countries)
    # Building two country stores at a time bounds simultaneous Python/Arrow allocations.
    with cf.ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(_build_one,tasks))
    print('Prepared all train/test countries; raw and processed input files unchanged.',flush=True)
