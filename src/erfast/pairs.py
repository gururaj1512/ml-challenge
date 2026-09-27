from __future__ import annotations
import concurrent.futures as cf
from functools import lru_cache
import json
import time
import numpy as np
from .common import Config,atomic_parquet,check_disk,connect,lock,read_json,records,write_json
from .features import FEATURES,make_features
from .lexical import PairCosine,ROUTES

_STATE=None


def fuse(route_ids,rrf_constant,cap):
    scores={};ranks={}
    for route,ids in route_ids.items():
        for rank,tid in enumerate(ids,1):
            tid=int(tid)
            if tid<0:continue
            if route in ranks.setdefault(tid,{}):continue
            ranks[tid][route]=rank
            scores[tid]=scores.get(tid,0)+1.0/(rrf_constant+rank)
    order=sorted(scores,key=lambda i:(-scores[i],i))
    return order[:cap],ranks,order


class _Adapter:
    def __init__(self,vals,targets,freq):
        self.lookup={int(t):i for i,t in enumerate(targets)};self.vals=vals;self.freq=freq
    def cosine(self,a,b,field):return float(self.vals[field][self.lookup[b['rid']]])
    def name_frequency(self,name):return self.freq(name)


def _init(config,split,country,budget):
    global _STATE
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    cfg=Config(config);root=cfg.country_dir(split,country)
    routes={r:np.load(root/'lexical'/r/'top_ids.npy',mmap_mode='r') for r in (*ROUTES,'embedding')}
    tc=connect(root/'targets.sqlite',True);qc=connect(root/'queries.sqlite',True)
    @lru_cache(maxsize=20000)
    def freq(name):
        return tc.execute('SELECT count(*) FROM records WHERE normalized_name=?',(name,)).fetchone()[0] if name else 0
    _STATE=(cfg,split,country,budget,root,routes,tc,qc,PairCosine(root),np.load(root/'targets_e5.npy',mmap_mode='r'),np.load(root/'queries_e5.npy',mmap_mode='r'),freq)


def _task(bounds):
    import pyarrow as pa
    cfg,split,country,budget,root,routes,tc,qc,cos,te,qe,freq=_STATE
    lo,hi=bounds;out=cfg.work/'pairs'/split/root.name/f'{lo:09d}'
    if (out/'complete.json').exists():return out
    out.mkdir(parents=True,exist_ok=True);check_disk(cfg);start=time.monotonic()
    qrecords=records(qc,range(lo,hi));selected=[];all_targets=set();qsumm=[]
    for qid in range(lo,hi):
        q=qrecords[qid];truth=set(json.loads(q['truth']))
        r={name:[int(i) for i in arr[qid] if i>=0] for name,arr in routes.items()}
        if not q['normalized_name']:r['embedding']=[]
        exact=[int(z[0]) for z in tc.execute('SELECT rid FROM records WHERE normalized_name=? ORDER BY rid',(q['normalized_name'],))] if q['normalized_name'] else []
        # Preserve the COMPLETE exact-name hit set; order it by address cosine before fusion.
        if exact:
            values=cos.values(qid,exact)['address_char'];order=np.lexsort((np.asarray(exact),-values));exact=[exact[int(j)] for j in order]
        r['exact']=exact
        kept,ranks,raw=fuse(r,cfg['retrieval']['rrf_constant'],budget)
        selected.append((qid,kept,ranks,raw));all_targets.update(kept)
        # Raw retrieval truth is diagnostic only; never used by fusion or candidate selection.
        raw_rec=records(tc,raw) if truth else {}
        qsumm.append({'source1_entity_id':q['entity_id'],'country':country,'role':q['role'],'truth':sorted(truth),
                      'raw_count':len(raw),'candidate_count':len(kept),'raw_true_links':sum(raw_rec[i]['entity_id'] in truth for i in raw_rec)})
    trecords=records(tc,all_targets);features=[];pairs=[]
    for qid,kept,ranks,raw in selected:
        q=qrecords[qid];truth=set(json.loads(q['truth']))
        if not kept:continue
        vals=cos.values(qid,kept);adapter=_Adapter(vals,kept,freq)
        dense=np.asarray(te[kept],dtype=np.float32)@np.asarray(qe[qid],dtype=np.float32)
        for j,tid in enumerate(kept):
            t=trecords[tid];f=make_features(q,t,ranks[tid],adapter,float(dense[j]))
            features.append([f[name] for name in FEATURES])
            pairs.append({'source1_entity_id':q['entity_id'],'target_entity_id':t['entity_id'],
                          'role':q['role'],'candidate_rank':j+1,'label':int(t['entity_id'] in truth),'country':country})
    X=np.asarray(features,dtype=np.float32).reshape(-1,len(FEATURES))
    np.save(out/'X.npy',X)
    ps=pa.schema([('source1_entity_id',pa.string()),('target_entity_id',pa.string()),('role',pa.string()),('candidate_rank',pa.int32()),('label',pa.int8()),('country',pa.string())])
    qs=pa.schema([('source1_entity_id',pa.string()),('country',pa.string()),('role',pa.string()),('truth',pa.list_(pa.string())),('raw_count',pa.int32()),('candidate_count',pa.int32()),('raw_true_links',pa.int32())])
    atomic_parquet(out/'pairs.parquet',pa.Table.from_pylist(pairs,schema=ps))
    atomic_parquet(out/'queries.parquet',pa.Table.from_pylist(qsumm,schema=qs))
    write_json(out/'complete.json',{'queries':hi-lo,'pairs':len(pairs),'lo':lo,'hi':hi,'budget':budget,'seconds':time.monotonic()-start})
    return out


def mine(cfg,split):
    budget=max(cfg['retrieval']['candidate_budgets']) if split=='train' else read_json(cfg.work/'models/selection.json')['candidate_budget']
    for c in cfg.countries(split):
        root=cfg.country_dir(split,c)
        for r in (*ROUTES,'embedding'):
            if not(root/'lexical'/r/'complete.json').exists():raise RuntimeError(f'Missing route {r}, {split}/{c}; no algorithm is silently skipped')
        n=read_json(root/'complete.json')['queries'];batch=cfg['feature_shard_rows']
        boundaries=[(i,min(n,i+batch)) for i in range(0,n,batch)];out=cfg.work/'pairs'/split/root.name
        with lock(out/'mine.lock'):
            if (out/'complete.json').exists():continue
            t=time.monotonic()
            with cf.ProcessPoolExecutor(max_workers=cfg['feature_workers'],initializer=_init,initargs=(str(cfg.path),split,c,budget)) as pool:
                futures=[pool.submit(_task,b) for b in boundaries]
                for i,f in enumerate(cf.as_completed(futures),1):
                    f.result();rate=i/max(.001,time.monotonic()-t)
                    print(f'Features {split}/{c}: {i}/{len(boundaries)} shards; ETA {(len(boundaries)-i)/max(rate,.001)/60:.1f} min',flush=True)
            write_json(out/'complete.json',{'queries':n,'budget':budget,'shards':len(boundaries)})
