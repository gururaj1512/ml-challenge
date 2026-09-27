from __future__ import annotations
import concurrent.futures as cf
import gc
import math
import multiprocessing as mp
import time
import numpy as np
from .common import Config,check_disk,lock,read_json,write_json


def prepare_model(cfg):
    from huggingface_hub import HfApi,snapshot_download
    root=cfg.work/'assets';root.mkdir(parents=True,exist_ok=True)
    lockfile=root/'model.json'
    if lockfile.exists():return
    revision=cfg['embedding']['revision']
    reproduced=cfg.root/'artifacts/e5_revision.json'
    if reproduced.exists():
        pinned=read_json(reproduced)
        if pinned['model']!=cfg['embedding']['model']:raise ValueError('Packaged E5 model ID disagrees with configuration')
        revision=pinned['revision']
    info=HfApi().model_info(cfg['embedding']['model'],revision=revision)
    card=info.card_data.to_dict() if info.card_data is not None else {}
    if card.get('license') not in ('mit','apache-2.0'):raise RuntimeError('Model license not confirmed MIT/Apache-2.0')
    snapshot_download(repo_id=info.id,revision=info.sha,local_dir=str(root/'e5'),allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.md','1_Pooling/*'])
    write_json(lockfile,{'model':info.id,'revision':info.sha,'license':card.get('license')})


def _encode_worker(args):
    config,parquet,array,start,end,gpu,kind=args;cfg=Config(config)
    import torch
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required. No CPU fallback in this EC2 runner.')
    device=f'cuda:{gpu}';model=SentenceTransformer(str(cfg.work/'assets/e5'),device=device)
    model.max_seq_length=cfg['embedding']['max_length'];model.eval()
    output=np.load(array,mmap_mode='r+');pf=pq.ParquetFile(parquet)
    cursor=0;done=0;t=time.monotonic();prefix='passage: ' if kind=='targets' else 'query: '
    for group in range(pf.num_row_groups):
        n=pf.metadata.row_group(group).num_rows;lo=max(start,cursor);hi=min(end,cursor+n)
        if lo<hi:
            names=pf.read_row_group(group,columns=['normalized_name']).column(0).slice(lo-cursor,hi-lo).to_pylist()
            texts=[prefix+(s or '') for s in names]
            with torch.inference_mode():
                e=model.encode(texts,batch_size=cfg['embedding']['batch_size'],show_progress_bar=False,normalize_embeddings=True,convert_to_numpy=True)
            if e.shape!=(hi-lo,cfg['embedding']['dimension']) or not np.isfinite(e).all():raise ValueError('Invalid encoder output')
            output[lo:hi]=e.astype(np.float32);output.flush();done+=hi-lo
            if done and (group%10==0 or hi==end):print(f'GPU {gpu} {kind}: {done:,}/{end-start:,}; {done/max(.001,time.monotonic()-t):.0f} names/s',flush=True)
        cursor+=n
        if cursor>=end:break
    if done!=end-start:raise RuntimeError('Encoder worker did not write all assigned rows')
    output.flush();return done


def encode(cfg,split,kind):
    prepare_model(cfg)
    import torch
    ids=cfg['embedding']['gpu_ids']
    if max(ids)>=torch.cuda.device_count():raise RuntimeError('Requested GPU IDs are not available. Set configuration before starting a fresh run.')
    for country in cfg.countries(split):
        root=cfg.country_dir(split,country);n=read_json(root/'complete.json')[kind]
        done=root/f'{kind}_e5.complete.json';out=root/f'{kind}_e5.npy'
        with lock(root/f'{kind}_e5.lock'):
            if done.exists():continue
            check_disk(cfg);x=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=(n,cfg['embedding']['dimension']));x.flush();del x
            if n:
                boundaries=np.linspace(0,n,len(ids)+1,dtype=np.int64)
                args=[(str(cfg.path),str(root/f'{kind}.parquet'),str(out),int(boundaries[j]),int(boundaries[j+1]),g,kind) for j,g in enumerate(ids) if boundaries[j]<boundaries[j+1]]
                with cf.ProcessPoolExecutor(max_workers=len(args),mp_context=mp.get_context('spawn')) as pool:
                    total=sum(pool.map(_encode_worker,args))
                if total!=n:raise RuntimeError('Incomplete embeddings')
            write_json(done,{'rows':n,'dim':cfg['embedding']['dimension'],'dtype':'float32','model':read_json(cfg.work/'assets/model.json')})
        print('Embeddings complete',split,country,kind,n,flush=True)


def ann(cfg,split):
    import faiss
    if not hasattr(faiss,'StandardGpuResources'):raise RuntimeError('Install conda faiss-gpu; CPU-only wheel is not this configuration.')
    faiss.omp_set_num_threads(cfg['cpu_threads']);gpu=cfg['embedding']['gpu_ids'][0]
    for country in cfg.countries(split):
        root=cfg.country_dir(split,country);complete=root/'ann.complete.json'
        if complete.exists():continue
        if not(root/'targets_e5.complete.json').exists() or not(root/'queries_e5.complete.json').exists():raise RuntimeError('Both target and query embeddings are required')
        e=np.load(root/'targets_e5.npy',mmap_mode='r');q=np.load(root/'queries_e5.npy',mmap_mode='r')
        n,d=e.shape;lists=min(cfg['embedding']['ann_lists'],max(1,n//64));train_n=min(n,max(39*lists,cfg['embedding']['ann_train_vectors']))
        rng=np.random.default_rng(cfg['seed']);train=np.ascontiguousarray(e[rng.choice(n,train_n,replace=False)])
        cpu=faiss.IndexIVFFlat(faiss.IndexFlatIP(d),d,lists,faiss.METRIC_INNER_PRODUCT)
        cpu.cp.seed=cfg['seed'];cpu.cp.niter=20
        res=faiss.StandardGpuResources();res.setTempMemory(256*2**20)
        opt=faiss.GpuClonerOptions();opt.useFloat16=False
        index=faiss.index_cpu_to_gpu(res,gpu,cpu,opt)
        print('GPU IVF-Flat training',split,country,'lists',lists,'sample',train_n,flush=True)
        index.train(train);del train
        for s in range(0,n,32768):index.add(np.ascontiguousarray(e[s:s+32768]))
        index.nprobe=min(cfg['embedding']['ann_nprobe'],lists)
        cpu=faiss.index_gpu_to_cpu(index);faiss.write_index(cpu,str(root/'e5.ivfflat'))
        del index,res;gc.collect()
        resources=[]
        for _ in cfg['embedding']['gpu_ids']:
            r=faiss.StandardGpuResources();r.setTempMemory(256*2**20);resources.append(r)
        co=faiss.GpuMultipleClonerOptions();co.shard=False;co.useFloat16=False
        index=faiss.index_cpu_to_gpu_multiple_py(resources,cpu,co=co,gpus=cfg['embedding']['gpu_ids'])
        faiss.GpuParameterSpace().set_index_parameter(index,'nprobe',min(lists,cfg['embedding']['ann_nprobe']))
        k=cfg['retrieval']['route_k']['embedding'];search_k=min(n,cfg['embedding']['ann_search_k'])
        out=root/'lexical/embedding';out.mkdir(parents=True,exist_ok=True)
        ids=np.lib.format.open_memmap(out/'top_ids.npy',mode='w+',dtype=np.int64,shape=(len(q),k));ids[:]=-1
        scores=np.lib.format.open_memmap(out/'top_scores.npy',mode='w+',dtype=np.float32,shape=(len(q),k));scores[:]=0
        start=time.monotonic();batch=cfg['embedding']['ann_query_batch']
        for s in range(0,len(q),batch):
            qq=np.ascontiguousarray(q[s:s+batch]);_,ii=index.search(qq,search_k)
            for j,neighbors in enumerate(ii):
                neighbors=np.unique(neighbors[neighbors>=0]);ss=e[neighbors]@qq[j]
                order=np.lexsort((neighbors,-ss))[:k]
                ids[s+j,:len(order)]=neighbors[order];scores[s+j,:len(order)]=ss[order]
            if s==0 or s//batch%20==0:print('Dense queries',country,min(len(q),s+batch),'/',len(q),'elapsed',round(time.monotonic()-start),flush=True)
        ids.flush();scores.flush()
        write_json(out/'complete.json',{'route':'embedding','queries':len(q),'k':k})
        write_json(complete,{'rows':n,'nlist':lists,'nprobe':min(lists,cfg['embedding']['ann_nprobe']),'index':'IVF-Flat, float32, no PQ','approximate':True})
        del ids,scores,index,cpu,resources,e,q;gc.collect()
