from __future__ import annotations
import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


class Config:
    def __init__(self,path):
        self.path=Path(path).resolve();self.root=self.path.parent;self.d=read_json(self.path)
        self.work=(self.root/self.d['work_dir']).resolve()
        self.processed=(self.root/self.d['processed_dir']).resolve()
        self.raw=(self.root/self.d['raw_dir']).resolve()
        self.fingerprint=digest(self.d)
    def __getitem__(self,k):return self.d[k]
    def country_dir(self,split,country):
        token=hashlib.sha256(country.encode()).hexdigest()[:16]
        return self.work/'corpus'/split/token
    def countries(self,split):return read_json(self.work/f'{split}.json')['countries']


def bind_run(cfg):
    cfg.work.mkdir(parents=True,exist_ok=True)
    p=cfg.work/'run.json'
    if p.exists() and read_json(p)['config_hash']!=cfg.fingerprint:
        raise RuntimeError('Configuration changed. Use a NEW work_dir for a different experiment.')
    if not p.exists():write_json(p,{'config_hash':cfg.fingerprint,'config':cfg.d,'created_at':time.time()})


@contextlib.contextmanager
def lock(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Another writer is using {path}')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def check_disk(cfg):
    free=shutil.disk_usage(cfg.work).free/2**30
    if free<cfg['minimum_free_gib']:
        raise RuntimeError(f'Free disk {free:.1f} GiB below reserve. No data was deleted.')


def connect(path,readonly=False):
    path=Path(path)
    if readonly:
        con=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        con.execute('PRAGMA query_only=ON')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);con=sqlite3.connect(path)
        con.execute('PRAGMA journal_mode=DELETE');con.execute('PRAGMA synchronous=FULL')
    con.execute('PRAGMA cache_size=-65536');con.row_factory=sqlite3.Row
    return con


def load_sparse(folder):
    from scipy.sparse import csr_matrix
    folder=Path(folder);shape=tuple(read_json(folder/'shape.json')['shape'])
    return csr_matrix((np.load(folder/'data.npy',mmap_mode='r'),
        np.load(folder/'indices.npy',mmap_mode='r'),np.load(folder/'indptr.npy',mmap_mode='r')),shape=shape,copy=False)


def save_sparse(folder,x):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    x=x.tocsr();x.sum_duplicates();x.sort_indices()
    if x.shape[1]>=2**31:raise ValueError('Column count exceeds int32 native search support')
    np.save(folder/'data.npy',np.asarray(x.data,dtype=np.float32))
    np.save(folder/'indices.npy',np.asarray(x.indices,dtype=np.int32))
    np.save(folder/'indptr.npy',np.asarray(x.indptr,dtype=np.int64))
    write_json(folder/'shape.json',{'shape':list(x.shape),'nnz':int(x.nnz)})


def atomic_parquet(path,table):
    import pyarrow.parquet as pq
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    pq.write_table(table,tmp,compression='zstd');tmp.replace(path)


def rows_from_parquet(path,columns=None,batch=50000):
    import pyarrow.parquet as pq
    for x in pq.ParquetFile(path).iter_batches(batch_size=batch,columns=columns,use_threads=False):
        yield x.to_pydict()


def records(con,rids,table='records'):
    ans={}
    ids=sorted(set(int(x) for x in rids))
    for s in range(0,len(ids),500):
        a=ids[s:s+500]
        if not a:continue
        for r in con.execute(f'SELECT * FROM {table} WHERE rid IN ({",".join("?" for _ in a)})',a):
            row=dict(r);ans[row['rid']]=row
    if len(ans)!=len(ids):raise ValueError('A candidate row is absent from its target store')
    return ans
