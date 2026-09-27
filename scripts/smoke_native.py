"""Small REAL native dependency test on EC2; no business data or model downloads."""
from pathlib import Path
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
import torch
import lightgbm as lgb
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from erfast.lexical import make_postings
from erfast.wand import search

if not torch.cuda.is_available():raise SystemExit('CUDA unavailable')
if not hasattr(faiss,'StandardGpuResources'):raise SystemExit('FAISS GPU missing')
rng=np.random.default_rng(7)
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'test.parquet';pq.write_table(pa.table({'x':[1,2,3],'names':['अर्बन','Gríll','ABC']}),p)
    assert pq.ParquetFile(p).metadata.num_rows==3
v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),dtype=np.float32)
t=v.fit_transform(['ABC Private Limited','ABC Pvt Ltd','Cozy Grill','Cozy Gríll'])
q=v.transform(['ABC Pvt Ltd','Cozy Grill']);post,m=make_postings(t);ids,s=search(post,m,q,2,2)
assert ids[0,0]==1 and ids[1,0]==2
x=rng.normal(size=(1000,48)).astype(np.float32);y=(x[:,0]+x[:,1]>0).astype(int)
model=lgb.train({'objective':'binary','verbosity':-1,'num_threads':2},lgb.Dataset(x,label=y),num_boost_round=20)
assert len(model.predict(x[:3]))==3
z=rng.normal(size=(1000,384)).astype(np.float32);faiss.normalize_L2(z)
res=faiss.StandardGpuResources();res.setTempMemory(64*2**20)
cpu=faiss.IndexIVFFlat(faiss.IndexFlatIP(384),384,4,faiss.METRIC_INNER_PRODUCT)
gpu=faiss.index_cpu_to_gpu(res,0,cpu);gpu.train(z);gpu.add(z);gpu.nprobe=4
_,ii=gpu.search(z[:10],2);assert np.array_equal(ii[:,0],np.arange(10))
print('PASS: Arrow, full TF-IDF, native WAND, CPU LightGBM, and CUDA FAISS IVF-Flat.')
