from __future__ import annotations
import ctypes as C
import hashlib
import subprocess
from pathlib import Path
import numpy as np

_LIB = None

def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    src = Path(__file__).resolve().parents[2] / 'native' / 'wand.cpp'
    tag = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    out = src.parent / f'wand-{tag}.so'
    if not out.exists():
        # Build before spawning workers; native source is bundled, no remote code download.
        tmp = out.with_suffix('.tmp.so')
        subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-shared','-fPIC',str(src),'-o',str(tmp)],check=True)
        tmp.replace(out)
    lib=C.CDLL(str(out))
    p64=np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    p32=np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
    pf=np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    lib.er_wand.argtypes=[p64,p32,pf,pf,p64,p32,pf,C.c_int64,C.c_int,C.c_int,p64,pf]
    lib.er_wand.restype=C.c_int
    _LIB=lib
    return lib

def search(postings, maxima, query, k, threads=1):
    """Exact positive dot-product top-k; ties prefer smaller row IDs."""
    query=query.tocsr(copy=False)
    query.sum_duplicates(); query.sort_indices()
    ids=np.full((query.shape[0], k),-1,dtype=np.int64)
    scores=np.zeros(ids.shape,dtype=np.float32)
    if not query.shape[0]: return ids,scores
    rc=library().er_wand(
        np.ascontiguousarray(postings.indptr,dtype=np.int64),
        np.ascontiguousarray(postings.indices,dtype=np.int32),
        np.ascontiguousarray(postings.data,dtype=np.float32),
        np.ascontiguousarray(maxima,dtype=np.float32),
        np.ascontiguousarray(query.indptr,dtype=np.int64),
        np.ascontiguousarray(query.indices,dtype=np.int32),
        np.ascontiguousarray(query.data,dtype=np.float32),len(ids),k,threads,ids,scores)
    if rc: raise RuntimeError(f'Native WAND failed ({rc})')
    return ids,scores
