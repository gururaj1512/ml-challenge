from __future__ import annotations
import concurrent.futures as cf
import gc
import itertools
import time
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from .common import Config,check_disk,load_sparse,lock,read_json,rows_from_parquet,save_sparse,write_json
from .text import numeric_tokens
from . import wand

ROUTES=('name','stripped','accent','address_char','address_word','combined','numeric')
COLUMN={'name':'normalized_name','stripped':'stripped_name','accent':'accent_name','address_char':'normalized_address','address_word':'normalized_address','numeric':'normalized_address'}


def number_analyzer(text):return sorted(numeric_tokens(text))


def text_batches(path,route,batch_size=50000):
    cols=['normalized_name','normalized_address'] if route=='combined' else [COLUMN[route]]
    for b in rows_from_parquet(path,cols,batch_size):
        if route=='combined':yield [((n or '')+' '+(a or '')).strip() for n,a in zip(b[cols[0]],b[cols[1]])]
        else:yield [x or '' for x in b[cols[0]]]


def text_corpus(cfg,split,route,include_queries=True):
    for kind in ('targets','queries') if include_queries else ('targets',):
        for c in cfg.countries(split):
            for b in text_batches(cfg.country_dir(split,c)/f'{kind}.parquet',route,cfg['read_batch_rows']):yield from b


def transformed_corpus(vectorizer,path,route,rows):
    parts=[vectorizer.transform(b).tocsr() for b in text_batches(path,route,rows)]
    if not parts:return sparse.csr_matrix((0,len(vectorizer.vocabulary_)),dtype=np.float32)
    result=sparse.vstack(parts,format='csr',dtype=np.float32)
    del parts
    result.sum_duplicates();result.sort_indices()
    return result


def make_postings(x):
    p=x.T.tocsr();p.sum_duplicates();p.sort_indices()
    maxima=np.zeros(p.shape[0],dtype=np.float32)
    active=np.flatnonzero(np.diff(p.indptr))
    if len(active):maxima[active]=np.maximum.reduceat(p.data,p.indptr[active])
    return p,maxima


def _route(args):
    path,split,route=args;cfg=Config(path);base=cfg.work/'vectorizers'/split/route;base.mkdir(parents=True,exist_ok=True)
    with lock(base/'build.lock'):
        if (base/'complete.json').exists():return
        check_disk(cfg);start=time.monotonic()
        if (base/'vectorizer.joblib').exists():
            vec=joblib.load(base/'vectorizer.joblib')
        elif route=='numeric':
            vec=CountVectorizer(analyzer=number_analyzer,binary=True,dtype=np.float32)
            print('Fit full-target numeric vocabulary',split,flush=True)
            x=vec.fit_transform(text_corpus(cfg,split,route,False)).tocsr()
            df=np.bincount(x.indices,minlength=x.shape[1]);idf=(np.log((1+x.shape[0])/(1+df))+1).astype(np.float32)
            np.save(base/'idf.npy',idf);del x,df;gc.collect()
            joblib.dump(vec,base/'vectorizer.joblib')
        else:
            r=cfg['retrieval'];limit=r['name_max_features']
            if route=='address_char':limit=r['address_char_max_features']
            if route=='address_word':limit=r['address_word_max_features']
            if route=='combined':limit=r['combined_max_features']
            kw=dict(analyzer='char_wb',ngram_range=(3,5),min_df=1,sublinear_tf=True,max_features=limit,dtype=np.float32)
            if route=='address_word':kw.update(analyzer='word',ngram_range=(1,2),token_pattern=r'(?u)\b[\w/-]+\b')
            vec=TfidfVectorizer(**kw)
            print('Fit full-split TF-IDF',split,route,'(no sampling)',flush=True)
            vec.fit(text_corpus(cfg,split,route,True));joblib.dump(vec,base/'vectorizer.joblib')
        idf=np.load(base/'idf.npy') if route=='numeric' else None
        for country in cfg.countries(split):
            root=cfg.country_dir(split,country);out=root/'lexical'/route
            if (out/'complete.json').exists():continue
            check_disk(cfg)
            x=transformed_corpus(vec,root/'targets.parquet',route,cfg['read_batch_rows'])
            save_sparse(out/'targets',x)
            post,maxima=make_postings(x);save_sparse(out/'postings',post)
            np.save(out/'maxima.npy',maxima);del post,maxima
            q=transformed_corpus(vec,root/'queries.parquet',route,cfg['read_batch_rows'])
            if route=='numeric':q=q.multiply(idf).tocsr()
            save_sparse(out/'queries',q)
            n,m=x.shape[0],q.shape[0];del x;gc.collect()
            p=load_sparse(out/'postings');mx=np.load(out/'maxima.npy',mmap_mode='r')
            k=cfg['retrieval']['route_k'][route]
            rid=np.lib.format.open_memmap(out/'top_ids.npy',mode='w+',dtype=np.int64,shape=(m,k))
            score=np.lib.format.open_memmap(out/'top_scores.npy',mode='w+',dtype=np.float32,shape=(m,k))
            t=time.monotonic()
            for s in range(0,m,cfg['query_batch_rows']):
                a,b=wand.search(p,mx,q[s:s+cfg['query_batch_rows']],k,cfg['cpu_threads'])
                rid[s:s+len(a)]=a;score[s:s+len(a)]=b
                if s==0 or (s//cfg['query_batch_rows'])%20==0:
                    done=min(m,s+len(a));rate=done/max(.001,time.monotonic()-t)
                    print(f'{split}/{country}/{route} {done:,}/{m:,} queries; {rate:.1f} q/s; ETA current route {(m-done)/max(rate,.001)/60:.1f} min',flush=True)
            rid.flush();score.flush();del rid,score,p,mx,q
            write_json(out/'complete.json',{'rows':n,'queries':m,'route':route,'k':k,'exact_positive_dot_topk':True})
        write_json(base/'complete.json',{'split':split,'route':route,'seconds':time.monotonic()-start,'vocabulary':len(vec.vocabulary_)})


def build(cfg,split):
    wand.library() # build native code once in parent; worker processes only load it
    tasks=[(str(cfg.path),split,r) for r in ROUTES]
    with cf.ProcessPoolExecutor(max_workers=cfg['lexical_workers']) as pool:list(pool.map(_route,tasks))


class PairCosine:
    def __init__(self,root):
        self.q={};self.t={}
        for r in ('name','address_char','address_word'):
            self.q[r]=load_sparse(root/'lexical'/r/'queries');self.t[r]=load_sparse(root/'lexical'/r/'targets')
    def values(self,qid,tids):
        return {r:np.asarray(self.t[r][tids].dot(self.q[r][qid].T).toarray()).ravel() for r in self.q}
