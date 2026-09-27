from __future__ import annotations
import json
import time
import numpy as np
from .common import atomic_parquet,read_json,sha,write_json
from .features import FEATURES
from .metrics import bootstrap_interval,choose_budget,evaluate,public_metrics,entity_f05


def shards(cfg,split):
    result=[]
    for c in cfg.countries(split):
        root=cfg.work/'pairs'/split/cfg.country_dir(split,c).name
        if not(root/'complete.json').exists():raise RuntimeError(f'Candidate features incomplete: {split}/{c}')
        result.extend(sorted(p.parent for p in root.glob('[0-9]*/complete.json')))
    return result


def load_role(cfg,role):
    import pyarrow.parquet as pq
    xs=[];ps=[];qs=[]
    for s in shards(cfg,'train'):
        pair=pq.ParquetFile(s/'pairs.parquet').read().to_pylist()
        ix=np.array([i for i,r in enumerate(pair) if r['role']==role],dtype=np.int64)
        if len(ix):xs.append(np.load(s/'X.npy',mmap_mode='r')[ix]);ps.extend(pair[i] for i in ix)
        for q in pq.ParquetFile(s/'queries.parquet').read().to_pylist():
            if q['role']==role:qs.append({'entity_id':q['source1_entity_id'],'country':q['country'],'truth':q['truth'],'raw_count':q['raw_count'],'raw_true_links':q['raw_true_links']})
    x=np.concatenate(xs) if xs else np.empty((0,len(FEATURES)),dtype=np.float32)
    if len(qs)!=cfg['sample'][role]:raise ValueError('Missing/duplicate sampled queries in feature output')
    if len({q['entity_id'] for q in qs})!=len(qs):raise ValueError('Duplicate query')
    return x,ps,qs


def oracle(q,p,budget):
    truth={x['entity_id']:set(x['truth']) for x in q};c={x['entity_id']:set() for x in q}
    for r in p:
        if r['candidate_rank']<=budget:c[r['source1_entity_id']].add(r['target_entity_id'])
    return float(np.mean([entity_f05(x['truth'],c[x['entity_id']]&truth[x['entity_id']]) for x in q]))


def fit(cfg):
    import lightgbm as lgb
    root=cfg.work/'models';root.mkdir(parents=True,exist_ok=True)
    if (root/'fit.json').exists():return
    x,p,_=load_role(cfg,'fit');v,vp,_=load_role(cfg,'early')
    y=np.asarray([r['label'] for r in p],dtype=np.int8);yv=np.asarray([r['label'] for r in vp],dtype=np.int8)
    if len(np.unique(y))!=2:raise ValueError('Training sample lacks both classes')
    h=cfg['matcher'];params={k:h[k] for k in ['learning_rate','num_leaves','min_data_in_leaf','feature_fraction','lambda_l2','max_bin','num_threads']}
    params.update(objective='binary',metric='binary_logloss',seed=cfg['seed'],deterministic=True,force_col_wise=True,bagging_freq=0,bagging_fraction=1.0,verbosity=-1)
    train=lgb.Dataset(x,label=y,feature_name=FEATURES);early=lgb.Dataset(v,label=yv,reference=train)
    hist={};start=time.monotonic()
    model=lgb.train(params,train,num_boost_round=h['num_boost_round'],valid_sets=[early],valid_names=['early'],callbacks=[lgb.early_stopping(h['early_stopping_rounds']),lgb.log_evaluation(100),lgb.record_evaluation(hist)])
    model.save_model(str(root/'matcher.txt'),num_iteration=model.best_iteration)
    write_json(root/'fit.json',{'model_sha256':sha(root/'matcher.txt'),'config':cfg.fingerprint,'best_iteration':model.best_iteration,'fit_pairs':len(y),'early_pairs':len(yv),'feature_names':FEATURES,'parameters':params,'round_limit':h['num_boost_round'],'seconds':time.monotonic()-start})
    write_json(root/'learning_curve.json',hist)


def load_model(cfg):
    import lightgbm as lgb
    p=cfg.work/'models/matcher.txt';m=read_json(cfg.work/'models/fit.json')
    if sha(p)!=m['model_sha256'] or cfg.fingerprint!=m['config']:raise ValueError('Model/config provenance mismatch')
    return lgb.Booster(model_file=str(p))


def calibrate(cfg):
    root=cfg.work/'models';model=load_model(cfg);x,p,q=load_role(cfg,'calibrate')
    probs=model.predict(x,num_threads=cfg['matcher']['num_threads']);h=cfg['matcher']
    thresholds=np.round(np.arange(h['threshold_start'],h['threshold_end']+1e-6,h['threshold_step']),4)
    rows=[]
    for b in cfg['retrieval']['candidate_budgets']:
        best=None
        for t in thresholds:
            r=evaluate(q,p,probs,b,float(t))
            if best is None or (r['macro_f05'],t)>(best['macro_f05'],best['threshold']):best=r
        best=public_metrics(best);best['candidate_oracle_macro_f05']=oracle(q,p,b);rows.append(best)
        print(f"cap={b} threshold={best['threshold']:.2f} macroF0.5={best['macro_f05']:.6f} candidateRecall={best['candidate_link_recall']:.6f} meanC={best['mean_candidates']:.2f}",flush=True)
    r=cfg['retrieval'];selected=choose_budget(rows,r['f05_tolerance'],r['recall_tolerance'])
    write_json(root/'calibration_budgets.json',rows)
    write_json(root/'selection.json',{'candidate_budget':selected['candidate_budget'],'threshold':selected['threshold'],'model_sha256':sha(root/'matcher.txt'),'config':cfg.fingerprint,'calibration':selected})


def audit(cfg):
    root=cfg.work/'models';m=load_model(cfg);s=read_json(root/'selection.json')
    x,p,q=load_role(cfg,'audit');prob=m.predict(x,num_threads=cfg['matcher']['num_threads'])
    result=evaluate(q,p,prob,s['candidate_budget'],s['threshold'])
    interval=bootstrap_interval(result['_entity_scores'],cfg['seed'],500)
    result=public_metrics(result);result['candidate_oracle_macro_f05']=oracle(q,p,s['candidate_budget'])
    result['candidate_oracle_reaches_0_99']=result['candidate_oracle_macro_f05']>=0.99
    result['bootstrap95']=interval;result['model_sha256']=sha(root/'matcher.txt');result['selection_sha256']=sha(root/'selection.json')
    result['scope']='S1-disjoint audit; no labeled France estimate; no guaranteed leaderboard score'
    write_json(root/'audit.json',result);print(json.dumps(result,indent=2))


def accept(cfg):
    root=cfg.work/'models';a=read_json(root/'audit.json');s=read_json(root/'selection.json')
    if a['model_sha256']!=sha(root/'matcher.txt') or a['selection_sha256']!=sha(root/'selection.json'):raise ValueError('Audit is stale')
    write_json(root/'deployment.json',{'model_sha256':a['model_sha256'],'selection_sha256':a['selection_sha256'],'candidate_budget':s['candidate_budget'],'threshold':s['threshold'],'config':cfg.fingerprint})
