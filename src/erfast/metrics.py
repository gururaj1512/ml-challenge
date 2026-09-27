from __future__ import annotations
import math
import numpy as np


def entity_f05(truth, predictions) -> float:
    truth, predictions = set(truth), set(predictions)
    if not truth:
        return float(not predictions)
    tp = len(truth & predictions)
    fp = len(predictions-truth)
    fn = len(truth-predictions)
    return 1.25*tp/(1.25*tp+fp+.25*fn) if tp else 0.0


def evaluate(queries: list[dict], pair_rows: list[dict], probabilities: np.ndarray,
             budget: int, threshold: float) -> dict:
    # All Source-1 entities are present even when there are no retrieved candidate rows.
    predictions = {q['entity_id']: set() for q in queries}
    candidates = {q['entity_id']: set() for q in queries}
    for row, p in zip(pair_rows, probabilities):
        if row['candidate_rank'] > budget:
            continue
        qid, tid = row['source1_entity_id'], row['target_entity_id']
        candidates[qid].add(tid)
        if p >= threshold:
            predictions[qid].add(tid)
    scores, ns_scores, singleton = [], [], []
    true_links = kept_links = tp = fp = 0
    complete = non_singletons = 0
    country_scores = {}
    for q in queries:
        qid = q['entity_id']
        truth = set(q['truth'])
        cand = candidates[qid]
        pred = predictions[qid]
        score = entity_f05(truth, pred)
        scores.append(score)
        country_scores.setdefault(q['country'], []).append(score)
        true_links += len(truth)
        kept_links += len(truth & cand)
        tp += len(truth & pred)
        fp += len(pred-truth)
        if truth:
            non_singletons += 1
            complete += int(truth.issubset(cand))
            ns_scores.append(score)
        else:
            singleton.append(int(not pred))
    counts = [len(candidates[q['entity_id']]) for q in queries]
    return {
        'entities': len(queries), 'true_links': true_links, 'candidate_true_links': kept_links,
        'tp': tp, 'fp': fp, 'fn_end_to_end': true_links-tp, 'fn_blocking': true_links-kept_links,
        'fn_matching': kept_links-tp,
        'macro_f05': float(np.mean(scores)),
        'non_singleton_macro_f05': float(np.mean(ns_scores)) if ns_scores else None,
        'singleton_accuracy': float(np.mean(singleton)) if singleton else None,
        'singleton_entities': len(singleton),
        'candidate_link_recall': kept_links/true_links if true_links else None,
        'complete_non_singleton_recall': complete/non_singletons if non_singletons else None,
        'pair_precision': tp/(tp+fp) if tp+fp else 0.,
        'conditional_matcher_recall': tp/kept_links if kept_links else 0.,
        'end_to_end_link_recall': tp/true_links if true_links else 0.,
        'mean_candidates': float(np.mean(counts)), 'p95_candidates': float(np.percentile(counts,95)),
        'max_candidates': int(max(counts, default=0)), 'total_candidates': int(sum(counts)),
        'candidate_budget': budget, 'threshold': float(threshold),
        'by_country_macro_f05': {c: float(np.mean(s)) for c,s in country_scores.items()},
        '_entity_scores': scores,
    }


def choose_budget(rows: list[dict], max_f05_loss: float, max_recall_loss: float) -> dict:
    # No invented combined organizer scoring formula. Favor compactness within explicit tolerances.
    best_f = max(r['macro_f05'] for r in rows)
    best_r = max((r['candidate_link_recall'] or 0.) for r in rows)
    feasible = [r for r in rows if r['macro_f05'] >= best_f-max_f05_loss
                and (r['candidate_link_recall'] or 0.) >= best_r-max_recall_loss]
    if not feasible:
        # Can happen if the highest-recall and highest-F scores conflict beyond tolerances.
        raise RuntimeError('No candidate budget meets both tolerances. Review calibration table; do not silently choose.')
    return min(feasible, key=lambda r: (r['mean_candidates'], -r['macro_f05'], r['candidate_budget']))


def bootstrap_interval(scores, seed: int, n: int) -> list[float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(scores)
    if not len(a):
        return [0.,0.]
    values = [float(a[rng.integers(0,len(a),len(a))].mean()) for _ in range(n)]
    return [float(x) for x in np.quantile(values,[.025,.975])]


def public_metrics(result: dict) -> dict:
    return {k:v for k,v in result.items() if not k.startswith('_')}
