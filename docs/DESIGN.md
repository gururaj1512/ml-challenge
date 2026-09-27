# Design and limits

## Source-derived decisions

The task is anchored one-to-many matching: S1 is deduplicated, each S1 has zero or
many S2/S3 links. Every test S1, including France, must be output. The matching score
is macro per-S1 F0.5, including correct-empty singleton credit. Candidate output is
the actual set fed to final matcher inference. The user's update adds candidate-set
compactness to the final review without publishing a numerical combined objective.

Pinned V4 has nine routes: exact normalized name; name character TF-IDF; address
character TF-IDF; address word/bigram TF-IDF; numeric-token retrieval; stripped-name
character TF-IDF; Latin-accent-folded character TF-IDF; combined name/address
character TF-IDF; multilingual E5. All remain enabled. Route Ks remain 10 for primary
lexical/numeric routes, 5 for auxiliary stripped/accent/combined, and 20 for E5.
The exact-name list is fully enumerated before final rank fusion, rather than
silently keeping only 16 exact-name records.

Pinned Matcher V2 has 48 features. FEATURES in src/erfast/features.py preserves that
ordered schema. The 500-tree prototype was not evidence of an optimal round count;
the user's cached refits improved calibration and early-stopped at 2679 under a 5000
limit. This run uses one fresh max-5000 fit and patience 50. It does not force exactly
2679 rounds or reuse an old threshold on new retrieval distributions.

## New engineering choices (not measured results)

1. Explicit TF-IDF vocabulary fitted on ALL target records plus the current S1 query
   corpus for the split; no labels involved. Unlike the benchmark's 150k/200k feature
   caps, this large-memory configuration sets max_features=None and min_df=1. Rare
   word/character features are not dropped by a sampled vocabulary or feature cap.
   Vocabulary/IDF are global within each train/test split; country filtering is applied
   through separate target postings. The additional features change TF-IDF values,
   so the matcher is retrained. Train and test IDF are separately fitted, as in the
   transductive text indexing of the original benchmark, never using test labels.
2. Sorted inverted postings and a positive-score exact WAND implementation replace
   exhaustive sparse matrix materialization and BM25 shortlisting. Every nonzero
   query term is retained; there is no max-query-terms shortcut. Top-k is exact over
   the stored float32 TF-IDF representation, subject to normal numerical rounding.
   Similarity-zero records do not fill unused slots. Tie order is ascending target
   row ID, not arbitrary np.argpartition boundary order.
3. E5 remains pretrained/frozen and uses query:/passage: prefixes, length 512, L2
   normalization, float32 inference and stored vectors. Four independent CUDA
   processes encode disjoint row ranges. These are not four models in an ensemble.
4. FAISS GPU IndexIVFFlat, not IndexIVFPQ: no product quantization of target vectors.
   Up to 8192 coarse lists, nprobe 128, training sample up to 524288 (at least 39/list),
   100 returned neighbors, original-vector cosine reranking, keep 20. IVF still can
   miss exact nearest neighbors. It preserves the dense route, not exhaustive E5's
   mathematical candidate identity. GPU replicas parallelize query search.
5. Label-free RRF (constant 60) combines route ranks. Calibration selects a compact
   final budget among 16/24/32/48/64/96 within explicit F0.5/recall tolerances. This is
   not the organizer's unspecified final ranking formula. Internal route pools and
   final candidate counts are reported honestly. No model scores a hidden larger test
   set and then reports only positive predictions as candidates.
6. 50k fit, 3k early, 5k calibration, 5k audit anchors. All targets are indexed. We do
   not claim every labeled S1 is used for supervised fitting; that would be a different
   and more expensive run. Known previous sample files can be excluded and the actual
   exclusion list is recorded. No learned country one-hot; France remains included.
7. Empty/empty strings do not become strong matching evidence. The structured token
   tokenizer handles 5A1 and G-1 and spaced 14 / 15. These are explicit corrections to
   old helper behavior, not an assertion that every feature value is bit-identical.
   Name ambiguity frequency is country-local in the index store.
8. No target-uniqueness postprocessing is imposed. The empirical training uniqueness
   observation is not used to force one match per source or alter submitted candidates.

## WAND correctness sketch

Each positive query-term contribution has upper bound q[t]*max_doc_weight[t]. At
any step the posting iterators are sorted by their next document ID. Documents
before a pivot cannot receive contributions from terms whose next document is after
that pivot. When the sum of available upper bounds is smaller than the current
kth-best score, those document ranges can be skipped without changing the top-k.
Candidates that could exceed that bound receive the full dot-product score. The
implementation adds a small conservative floating-point tolerance to the bound,
never lowers it to prune additional candidates. The candidate heap is O(k); target
postings are reusable, not re-vectorized for every query. Worst-case query work can
still be high for common terms/weak bounds. WAND is not a runtime guarantee.

## Implementation differences from the last Mac package

Removed: BM25 shortlist=48, max_query_terms=32, exact-name top-16, IVF-PQ vector
compression, 500-round training ceiling, single-query GPU/dense scans. Restored:
separate V4 name/address/auxiliary routes with exact full-vocabulary lexical scoring.
Added: native indexed pruning, multicore query search, multi-GPU embedding and
IVF-Flat, parallel shard feature extraction. No algorithm is conditionally dropped
when the machine is slow. A stage error stops rather than returning a cheaper model.

## Time and score honesty

The previous audit was ~0.9406; its candidate oracle was ~0.9848. Therefore, .99 was
not attainable on that audit without changing candidate generation, even with an
oracle matcher. Neither larger hardware nor a new tree limit proves .99 accuracy.
This backend's audit is unknown until run. The prior .9856 small-corpus result is not
a full-corpus score. New model fitting may be quick once pairs exist; full indexing,
encoding, retrieval and inference have not been timed on the recommended EC2 host.
Do not interpret a 1-2 hour requested training budget as a measured end-to-end SLA.

## Primary technical references

- WAND: Broder et al., Efficient query evaluation using a two-level retrieval process,
  CIKM 2003, DOI 10.1145/956863.956944; IBM Research publication page.
- scikit-learn TfidfVectorizer official API (smooth IDF, sublinear TF, explicit vocabulary).
- FAISS official Faiss-on-the-GPU wiki and INSTALL.md (IVF-Flat, GPU replicas, conda packages).
- intfloat/multilingual-e5-small official model card (MIT, prefixes, pooling/normalization).
- LightGBM 4.6.0 official early-stopping and parameter documentation.
- AWS EC2 accelerated-computing specifications, EC2 instance quotas and official DLAMI SSM query.

These are library/infrastructure references, not sources of external business data.
