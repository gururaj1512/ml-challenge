# Testing status

Executed in the response-generation container:

- Python compile checks across all source and script files.
- 12 core tests passed (see core-test-output.txt).
- The WAND randomized test includes 40 trials x 30 queries against brute-force SciPy
  dot-product top-k, normalized and unnormalized positive sparse vectors, 1200 queries
  in total. Additional tests cover ties, all-zero queries, explicit lexical TF-IDF,
  mmap/CSR serialization, feature ordering, structured tokens and F0.5/budget logic.

Not executed here:

- The conda dependency solve on the target EC2 image.
- Real PyArrow/SQLite/full pipeline integration, because PyArrow was unavailable and
  package download network resolution was unavailable in this container.
- CUDA E5 model inference or FAISS GPU (no CUDA hardware/FAISS package here).
- The user's full train/test dataset and any runtime or F0.5 measurement.

On EC2 the bootstrap explicitly runs scripts/smoke_native.py and
scripts/smoke_pipeline.py before full data. The latter uses actual Arrow, SQLite,
TF-IDF, native WAND, GPU FAISS and LightGBM, but deterministic synthetic E5 vectors
(no model download during the smoke test). No claim is made that the E5 model itself
was exercised by this synthetic integration test.
