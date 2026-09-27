# EC2 final entity-resolution run

This is a fresh source repository, not a patch to the Mac run. It retains all nine
Retriever V4 routes and the ordered 48-feature Matcher V2 schema. It replaces the
low-memory BM25 shortlist and PQ-compressed dense index, not the task or features.

**No measured full-dataset score accompanies this code. Neither 0.99 F0.5 nor a
one-to-two-hour from-scratch completion is guaranteed.** Building six full-vocabulary
TF-IDF indexes, embedding ~20M train/test targets, and scoring every test anchor are
substantial jobs. LightGBM fitting alone is not the runtime bottleneck. Use actual
progress rates; the first route's ETA is not an end-to-end ETA.

## Machine

Use EC2 `g6.24xlarge` (96 vCPU, 384 GiB, four NVIDIA L4 GPUs) or `g5.24xlarge`
(96 vCPU, 384 GiB, four A10G GPUs). `g6e.12xlarge` has 48 vCPU, 384 GiB and four
L40S GPUs: set `cpu_threads` to 16 before starting on that machine. Set 1000 GiB gp3
root storage, preferably 6000 IOPS / 500 MiB/s; these are proposed provisions, not
measured minimums. This repository requires a large RAM machine and CUDA; it will
not silently substitute CPU inference or drop retrieval routes.

EC2 GPU quota is separate from SageMaker and can itself be zero. Run `aws/check_ec2.sh`
from an authenticated AWS CLI BEFORE renting compute. `g6.24xlarge` needs 96 unused
vCPUs in the EC2 Running On-Demand G and VT quota in us-east-1, not 96 GPUs. EC2
instance-type offerings do not guarantee available capacity. Do not use Spot as
a quota workaround or depend on it for an uninterrupted six-hour deadline.

## Layout

```
~/student_resource/
  er_ec2_final/                # this repo, current directory for all commands
  data/processed/              # existing processed/v2 input
  data/raw/                    # original TSV for official validator/reproduction
  data/previous_samples/       # optional previously viewed sample IDs, for exclusion
  data/er_ec2_final_run/        # new run only
```

No command writes into raw/ or processed/. There is one model pipeline. Completed
stages can be reused with the same config; errors stop rather than choosing a
different model or silently dropping an algorithm. This is not a rollback system.

## AWS preparation

1. Region us-east-1. Run `bash aws/check_ec2.sh` with an authenticated AWS CLI.
2. IAM: create role `EntityResolutionEC2Role`, trusted service EC2. Attach the inline
   policy in `aws/s3-policy.json`. `aws/ec2-role-trust.json` is the trust document.
   No permission to write/delete raw or processed inputs is granted by that policy.
3. EC2 Launch Instance: select an AWS-owned x86_64 Deep Learning Base GPU Ubuntu 22.04
   AMI with NVIDIA drivers. `check_ec2.sh` prints the current official Base GPU AMI ID via SSM.
4. Select the instance above, On-Demand. Set gp3 storage. Require IMDSv2. Attach the
   IAM instance profile. Keep block-public access on S3. Allow SSH port 22 only from
   your IP (or use SSM); do not expose Jupyter publicly. Keep required outbound HTTPS.
5. Connect with the console-provided SSH command, typically:
   `ssh -i ~/.ssh/er-final.pem ubuntu@PUBLIC_IP`.
6. Check `nvidia-smi`, `free -h`, `df -h ~`. Review the EC2 console's current hourly
   rate and volume cost before launching. Stop/terminate after copying final artifacts;
   EBS volumes can continue to incur charges after compute stops.

## Git repository

On the Mac, unzip into a NEW directory and push only this source directory:

```bash
cd er_ec2_final
git init -b main
git add .
git commit -m "Final EC2 V4 and Matcher V2 pipeline"
git remote add origin YOUR_PRIVATE_GIT_REMOTE
git push -u origin main
```

Do not commit datasets, keys, .env files or model caches. The included `.gitignore`
excludes the common artifact formats. Authenticate to your private repository with a
deploy key/SSH agent or Git credential flow; never place a token in source code.

On EC2:

```bash
mkdir -p ~/student_resource
cd ~/student_resource
git clone YOUR_PRIVATE_GIT_REMOTE er_ec2_final
cd er_ec2_final
mkdir -p ../data/processed ../data/raw
aws sts get-caller-identity
aws s3 sync s3://sagemaker-ml-challenge-gururaj-2026/processed/v2/ ../data/processed/ --region us-east-1 --only-show-errors
aws s3 sync s3://sagemaker-ml-challenge-gururaj-2026/raw/ ../data/raw/ --region us-east-1 --only-show-errors
```

If retained on your Mac, upload previously inspected sample Parquet files to
`ec2-final-input/previous_samples/` and download to `../data/previous_samples/`.
The sampling manifest records which previous IDs were actually excluded; absence of
those files is not evidence that none were previously inspected.

## Environment: Linux x86_64, Python 3.11, CUDA

Use a new environment, not the Mac virtualenv copied to Linux.

```bash
bash scripts/bootstrap_ec2.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate erfinal
```

The bootstrap installs the conda GPU FAISS package and CUDA PyTorch together, then
installs pinned Python libraries, compiles/runs core tests, runs a small native
CUDA/Arrow/LightGBM smoke check and a synthetic full-pipeline integration test, and checks hardware. The synthetic pipeline uses deterministic vector fixtures instead of downloading E5; real encoder execution is first exercised during the actual encoding stage. Do not install `faiss-cpu` over
`faiss-gpu`. A newer NVIDIA host driver can run this CUDA 12.1 environment.

If conda is not present in the DLAMI, install Miniforge first (official installer):

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/Miniforge3.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
bash scripts/bootstrap_ec2.sh
conda activate erfinal
```

The dependency solver has not been executed in the response-generation environment.
Resolve any dependency or native smoke failure before spending time on full data.

## One train run

Create a persistent terminal session first:

```bash
tmux new -s er-final
```

Within tmux, activate erfinal and run from this repository:

```bash
mkdir -p logs
set -o pipefail
python -u scripts/run_plan.py train 2>&1 | tee logs/train.log
```

This performs, in order:

1. Check hardware/dependencies and validate all seven processed file row counts.
2. Sample 50k fit / 3k early / 5k calibration / 5k audit anchors, disjoint by S1.
3. Create country-specific target stores from ALL training targets, and query stores.
4. Resolve/download/pin the MIT E5 snapshot.
5. In parallel: full-split TF-IDF lexical indexes/search, and multi-GPU target/query
   E5 inference followed by GPU IVF-Flat search. No sparse route is disabled.
6. Fuse all nine routes; compute all 48 features only for retained candidates.
7. Fresh LightGBM fit, max 5000 rounds / early stopping 50, same 31-leaf architecture.
8. Calibration-only cap/threshold selection, then one audit without retuning on it.

View lexical progress in a second session:

```bash
tail -f logs/train-lexical.log
```

Detach tmux with Ctrl+B then D. Reattach with `tmux attach -t er-final`.
Disconnecting SSH is not a request to kill the running processes.

The code uses full-split TF-IDF fitting (not 20k sampled fitting), exact positive
WAND dot-product top-K (not BM25 shortlisting), and float32 IVF-Flat (not IVF-PQ).
IVF-Flat remains approximate because nprobe scans only some lists. There is no claim
that it is identical to exhaustive E5 top-K. Raw floats are retained and reranked.

LightGBM parameters: learning rate .04, leaves 31, min leaf 15, feature fraction .9,
L2 1, max_bin 255, deterministic seed. Row bagging is off, matching the effective
behavior of the pinned model. No old threshold or trained weights are reused.

## Review, then one test run

```bash
cat ../data/er_ec2_final_run/models/calibration_budgets.json
cat ../data/er_ec2_final_run/models/audit.json
```

Report both blocking recall and matching recall. If candidate-oracle F0.5 < .99,
.99 is impossible on that audit with those candidate sets, even for an oracle
classifier. The code prints that condition; it does not manufacture a score.
The country report is US/India only for labeled data; there is no labeled France
accuracy claim. Do not tune repeatedly on audit results.

After accepting the actual audit:

```bash
set -o pipefail
python -u scripts/run_plan.py test --accept-audit 2>&1 | tee logs/test.log
```

This freezes the selected model, runs all train-equivalent retrieval procedures on
all test countries INCLUDING FRANCE, scores exactly the submitted candidate lists,
exports both TSVs, validates coverage/IDs/duplicates/subset/threshold consistency,
and creates the final ZIP.

Individual equivalent commands are:

```bash
python -m erfast.cli accept --yes
python -m erfast.cli lexical --split test
python -m erfast.cli encode --split test --kind targets
python -m erfast.cli encode --split test --kind queries
python -m erfast.cli ann --split test
python -m erfast.cli mine --split test
python -m erfast.cli predict
python -m erfast.cli export
python -m erfast.cli package
```

Do not change config settings halfway through a run. All caches belong to the one
new run. The source does not modify, import or replace your previous deployed model.

## Candidate ranking update

The final candidate set is fixed BEFORE LightGBM scoring. The raw route union is
label-free, rank-fused and deduplicated. Calibration considers caps 16,24,32,48,64,96.
Choose the smallest mean candidate set within .001 absolute F0.5 and .002 absolute
candidate-recall of the calibration best. These tolerances are engineering choices,
NOT an invented official ranking formula. Exact-name hits are retrieved completely
and then participate in fusion; there is no silent exact-name top-16 cap.

The same finalized candidate set is used for test feature scoring and exported to
candidate_pairs.tsv. Never reduce that file to positive predictions after the fact.

## Deliverables and upload

```
../data/er_ec2_final_run/output/matching_results.tsv
../data/er_ec2_final_run/output/candidate_pairs.tsv
../data/er_ec2_final_run/output/validation_report.json
../data/er_ec2_final_run/gururaj_submission.zip
```

Run the organizer validator as well, using your original utils script:

```bash
python ../utils/validate_submission.py \
 --matching ../data/er_ec2_final_run/output/matching_results.tsv \
 --candidate ../data/er_ec2_final_run/output/candidate_pairs.tsv \
 --test-dir ../data/raw/test
```

The official validator was not in the uploaded code package, so copy your existing
one into ../utils. The internal export checks are included and always run.

Upload results without touching raw/processed:

```bash
aws s3 sync ../data/er_ec2_final_run/output/ s3://sagemaker-ml-challenge-gururaj-2026/ec2-final/output/ --region us-east-1 --only-show-errors
aws s3 cp ../data/er_ec2_final_run/gururaj_submission.zip s3://sagemaker-ml-challenge-gururaj-2026/ec2-final/gururaj_submission.zip --region us-east-1
```

Final ZIP contains both required TSVs, complete runnable code/config/environment
instructions, trained matcher, exact E5 revision, fixed sampled S1/ground-truth rows for reproducible training splits, and methodology populated with
actual settings/results. The organizer's template body is not available here;
transfer the generated methodology to it if its headings are mandatory.

## Reproduction from raw TSVs

Existing processed/v2 inputs save repeated work; they do not change the algorithm.
For a fresh reproduction, create an EMPTY processed output directory and run:

```bash
python scripts/preprocess_raw.py --raw ../data/raw --out ../data/processed
```

Then run the normal plan. This normalizes text locally with no external business
lookup. The original four source columns and truth semantics are preserved.

## No timing/score promises

There is no defensible 0.99 score or two-hour promise without running this backend on
these data and hardware. The latest previous candidate oracle was ~.9848, so matching
alone could not achieve .99 on those previous candidate sets. This run changes
retrieval, not merely GPU size. Use progress measurements and leave time for test
inference, export, upload and the portal submission. Training completion alone is
not submission completion. Do not use final minutes for extra architecture trials.
