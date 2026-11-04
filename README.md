# Source-balanced Capped Fisher

Code for **Source-balanced Capped Fisher: Bounded Evidence Aggregation for User Inference in Language Models**, by Yang Liu, Ran Zou, and Wanrong Zhu.

This repository reproduces the main four-domain experiment: Tables 1–3 and Figure 3, together with the paired comparisons and cap ablation in Appendix C. It includes public-data retrieval, text processing, fixed document splits, GPT-Neo fine-tuning, document scoring, SCF, and the comparator aggregations. The separate temporal-cutoff experiment in Appendix C.7 is outside this package's scope.

## Quick reproduction from document scores

Run commands from the repository root. For the quick route, use Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
python reproduce.py --output results
python verify.py --results results
```

The supplied `scores/` files contain the original document scores. This route recomputes reference tails, all aggregations, conformal decisions, and all 500 source-role partitions. `verify.py` compares every displayed entry of Tables 1–3 with the paper. This does not rerun model training.

## Reproduction from raw data

The original calculation uses Linux, Python 3.11, PyTorch 2.4.1 with CUDA 12.1, Transformers 4.51.3, and an NVIDIA A100 GPU. Parameters and optimizer state use float32; forward calculations use bfloat16 autocast. A different GPU or software stack may change numerical results.

Fixed seeds control random choices but do not guarantee bitwise-identical training across hardware or software environments. Retraining may therefore change document scores and AUROC/TPR/FPR. `verify.py` checks agreement at the paper's printed precision: a mismatch after retraining should be assessed by its magnitude and effect on the method comparisons. The supplied-score route is the reference for reproducing the printed tables exactly.

```bash
set -e
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements-gpu.txt

for domain in reddit cc_news enron chromium; do
    python prepare.py --domain "$domain" --download
    python fit.py --domain "$domain" --output runs
done

python reproduce.py --data runs/scores --output results-from-raw
python verify.py --results results-from-raw
```

`prepare.py` retrieves the original public corpus versions and rebuilds model inputs from the split manifests. Raw texts and checkpoints remain local and are not included in this repository. Allow roughly 60 GB for the raw corpora, additional space for prepared inputs and four checkpoints, and 32 GB of system RAM. The six Reddit archives account for most of the download. Original A100 training took approximately ten GPU-hours for CC News and 1.5 hours for Enron; Reddit and Chromium are much smaller. Downloading, preprocessing, and scoring require additional time.

To use existing raw files, omit `--download` and pass `--raw /path/to/domain/files`. An existing copy of the pinned model can be supplied through `prepare.py --tokenizer /path/to/model` and `fit.py --model /path/to/model`. CPU execution is supported for debugging, but does not reproduce the paper's GPU precision. Run each training command in a fresh output directory; `fit.py` refuses to overwrite an existing domain run.

## Experimental protocol

Every method receives the same token-average likelihood-difference score and the same query documents. Larger scores indicate stronger participation evidence. Training and checkpoint-validation documents are excluded from the evaluated queries.

| Domain | Audit sources | Participants | Reference K | SCF calibration L | Queries/source | Training seed |
|---|---:|---:|---:|---:|---:|---:|
| Reddit | 267 | 84 | 40 | 59 | 16 | 20260914 |
| CC News | 2,709 | 1,348 | 40 | 59 | 1–16 | 171470688381003029 |
| Enron | 141 | 70 | 25 | 25 | 1–16 | 7129716279750356844 |
| Chromium | 199 | 50 | 40 | 59 | 16 | 20260914 |

- **Mean, Max, Min:** aggregate the original document scores.
- **RaMIA:** average the largest `max(1, floor(0.75*m))` scores in a bag of size `m`, matching the paper's specified trimmed-mean comparator.
- **Uncapped Fisher and SCF:** fit the same source-balanced inclusive upper-tail reference map, apply the same finite-reference smoothing, and average negative log tails. SCF caps each contribution at `-log(0.05)`; uncapped Fisher omits this cap.

SCF and uncapped Fisher use `K` controls to fit the reference map and `L` disjoint controls for calibration. Raw-score baselines use all `K+L` controls for calibration. All methods have the same total control budget and test targets. The 500 partitions change only the null sources' reference, calibration, and test roles. They do not retrain the model or resample documents and are not independent training replications.

`config/experiment.json` fixes model and training parameters. Ten epoch endpoints are evaluated, and the earliest minimum validation loss selects the checkpoint. Its `epoch` field records the paper's selected epoch for verification; it does not override validation selection. [Data construction](config/data_construction.md) describes each corpus, preprocessing, and split format.

## Files and outputs

- `prepare.py`: public-data retrieval and reconstruction of token inputs.
- `fit.py`: model fine-tuning, checkpoint selection, and likelihood-difference scoring.
- `scf.py`: aggregation, AUROC, and conservative conformal ranks.
- `reproduce.py`: matched-partition experiments and summaries.
- `verify.py`, `expected/`: comparison against the printed main tables.
- `splits/`: raw-record split manifests and the 500 source-role partitions.
- `scores/`: original numerical scores for the quick reproduction route.
- `tests/`: checks of tail weighting, ties, trimming, calibration, and preprocessing.

Generated results are not tracked. Outputs include `table1_auroc.csv`, `table2_cap_rates.csv`, `table3_audit_utility.csv`, `paired_counts.csv`, `alpha_auroc.csv`, `alpha_fpr.csv`, `alpha_sensitivity.pdf`, and `tables.md`. Per-partition counts are saved under `partitions/`. Training saves one selected checkpoint, `validation.csv`, and a compact `run.json` containing the configuration and runtime environment.

## License

The code is released under the [MIT License](LICENSE). Original corpora are obtained from their providers and remain subject to their respective terms; this repository does not redistribute their text.
