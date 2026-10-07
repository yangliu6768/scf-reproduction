# Source-balanced Capped Fisher

Code and numerical data for reproducing the SCF aggregation experiments on Reddit, CC News, Enron, and Chromium.

The supplied document scores and source partitions reproduce Tables 1–3, Figure 3, and the paired-count and cap-ablation tables in the appendix. The computation starts from document scores; model training and raw-text scoring are described in [data construction](config/data_construction.md).

## Run

Use Python 3.10–3.12. No GPU or model download is needed.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

To run one domain:

```sh
python reproduce.py --domain enron
```

To check the numerical routines:

```sh
python -m unittest test_scf.py
```

## Protocol

Every method uses the same document scores, query bags, and test sources. The document score is the token-average log-likelihood under the updated model minus that under the initial model; larger values indicate stronger participation evidence. Both models use the GPT-Neo-125M architecture.

| Domain | Source | Participant sources | Null sources | Reference K | Calibration L | Null test sources | Queries/source |
|---|---|---:|---:|---:|---:|---:|---:|
| Reddit | Account | 84 | 183 | 40 | 59 | 84 | 16 |
| CC News | Web domain | 1,348 | 1,361 | 40 | 59 | 1,262 | 1–16 |
| Enron | Mailbox owner | 70 | 71 | 25 | 25 | 21 | 1–16 |
| Chromium | Developer | 50 | 149 | 40 | 59 | 50 | 16 |

There are 500 source-role partitions per domain. Training membership, model weights, query documents, and their scores remain fixed. Within each partition, the first K null sources fit the reference map, the next L calibrate SCF, and the remaining null sources join all participating sources in the test panel. Raw-score baselines use all K+L controls for calibration.

Mean, Max, and Min aggregate the raw scores. The RaMIA-style statistic averages the largest `max(1, floor(0.75*m))` scores in a bag of size m. For SCF, each reference source receives equal total weight. With N reference documents, the smoothed tail is

$$
\widehat Q(s)=\frac{1+\frac{N}{K}\sum_{v=1}^{K}\frac{1}{m_v}\sum_{j=1}^{m_v}\mathbf 1\{S_{vj}\ge s\}}{N+1}.
$$

SCF averages `-log(max(Q, alpha))` within each source. The main cap is 0.05; the ablation uses 0, 0.01, 0.02, 0.05, and 0.10. A cap of zero retains reference smoothing and gives uncapped Fisher.

The source p-value is `(1 + number of calibration scores >= target score) / (calibration size + 1)`. Decisions use nominal levels 0.05, 0.10, and 0.20. AUROC gives half credit to equal scores. Paired wins and ties are computed from integer pair credits. Cap-activation rates use `Q < 0.05`, average within each source before averaging across sources, and exclude reference and calibration sources.

## Outputs

All outputs are written to `results/` by default; use `--output PATH` to change the destination.

| File | Paper result |
|---|---|
| `table1_auroc.csv` | Mean AUROC, Table 1 |
| `table2_cap_rates.csv` | Cap-activation percentages, Table 2 |
| `table3_audit_utility.csv` | TPR/FPR percentages, Table 3 |
| `alpha_sensitivity.pdf` | Change in AUROC relative to uncapped Fisher, Figure 3 |
| `paired_counts.csv` | Paired wins, ties, losses, and mean gain in percentage points, Table 6 |
| `alpha_auroc.csv` | Mean AUROC and standard deviation across partitions, Table 7 |
| `alpha_fpr.csv` | Cap-ablation FPR percentages, Table 8 |
| `tables.md` | Formatted main tables and paired counts |
| `partitions/*.npz` | Numerical results for each partition |

Expected Table 1 values:

| Domain | Mean | Max | Min | RaMIA | Uncapped Fisher | SCF |
|---|---:|---:|---:|---:|---:|---:|
| Reddit | .6443 | .6176 | .6158 | .6444 | .6438 | .6456 |
| CC News | .6305 | .5953 | .6050 | .6227 | .6327 | .6354 |
| Enron | .8068 | .7222 | .6104 | .7973 | .8227 | .8309 |
| Chromium | .6178 | .5527 | .5620 | .6196 | .6279 | .6346 |

## Data format

Each `data/*.npz` contains four arrays:

- `values`: float64 document scores concatenated across sources.
- `offsets`: source boundaries; source i occupies `values[offsets[i]:offsets[i+1]]`.
- `labels`: 1 for a participating source, 0 for a nonparticipating source.
- `null_order`: the 500 null-source permutations, expressed as source-row indices.

The release contains numerical score artifacts and anonymous source indices. Dataset origins, document construction, and model settings are recorded in [config/data_construction.md](config/data_construction.md).
