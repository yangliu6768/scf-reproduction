"""Reproduce the paper's tables and cap-sensitivity figure from document scores."""

import argparse
import csv
from fractions import Fraction
from pathlib import Path

import numpy as np

from scf import auc_credits, conformal_ranks, fisher_scores, raw_scores, reference_tail


DOMAINS = {"reddit": "Reddit", "cc_news": "CC News", "enron": "Enron", "chromium": "Chromium"}
ALPHAS = np.array([0.0, 0.01, 0.02, 0.05, 0.10])
LEVELS = [0.05, 0.10, 0.20]
METHODS = ["Mean", "Max", "Min", "RaMIA", "Uncapped Fisher", "SCF (.01)", "SCF (.02)", "SCF", "SCF (.10)"]
MAIN = [0, 1, 2, 3, 4, 7]
DECISIONS = [0, 1, 2, 3, 7]


def evaluate(domain, data_dir, output):
    with np.load(data_dir / f"{domain}.npz", allow_pickle=False) as data:
        values, offsets, labels, orders = (data[k] for k in ["values", "offsets", "labels", "null_order"])
    sizes = np.diff(offsets)
    positive, null = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    k, l = (25, 25) if domain == "enron" else (40, 59)
    if offsets[0] != 0 or offsets[-1] != len(values) or len(sizes) != len(labels):
        raise ValueError(f"Invalid source offsets: {domain}")
    if not np.isfinite(values).all() or np.any((sizes < 1) | (sizes > 16)):
        raise ValueError(f"Invalid query bags: {domain}")
    if orders.shape != (500, len(null)) or not np.all(np.sort(orders, axis=1) == null):
        raise ValueError(f"Invalid source partitions: {domain}")
    n1, n0 = len(positive), len(null) - k - l
    raw = raw_scores(values, offsets)
    credits = np.empty((len(orders), len(METHODS)), dtype=np.int64)
    tp = np.empty((len(orders), len(LEVELS), len(METHODS)), dtype=np.int64)
    fp = np.empty_like(tp)
    cap_rates = np.empty((len(orders), 2))
    for b, order in enumerate(orders):
        reference, calibration, negative = order[:k], order[k:k + l], order[k + l:]
        target = np.r_[positive, negative]
        q = reference_tail(values, offsets, reference)
        scores = np.column_stack([raw, fisher_scores(q, offsets, ALPHAS)])
        credits[b] = auc_credits(scores[positive], scores[negative])
        for columns, controls in [(slice(0, 4), order[:k + l]), (slice(4, None), calibration)]:
            ranks = conformal_ranks(scores[controls, columns], scores[target, columns])
            for j, level in enumerate(LEVELS):
                cutoff = int(Fraction(str(level)) * (len(controls) + 1))
                rejected = ranks <= cutoff
                tp[b, j, columns] = rejected[:n1].sum(axis=0)
                fp[b, j, columns] = rejected[n1:].sum(axis=0)
        per_source = np.add.reduceat((q < 0.05).astype(np.int64), offsets[:-1]) / sizes
        cap_rates[b] = [per_source[negative].mean(), per_source[positive].mean()]
        if (b + 1) % 100 == 0:
            print(f"{DOMAINS[domain]}: {b + 1}/{len(orders)} partitions", flush=True)
    arrays = output / "partitions"
    arrays.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays / f"{domain}.npz", credits=credits, true_positives=tp,
                        false_positives=fp, cap_rates=cap_rates, methods=np.array(METHODS),
                        alphas=ALPHAS, levels=np.array(LEVELS), participant_count=n1, null_count=n0)
    return dict(domain=domain, credits=credits, auroc=credits / (2 * n1 * n0),
                tp=tp, fp=fp, cap_rates=cap_rates, n1=n1, n0=n0)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results, output):
    tables = {name: [] for name in ["table1_auroc", "table2_cap_rates", "table3_audit_utility",
                                   "paired_counts", "alpha_auroc", "alpha_fpr"]}
    for result in results:
        domain, auc = DOMAINS[result["domain"]], result["auroc"]
        tables["table1_auroc"].append(dict(domain=domain, **{METHODS[j]: auc[:, j].mean() for j in MAIN}))
        rates = 100 * result["cap_rates"].mean(axis=0)
        tables["table2_cap_rates"].append(dict(domain=domain, nonparticipant_percent=rates[0], participant_percent=rates[1]))
        for j, level in enumerate(LEVELS):
            row = dict(domain=domain, level=level)
            for m in DECISIONS:
                row[METHODS[m] + " TPR (%)"] = 100 * (result["tp"][:, j, m] / result["n1"]).mean()
                row[METHODS[m] + " FPR (%)"] = 100 * (result["fp"][:, j, m] / result["n0"]).mean()
            tables["table3_audit_utility"].append(row)
        for j in range(5):
            difference = result["credits"][:, 7] - result["credits"][:, j]
            tables["paired_counts"].append(dict(domain=domain, comparator=METHODS[j],
                wins=int(np.sum(difference > 0)), ties=int(np.sum(difference == 0)),
                losses=int(np.sum(difference < 0)),
                mean_gain_pp=100 * np.mean(difference / (2 * result["n1"] * result["n0"]))))
        for j, alpha in enumerate(ALPHAS):
            column = 4 + j
            tables["alpha_auroc"].append(dict(domain=domain, alpha=alpha,
                auroc=auc[:, column].mean(), split_sd=auc[:, column].std(ddof=1),
                gain_over_uncapped=np.mean(auc[:, column] - auc[:, 4])))
            for t, level in enumerate(LEVELS):
                tables["alpha_fpr"].append(dict(domain=domain, alpha=alpha, level=level,
                    fpr_percent=100 * (result["fp"][:, t, column] / result["n0"]).mean()))
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    write_markdown(tables, output)
    plot_caps(results, output)


def write_markdown(tables, output):
    lines = ["# Reproduced results", "", "## Table 1: AUROC", "",
             "| Domain | " + " | ".join(METHODS[j] for j in MAIN) + " |",
             "|---|" + "---:|" * len(MAIN)]
    for row in tables["table1_auroc"]:
        lines.append("| " + row["domain"] + " | " + " | ".join(f"{row[METHODS[j]]:.4f}" for j in MAIN) + " |")
    lines += ["", "## Table 2: Cap activation (%)", "", "| Domain | Nonparticipant | Participant |", "|---|---:|---:|"]
    for row in tables["table2_cap_rates"]:
        lines.append(f"| {row['domain']} | {row['nonparticipant_percent']:.3f} | {row['participant_percent']:.3f} |")
    lines += ["", "## Table 3: TPR/FPR (%)", "",
              "| Domain | Level | " + " | ".join(METHODS[j] for j in DECISIONS) + " |",
              "|---|---:|" + "---:|" * len(DECISIONS)]
    for row in tables["table3_audit_utility"]:
        cells = [f"{row[METHODS[j] + ' TPR (%)']:.2f}/{row[METHODS[j] + ' FPR (%)']:.2f}" for j in DECISIONS]
        lines.append(f"| {row['domain']} | {row['level']:.0%} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired AUROC comparisons", "",
              "| Domain | Comparator | Wins | Ties | Losses | Mean gain (pp) |", "|---|---|---:|---:|---:|---:|"]
    for row in tables["paired_counts"]:
        lines.append(f"| {row['domain']} | {row['comparator']} | {row['wins']} | {row['ties']} | {row['losses']} | {row['mean_gain_pp']:+.6f} |")
    (output / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_caps(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 2.65), layout="constrained")
    for result in results:
        auc = result["auroc"]
        ax.plot(ALPHAS, (auc[:, 4:] - auc[:, [4]]).mean(axis=0), marker="o",
                markersize=3, label=DOMAINS[result["domain"]])
    ax.set(xlabel="Cap α (0: uncapped)", ylabel="Change in AUROC")
    ax.set_xticks(ALPHAS, ["0", ".01", ".02", ".05", ".10"])
    ax.axhline(0, color="0.7", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output / "alpha_sensitivity.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(output / "alpha_sensitivity.png", dpi=220, metadata={"Software": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=list(DOMAINS), nargs="+", default=list(DOMAINS))
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [evaluate(domain, args.data, args.output) for domain in args.domain]
    summarize(results, args.output)
    print(f"Saved tables and figures to {args.output}")


if __name__ == "__main__":
    main()
