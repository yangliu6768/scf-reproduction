"""Check exact agreement at the paper's displayed precision.

Retraining can change scores through hardware-dependent floating-point arithmetic;
a mismatch here reports numerical differences, not necessarily an implementation bug.
"""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TABLES = {"table1_auroc": 4, "table2_cap_rates": 3, "table3_audit_utility": 2}


def check(output):
    failures = []
    for table, precision in TABLES.items():
        with (ROOT / "expected" / f"{table}.csv").open() as handle:
            expected = list(csv.DictReader(handle))
        with (output / f"{table}.csv").open() as handle:
            actual = list(csv.DictReader(handle))
        keys = ["domain", "level"] if table == "table3_audit_utility" else ["domain"]
        def key(row):
            return tuple(float(row[k]) if k == "level" else row[k] for k in keys)
        observed = {key(row): row for row in actual}
        if len(observed) != len(actual) or observed.keys() != {key(row) for row in expected}:
            failures.append(f"{table}: missing, repeated, or unexpected rows")
        for row in expected:
            if key(row) not in observed:
                continue
            for column, value in row.items():
                if column in keys:
                    continue
                found = float(observed[key(row)][column])
                if f"{found:.{precision}f}" != f"{float(value):.{precision}f}":
                    failures.append(f"{table}, {key(row)}, {column}: expected {value}, got {found:.8f}")
    if failures:
        raise SystemExit("Tables differ from the paper:\n" + "\n".join(failures))
    print("Tables 1–3 match every displayed value in the paper.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    check(parser.parse_args().results)
