"""Rebuild the fixed training, validation, and query inputs from public corpora."""

import argparse
from collections import defaultdict
import csv
from email import policy
from email.parser import BytesParser
import gzip
import json
from pathlib import Path
import re
import tarfile
import urllib.request
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "reddit": ("fddemarco/pushshift-reddit-comments", "987111c7c262a3c1255f5da4ecc545782280e794", "data"),
    "cc_news": ("vblagoje/cc_news", "81eb2ce0d2a9dad6ad16b68ef750ec290880fa36", "plain_text"),
}
ARCHIVES = {
    "enron": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
    "chromium": "https://zissou.infosci.cornell.edu/convokit/datasets/chromium-corpus/chromium-corpus.zip",
}


def read_manifest(domain):
    with gzip.open(ROOT / "splits" / f"{domain}_documents.csv.gz", "rt") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ["row", "role", "source", "query_position", "prediction_tokens"]:
            row[key] = int(row[key])
        if (Path(row["file"]).name != row["file"] or row["row"] < 0 or row["source"] < 0
                or row["query_position"] < -1 or row["prediction_tokens"] < 1):
            raise ValueError("Invalid raw-record location or document metadata")
    if len({(r["file"], r["row"]) for r in rows}) != len(rows):
        raise ValueError("Repeated raw document in split manifest")
    return rows


def validate_manifest(rows, audit):
    roles = np.array([row["role"] for row in rows])
    if set(roles) != {0, 1, 2} or np.any(np.diff(roles) < 0):
        raise ValueError("Manifest roles must be ordered train, validation, then query")
    selected = [row for row in rows if row["query_position"] >= 0]
    selected.sort(key=lambda row: row["query_position"])
    positions = [row["query_position"] for row in selected]
    if positions != list(range(int(audit["offsets"][-1]))):
        raise ValueError("Missing or repeated audit query positions")
    sources = np.repeat(np.arange(len(audit["labels"])), np.diff(audit["offsets"]))
    if not np.array_equal([row["source"] for row in selected], sources) or any(row["role"] != 2 for row in selected):
        raise ValueError("Audit queries do not match the declared source bags")
    trained = {row["source"] for row in rows if row["role"] == 0}
    if not np.array_equal([i in trained for i in range(len(audit["labels"]))], audit["labels"]):
        raise ValueError("Source participation labels do not match training assignments")


def download(domain, files, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(files):
        target = directory / name
        if target.exists():
            continue
        print(f"Downloading {domain}/{name}", flush=True)
        if domain in DATASETS:
            from huggingface_hub import hf_hub_download
            repo, revision, prefix = DATASETS[domain]
            path = Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                                       filename=f"{prefix}/{name}", local_dir=directory))
            path.replace(target)
        else:
            partial = target.with_name(target.name + ".partial")
            with urllib.request.urlopen(ARCHIVES[domain], timeout=120) as response, partial.open("wb") as out:
                while chunk := response.read(8 * 1024**2):
                    out.write(chunk)
            partial.replace(target)


def email_body(raw):
    message = BytesParser(policy=policy.default).parsebytes(raw)
    parts = []
    candidates = message.walk() if message.is_multipart() else [message]
    for part in candidates:
        if part.is_multipart() or (message.is_multipart() and part.get_content_disposition() == "attachment"):
            continue
        if message.is_multipart() and part.get_content_type() != "text/plain":
            continue
        if not part.get_content_type().startswith("text/"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            value = part.get_payload(decode=False)
            parts.append(value if isinstance(value, str) else "")
        else:
            try:
                parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
            except LookupError:
                parts.append(payload.decode("utf-8", errors="replace"))
    return "\n".join(parts).replace("\r\n", "\n").replace("\r", "\n")


def clean_comment(text):
    return "\n".join(line for line in text.splitlines() if not re.match(r"^\s*>", line)).strip()


def read_texts(domain, rows, directory):
    files = defaultdict(dict)
    for index, row in enumerate(rows):
        files[row["file"]][row["row"]] = index
    for name, selected in files.items():
        path = directory / name
        last_row = max(selected)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; download the corpus or add --download")
        if domain in DATASETS:
            parquet = pq.ParquetFile(path)
            offset = 0
            indices = np.array(sorted(selected))
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                subset = indices[(indices >= offset) & (indices < offset + size)]
                if len(subset):
                    column = "body" if domain == "reddit" else "text"
                    table = parquet.read_row_group(group, columns=[column])
                    values = table.column(0).take(pa.array(subset - offset)).to_pylist()
                    for row_index, text in zip(subset, values):
                        yield selected[row_index], text
                offset += size
        elif domain == "chromium":
            with zipfile.ZipFile(path) as archive:
                member = next(x for x in archive.namelist() if x.endswith("/utterances.jsonl"))
                with archive.open(member) as handle:
                    for row_index, line in enumerate(handle):
                        if row_index in selected:
                            yield selected[row_index], clean_comment(json.loads(line)["text"])
                        if row_index >= last_row:
                            break
        else:
            with tarfile.open(path, "r|gz") as archive:
                for row_index, member in enumerate(archive):
                    if row_index in selected:
                        if not member.isfile():
                            raise ValueError("Manifest points to a non-file archive entry")
                        yield selected[row_index], email_body(archive.extractfile(member).read())
                    if row_index >= last_row:
                        break


def prepare(domain, raw, output, fetch=False, tokenizer_path=None):
    from transformers import AutoTokenizer

    rows = read_manifest(domain)
    with np.load(ROOT / "splits" / f"{domain}_audit.npz", allow_pickle=False) as audit:
        validate_manifest(rows, audit)
    if fetch:
        download(domain, {r["file"] for r in rows}, raw)
    config = json.loads((ROOT / "config/experiment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or config["model"]),
        **({} if tokenizer_path else {"revision": config["revision"]}))
    tokens = [None] * len(rows)
    pending = []

    def encode():
        ids = tokenizer([text for _, text in pending], add_special_tokens=False, padding=False,
                        truncation=True, max_length=257 if domain == "chromium" else 511)["input_ids"]
        for (index, _), sequence in zip(pending, ids):
            if domain != "chromium":
                sequence.append(tokenizer.eos_token_id)
            if len(sequence) - 1 != rows[index]["prediction_tokens"]:
                raise ValueError(f"Token count differs at manifest row {index}; check the corpus version")
            tokens[index] = np.asarray(sequence, dtype=np.uint16)
        pending.clear()

    count = 0
    for index, text in read_texts(domain, rows, raw):
        pending.append((index, text))
        count += 1
        if len(pending) == 2048:
            encode()
        if count % 50000 == 0:
            print(f"{domain}: processed {count}/{len(rows)} documents", flush=True)
    if pending:
        encode()
    if any(x is None for x in tokens):
        raise ValueError("Raw corpus does not contain all manifest records")
    roles = np.array([row["role"] for row in rows], dtype=np.uint8)
    query_position = np.array([row["query_position"] for row in rows], dtype=np.int64)
    fit_inputs = {x.tobytes() for x, role in zip(tokens, roles) if role < 2}
    overlaps = sum(x.tobytes() in fit_inputs for x, position in zip(tokens, query_position) if position >= 0)
    if overlaps:
        raise ValueError(f"{overlaps} audit queries exactly overlap training or validation inputs")
    offsets = np.r_[0, np.cumsum([len(x) for x in tokens])]
    settings = config["domains"][domain]
    counts = [int(np.sum(roles == role)) for role in range(3)]
    exposure = sum(len(x) - 1 for x, role in zip(tokens, roles) if role == 0)
    if counts[0] != settings["train_documents"] or counts[1] != settings["validation_documents"] or exposure != settings["train_tokens"]:
        raise ValueError("Training exposure differs from the experiment configuration")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{domain}.npz"
    np.savez_compressed(path, tokens=np.concatenate(tokens), offsets=offsets, roles=roles,
                        query_position=query_position)
    print(f"Saved {path}: train={counts[0]}, validation={counts[1]}, scoring={counts[2]}, "
          f"audit={np.sum(query_position >= 0)}, exact-input overlaps={overlaps}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=["reddit", "cc_news", "enron", "chromium"])
    parser.add_argument("--raw", type=Path, help="Folder containing this domain's raw files")
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tokenizer", type=Path, help="Optional local copy of the pinned tokenizer")
    args = parser.parse_args()
    prepare(args.domain, args.raw or Path("data/raw") / args.domain, args.output, args.download, args.tokenizer)
