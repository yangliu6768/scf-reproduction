"""Fine-tune GPT-Neo and compute the paper's document scores."""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parent


def collate(examples):
    width = max(map(len, examples))
    ids = torch.full((len(examples), width), 50256, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, tokens in enumerate(examples):
        ids[i, :len(tokens)] = torch.as_tensor(np.asarray(tokens, dtype=np.int64))
        mask[i, :len(tokens)] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    return dict(input_ids=ids, attention_mask=mask, labels=labels)


def sequence_likelihood(logits, ids, mask):
    nll = torch.nn.functional.cross_entropy(
        logits[:, :-1].contiguous().transpose(1, 2),
        ids[:, 1:].contiguous(), reduction="none",
    )
    return -(nll * mask[:, 1:]).sum(1), mask[:, 1:].sum(1)


def schedule(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def load_tokens(path):
    with np.load(path, allow_pickle=False) as data:
        values, offsets = data["tokens"], data["offsets"]
        roles, query_position = data["roles"], data["query_position"]
    if (offsets.ndim != 1 or len(offsets) < 2 or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 2)):
        raise ValueError("Invalid token offsets")
    if roles.shape != (len(offsets) - 1,) or query_position.shape != roles.shape or set(roles) != {0, 1, 2}:
        raise ValueError("Invalid document roles or query positions")
    if np.any((query_position >= 0) & (roles != 2)) or np.any(values > 50256):
        raise ValueError("Invalid query assignment or token ID")
    bags = [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return {role: [bags[i] for i in np.flatnonzero(roles == j)]
            for j, role in enumerate(["train", "validation", "query"])}, query_position[roles == 2]


def batches(tokens, size):
    for start in range(0, len(tokens), size):
        yield collate(tokens[start:start + size])


@torch.inference_mode()
def score(model, tokens, device, batch_size, loader=False):
    model.eval()
    values, totals = [], []
    stream = (DataLoader(tokens, batch_size=batch_size, collate_fn=collate, num_workers=0)
              if loader else batches(tokens, batch_size))
    for batch in stream:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
            ll, _ = sequence_likelihood(logits, ids, mask)
        values.append(ll.float().cpu().numpy().astype(np.float64))
        totals.append(float((-ll).float().sum().cpu()))
    return np.concatenate(values), sum(totals)


def fit(domain, prepared, output, model_path=None, device_name="cuda"):
    settings = json.loads((ROOT / "config/experiment.json").read_text())
    config = settings["domains"][domain]
    tokens, query_position = load_tokens(prepared / f"{domain}.npz")
    counts = {role: len(bags) for role, bags in tokens.items()}
    exposure = sum(len(x) - 1 for x in tokens["train"])
    if (counts["train"] != config["train_documents"] or exposure != config["train_tokens"]
            or counts["validation"] != config["validation_documents"]):
        raise ValueError("Prepared data do not match the paper's training exposure")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The experiment requires a CUDA GPU supporting bfloat16")
    destination = output / domain
    destination.mkdir(parents=True, exist_ok=False)
    seed = config["seed"]
    # Fixed seeds preserve RNG choices, but BF16 training need not be bitwise
    # identical across GPU models or software stacks; scores and metrics may vary.
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    kwargs = dict(torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True)
    if model_path is None:
        kwargs["revision"] = settings["revision"]
    initial = str(model_path or settings["model"])
    model = AutoModelForCausalLM.from_pretrained(initial, **kwargs).to(device)
    model.config.use_cache = False
    size = settings["batch_size"]
    loader_mode = config["shuffle"] == "dataloader"
    if not loader_mode:
        pre, _ = score(model, tokens["query"], device, size)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    if loader_mode:
        train_loader = DataLoader(tokens["train"], batch_size=size, shuffle=True,
                                  generator=generator, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 betas=settings["adam_betas"], eps=settings["adam_eps"],
                                 weight_decay=settings["weight_decay"])
    total = settings["epochs"] * math.ceil(counts["train"] / size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, settings["warmup_steps"], total))
    best, selected_epoch, step = math.inf, None, 0
    curve = []
    started = time.monotonic()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        if loader_mode:
            stream = train_loader
        else:
            order = torch.randperm(counts["train"], generator=generator).tolist()
            stream = batches([tokens["train"][i] for i in order], size)
        for batch in stream:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 1000 == 0:
                print(f"{domain}: epoch {epoch}, update {step}/{total}", flush=True)
        ll, batch_total = score(model, tokens["validation"], device, size, loader_mode)
        numerator = batch_total if loader_mode else -math.fsum(ll.tolist())
        nll = numerator / sum(len(x) - 1 for x in tokens["validation"])
        if not math.isfinite(nll):
            raise FloatingPointError("Nonfinite validation loss")
        improved = nll < best
        if improved:
            model.save_pretrained(destination / "checkpoint", safe_serialization=True)
            best, selected_epoch = nll, epoch
        curve.append(dict(epoch=epoch, updates=step, validation_nll=nll, new_best=improved))
        with (destination / "validation.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        print(f"{domain}: epoch {epoch}, validation NLL {nll:.10f}, best epoch {selected_epoch}", flush=True)
    training_seconds = time.monotonic() - started
    del model, optimizer, scheduler, loss, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(destination / "checkpoint", **{
        k: v for k, v in kwargs.items() if k != "revision"}).to(device)
    model.config.use_cache = False
    post, _ = score(model, tokens["query"], device, size, loader_mode)
    del model
    if loader_mode:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(initial, **kwargs).to(device)
        model.config.use_cache = False
        pre, _ = score(model, tokens["query"], device, size, loader_mode)
        del model
    lengths = np.array([len(x) - 1 for x in tokens["query"]])
    scores = (post - pre) / lengths
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite document scores")
    kept = query_position >= 0
    positions = query_position[kept]
    if not np.array_equal(np.sort(positions), np.arange(len(positions))):
        raise ValueError("Missing or repeated audit query positions")
    values = np.empty(len(positions), dtype=np.float64)
    values[positions] = scores[kept]
    result_dir = output / "scores"
    result_dir.mkdir(exist_ok=True)
    with np.load(ROOT / "splits" / f"{domain}_audit.npz", allow_pickle=False) as audit:
        if len(values) != audit["offsets"][-1]:
            raise ValueError("Audit queries do not match the source bags")
        np.savez_compressed(result_dir / f"{domain}.npz", values=values)
    run = dict(domain=domain, training_seed=seed, selected_epoch=selected_epoch,
               expected_epoch=config["epoch"], epoch_matches_paper=selected_epoch == config["epoch"],
               selected_update=selected_epoch * math.ceil(counts["train"] / size),
               validation_nll=best, train_documents=counts["train"], train_tokens=exposure,
               validation_documents=counts["validation"], training_seconds=training_seconds,
               elapsed_seconds=time.monotonic() - started, python=platform.python_version(),
               numpy=np.__version__, torch=torch.__version__,
               transformers=transformers.__version__, gpu=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
               configuration=settings, scoring_batch_size=size)
    (destination / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Saved {domain} scores to {result_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=["reddit", "cc_news", "enron", "chromium"])
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--model", type=Path, help="Optional local copy of the pinned pretrained model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fit(args.domain, args.data, args.output, args.model, args.device)
