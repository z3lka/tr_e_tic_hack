from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return SPACE_RE.sub(" ", str(value).strip())


def build_item_text(row: Any) -> str:
    parts = [clean_text(getattr(row, column, "")) for column in ITEM_COLUMNS[1:]]
    return " | ".join(part for part in parts if part)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_transformer_stack() -> tuple[Any, ...]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Contrastive training and encoding require torch and transformers") from exc
    return torch, nn, F, DataLoader, Dataset, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def seed_everything(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
    return torch.device(requested)


def load_positive_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "label" in frame:
        frame = frame.loc[frame["label"].ne("0")]
    frame = frame[["term_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No positives found in {path}")
    return frame


def grouped_inner_split(
    positives: pd.DataFrame, valid_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < valid_size < 1.0:
        raise ValueError(f"valid_size must be in (0, 1), got {valid_size}")
    terms = positives["term_id"].drop_duplicates().astype(str).to_numpy()
    if len(terms) < 2:
        raise ValueError("At least two terms are required for grouped checkpoint selection")
    rng = np.random.default_rng(seed)
    terms = terms.copy()
    rng.shuffle(terms)
    n_valid = min(len(terms) - 1, max(1, int(round(len(terms) * valid_size))))
    valid_terms = set(terms[:n_valid])
    inner_train = positives.loc[~positives["term_id"].isin(valid_terms)].reset_index(drop=True)
    inner_valid = positives.loc[positives["term_id"].isin(valid_terms)].reset_index(drop=True)
    if set(inner_train["term_id"]) & set(inner_valid["term_id"]):
        raise AssertionError("Term leakage in inner validation split")
    split = pd.DataFrame(
        {
            "term_id": terms,
            "split": ["inner_valid" if term in valid_terms else "inner_train" for term in terms],
        }
    ).sort_values("term_id").reset_index(drop=True)
    return inner_train, inner_valid, split


def apply_outer_term_split(
    positives: pd.DataFrame,
    split_manifest: Path | None,
    outer_fold: int,
) -> tuple[pd.DataFrame, set[str]]:
    if split_manifest is None or outer_fold < 0:
        return positives.reset_index(drop=True), set()
    split = pd.read_csv(split_manifest, dtype={"term_id": str}, keep_default_na=False)
    if not {"term_id", "fold"}.issubset(split.columns):
        raise ValueError(f"{split_manifest} must contain term_id and fold")
    split["fold"] = pd.to_numeric(split["fold"], errors="raise").astype(int)
    outer_terms = set(split.loc[split["fold"].eq(outer_fold), "term_id"].astype(str))
    if not outer_terms:
        raise ValueError(f"No terms assigned to outer fold {outer_fold}")
    outer_train = positives.loc[~positives["term_id"].isin(outer_terms)].reset_index(drop=True)
    if set(outer_train["term_id"]) & outer_terms:
        raise AssertionError("Outer term leakage")
    if outer_train.empty:
        raise ValueError("Outer split removed every positive pair")
    return outer_train, outer_terms


def known_positive_mask_numpy(
    batch_term_ids: list[str],
    candidate_item_ids: list[str],
    positives_by_term: dict[str, set[str]],
    target_columns: np.ndarray,
) -> np.ndarray:
    """True entries must be removed from logits; designated targets are always retained."""
    if len(batch_term_ids) != len(target_columns):
        raise ValueError("target_columns must contain one target for every query")
    mask = np.zeros((len(batch_term_ids), len(candidate_item_ids)), dtype=bool)
    for row, term_id in enumerate(batch_term_ids):
        known = positives_by_term.get(str(term_id), set())
        if known:
            mask[row] = np.fromiter(
                (str(item_id) in known for item_id in candidate_item_ids),
                dtype=bool,
                count=len(candidate_item_ids),
            )
        target = int(target_columns[row])
        if target < 0 or target >= len(candidate_item_ids):
            raise IndexError(f"Invalid target column {target}")
        mask[row, target] = False
    return mask


def masked_infonce_loss(
    query_embeddings: Any,
    candidate_embeddings: Any,
    mask: Any,
    target_columns: Any,
    temperature: float,
    F: Any,
) -> Any:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = query_embeddings @ candidate_embeddings.T / temperature
    if logits.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} does not match logits {logits.shape}")
    if bool(mask.gather(1, target_columns.unsqueeze(1)).any().item()):
        raise AssertionError("A designated InfoNCE target was masked")
    logits = logits.masked_fill(mask, torch_finfo_min(logits))
    return F.cross_entropy(logits, target_columns)


def torch_finfo_min(tensor: Any) -> float:
    # -inf can create NaN gradients in some fp16 kernels; the finite minimum is safe here.
    import torch

    return torch.finfo(tensor.dtype).min


def source_family(value: str) -> str:
    value = value.lower()
    if "tfidf" in value or value.endswith("lexical"):
        return "lexical"
    if "embedding" in value:
        return "embedding"
    return "uniform"


def load_negative_pool(path: Path) -> dict[str, dict[str, list[str]]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"term_id", "item_id", "negative_source"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.duplicated(["term_id", "item_id"]).any():
        raise ValueError(f"{path} contains duplicate pairs")
    pool: dict[str, dict[str, list[str]]] = {}
    for row in frame.itertuples(index=False):
        family = source_family(str(row.negative_source))
        if family == "uniform":
            # Conservative miner backfills retain the requested family in their suffix.
            if "lexical" in str(row.negative_source):
                family = "lexical"
            elif "embedding" in str(row.negative_source):
                family = "embedding"
        pool.setdefault(str(row.term_id), {}).setdefault(family, []).append(str(row.item_id))
    return pool


def assemble_candidate_item_ids(
    batch_rows: pd.DataFrame,
    negative_pool: dict[str, dict[str, list[str]]],
    catalog_item_ids: np.ndarray,
    seed: int,
    epoch: int,
    uniform_multiplier: int = 3,
    use_semi_hard: bool = True,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Build B targets + B alternating semi-hard + 3B shared uniform candidates."""
    batch_size = len(batch_rows)
    if batch_size == 0:
        raise ValueError("Cannot assemble an empty contrastive batch")
    if uniform_multiplier < 0:
        raise ValueError("uniform_multiplier must be non-negative")
    positives = batch_rows["item_id"].astype(str).tolist()
    candidates = list(positives)
    sources = ["positive"] * batch_size
    selected = set(candidates)
    row_indices = (
        batch_rows["row_index"].astype(int).to_numpy()
        if "row_index" in batch_rows
        else np.arange(batch_size)
    )
    rng = np.random.default_rng(stable_seed(seed, epoch, *sorted(int(value) for value in row_indices)))

    if use_semi_hard:
        for local_row, row in enumerate(batch_rows.itertuples(index=False)):
            term_id = str(row.term_id)
            row_index = int(getattr(row, "row_index", local_row))
            preferred = "lexical" if (row_index + epoch) % 2 == 0 else "embedding"
            alternatives = [preferred, "embedding" if preferred == "lexical" else "lexical", "uniform"]
            chosen: str | None = None
            chosen_source = preferred
            for family in alternatives:
                values = negative_pool.get(term_id, {}).get(family, [])
                if not values:
                    continue
                offset = stable_seed(seed, epoch, row_index, family) % len(values)
                for step in range(len(values)):
                    item_id = str(values[(offset + step) % len(values)])
                    if item_id not in selected:
                        chosen = item_id
                        chosen_source = family
                        break
                if chosen is not None:
                    break
            if chosen is None:
                for _ in range(max(1_000, len(catalog_item_ids))):
                    item_id = str(catalog_item_ids[int(rng.integers(0, len(catalog_item_ids)))])
                    if item_id not in selected:
                        chosen = item_id
                        chosen_source = "uniform_backfill"
                        break
            if chosen is None:
                raise ValueError("Catalog is too small to create unique semi-hard candidates")
            candidates.append(chosen)
            sources.append(chosen_source)
            selected.add(chosen)

    uniform_needed = uniform_multiplier * batch_size
    attempts = 0
    while uniform_needed:
        item_id = str(catalog_item_ids[int(rng.integers(0, len(catalog_item_ids)))])
        attempts += 1
        if item_id in selected:
            if attempts > max(10_000, len(catalog_item_ids) * 10):
                for fallback in catalog_item_ids:
                    fallback_id = str(fallback)
                    if fallback_id not in selected:
                        item_id = fallback_id
                        break
                else:
                    raise ValueError("Catalog is too small for the requested shared uniform negatives")
            else:
                continue
        candidates.append(item_id)
        sources.append("uniform")
        selected.add(item_id)
        uniform_needed -= 1
    targets = np.arange(batch_size, dtype=np.int64)
    return candidates, targets, sources


def make_model_class(torch: Any, nn: Any, F: Any):
    class ContrastiveBiEncoder(nn.Module):
        def __init__(self, encoder: Any, hidden_size: int, projection_dim: int):
            super().__init__()
            self.encoder = encoder
            self.query_projection = nn.Linear(hidden_size, projection_dim)
            self.item_projection = nn.Linear(hidden_size, projection_dim)

        @staticmethod
        def pool(output: Any, attention_mask: Any) -> Any:
            token_embeddings = output.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        def encode_query(self, batch: dict[str, Any]) -> Any:
            pooled = self.pool(self.encoder(**batch), batch["attention_mask"])
            return F.normalize(self.query_projection(pooled), p=2, dim=1)

        def encode_item(self, batch: dict[str, Any]) -> Any:
            pooled = self.pool(self.encoder(**batch), batch["attention_mask"])
            return F.normalize(self.item_projection(pooled), p=2, dim=1)

    return ContrastiveBiEncoder


def load_text_context(data_dir: Path) -> tuple[dict[str, str], dict[str, str], np.ndarray]:
    terms = pd.read_csv(data_dir / "terms.csv", usecols=["term_id", "query"], dtype=str, keep_default_na=False)
    term_texts = dict(zip(terms["term_id"].astype(str), terms["query"].map(clean_text)))
    items = pd.read_csv(data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype=str, keep_default_na=False)
    item_ids = items["item_id"].astype(str).to_numpy()
    item_texts = {str(row.item_id): build_item_text(row) for row in items.itertuples(index=False)}
    return term_texts, item_texts, item_ids


class HybridBatchBuilder:
    def __init__(
        self,
        tokenizer: Any,
        term_texts: dict[str, str],
        item_texts: dict[str, str],
        catalog_item_ids: np.ndarray,
        negative_pool: dict[str, dict[str, list[str]]],
        positives_by_term: dict[str, set[str]],
        query_max_length: int,
        item_max_length: int,
        seed: int,
        uniform_multiplier: int,
        use_semi_hard: bool,
    ):
        self.tokenizer = tokenizer
        self.term_texts = term_texts
        self.item_texts = item_texts
        self.catalog_item_ids = catalog_item_ids
        self.negative_pool = negative_pool
        self.positives_by_term = positives_by_term
        self.query_max_length = query_max_length
        self.item_max_length = item_max_length
        self.seed = seed
        self.uniform_multiplier = uniform_multiplier
        self.use_semi_hard = use_semi_hard
        self.epoch = 0
        self.source_counts: Counter[str] = Counter()

    def __call__(self, records: list[dict[str, object]]) -> tuple[Any, ...]:
        rows = pd.DataFrame(records)
        candidates, targets, sources = assemble_candidate_item_ids(
            rows,
            self.negative_pool,
            self.catalog_item_ids,
            self.seed,
            self.epoch,
            self.uniform_multiplier,
            self.use_semi_hard,
        )
        term_ids = rows["term_id"].astype(str).tolist()
        query_texts = [self.term_texts[term_id] for term_id in term_ids]
        try:
            item_batch_texts = [self.item_texts[item_id] for item_id in candidates]
        except KeyError as exc:
            raise KeyError(f"Negative pool contains unknown catalog item {exc.args[0]}") from exc
        query_batch = self.tokenizer(
            query_texts, padding=True, truncation=True, max_length=self.query_max_length, return_tensors="pt"
        )
        item_batch = self.tokenizer(
            item_batch_texts, padding=True, truncation=True, max_length=self.item_max_length, return_tensors="pt"
        )
        mask = known_positive_mask_numpy(term_ids, candidates, self.positives_by_term, targets)
        self.source_counts.update(sources)
        return query_batch, item_batch, mask, targets


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_checkpoint(
    model_dir: Path,
    model: Any,
    tokenizer: Any,
    config: dict[str, object],
    torch: Any,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(
        {
            "query_projection": model.query_projection.state_dict(),
            "item_projection": model.item_projection.state_dict(),
        },
        model_dir / "projection_heads.pt",
    )
    (model_dir / "contrastive_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_checkpoint(model_dir: Path, device: Any) -> tuple[Any, ...]:
    torch, nn, F, _, _, AutoModel, AutoTokenizer, _ = require_transformer_stack()
    config = json.loads((model_dir / "contrastive_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    Model = make_model_class(torch, nn, F)
    model = Model(encoder, int(config["hidden_size"]), int(config["projection_dim"]))
    state = torch.load(model_dir / "projection_heads.pt", map_location="cpu")
    model.query_projection.load_state_dict(state["query_projection"])
    model.item_projection.load_state_dict(state["item_projection"])
    model.to(device).eval()
    return torch, F, tokenizer, model, config


def train(args: argparse.Namespace) -> None:
    started = time.time()
    torch, nn, F, DataLoader, Dataset, AutoModel, AutoTokenizer, get_scheduler = require_transformer_stack()
    seed_everything(args.seed, torch)
    device = choose_device(torch, args.device)
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    positives = load_positive_pairs(data_dir / "training_pairs.csv")
    positives, outer_terms = apply_outer_term_split(
        positives, Path(args.split_manifest) if args.split_manifest else None, args.outer_fold
    )
    if args.limit_positives and len(positives) > args.limit_positives:
        kept_terms = positives["term_id"].drop_duplicates().astype(str)
        positives = positives.loc[positives["term_id"].isin(set(kept_terms))].head(args.limit_positives).reset_index(drop=True)
    inner_train, inner_valid, inner_split = grouped_inner_split(positives, args.inner_valid_size, args.seed)
    inner_split["outer_fold"] = args.outer_fold
    inner_split["used_in_full_refit"] = int(args.refit_full)
    inner_split.to_csv(model_dir / "training_term_split.csv", index=False)
    positives_by_term = {
        str(term_id): set(group["item_id"].astype(str))
        for term_id, group in positives.groupby("term_id", sort=False)
    }
    term_texts, item_texts, catalog_item_ids = load_text_context(data_dir)
    missing_terms = set(positives["term_id"].astype(str)) - set(term_texts)
    missing_items = set(positives["item_id"].astype(str)) - set(item_texts)
    if missing_terms or missing_items:
        raise KeyError(f"missing_terms={len(missing_terms)} missing_items={len(missing_items)}")
    negative_pool = load_negative_pool(Path(args.negative_pool)) if args.use_semi_hard else {}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    hidden_size = int(encoder.config.hidden_size)
    Model = make_model_class(torch, nn, F)
    model = Model(encoder, hidden_size, args.projection_dim).to(device)

    class PositiveDataset(Dataset):
        def __init__(self, rows: pd.DataFrame):
            self.rows = rows.reset_index(drop=True).copy()
            self.rows["row_index"] = np.arange(len(self.rows), dtype=np.int64)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, object]:
            row = self.rows.iloc[index]
            return {"term_id": str(row.term_id), "item_id": str(row.item_id), "row_index": int(row.row_index)}

    train_builder = HybridBatchBuilder(
        tokenizer, term_texts, item_texts, catalog_item_ids, negative_pool, positives_by_term,
        args.query_max_length, args.item_max_length, args.seed, args.uniform_multiplier, args.use_semi_hard,
    )
    valid_builder = HybridBatchBuilder(
        tokenizer, term_texts, item_texts, catalog_item_ids, negative_pool, positives_by_term,
        args.query_max_length, args.item_max_length, args.seed + 1, args.uniform_multiplier, args.use_semi_hard,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        PositiveDataset(inner_train), batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=train_builder,
    )
    valid_loader = DataLoader(
        PositiveDataset(inner_valid), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=valid_builder,
    )
    parameter_groups = [
        {"params": model.encoder.parameters(), "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": model.query_projection.parameters(), "lr": args.head_learning_rate, "weight_decay": args.weight_decay},
        {"params": model.item_projection.parameters(), "lr": args.head_learning_rate, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(parameter_groups)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = min(args.max_steps, steps_per_epoch * args.epochs) if args.max_steps else steps_per_epoch * args.epochs
    scheduler = get_scheduler(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )
    amp_enabled = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    cache_fingerprint = None
    if args.cache_manifest:
        cache_fingerprint = json.loads(Path(args.cache_manifest).read_text(encoding="utf-8")).get("cache_key")
    base_config: dict[str, object] = {
        "base_model": args.model_name,
        "hidden_size": hidden_size,
        "projection_dim": args.projection_dim,
        "temperature": args.temperature,
        "query_max_length": args.query_max_length,
        "item_max_length": args.item_max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "uniform_multiplier": args.uniform_multiplier,
        "use_semi_hard": args.use_semi_hard,
        "fp16": args.fp16,
        "split_seed": args.seed,
        "outer_fold": args.outer_fold,
        "outer_holdout_terms": len(outer_terms),
        "inner_train_terms": int(inner_train["term_id"].nunique()),
        "inner_valid_terms": int(inner_valid["term_id"].nunique()),
        "cache_fingerprint": cache_fingerprint,
        "negative_pool_fingerprint": file_fingerprint(Path(args.negative_pool)) if args.use_semi_hard else None,
    }

    def batch_loss(batch: tuple[Any, ...]) -> Any:
        query_batch, item_batch, mask_array, target_array = batch
        query_batch = move_batch(query_batch, device)
        item_batch = move_batch(item_batch, device)
        mask = torch.from_numpy(mask_array).to(device=device, dtype=torch.bool)
        targets = torch.from_numpy(target_array).to(device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
            query_embedding = model.encode_query(query_batch)
            item_embedding = model.encode_item(item_batch)
            return masked_infonce_loss(query_embedding, item_embedding, mask, targets, args.temperature, F)

    def evaluate() -> float:
        model.eval()
        total_loss = 0.0
        total_rows = 0
        valid_builder.epoch = 0
        with torch.inference_mode():
            for batch in tqdm(valid_loader, desc="inner validation", leave=False):
                loss = batch_loss(batch)
                rows = len(batch[3])
                total_loss += float(loss.detach().cpu()) * rows
                total_rows += rows
        return total_loss / max(1, total_rows)

    best_loss = float("inf")
    best_epoch = 0
    global_step = 0
    epoch_metrics: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_builder.epoch = epoch - 1
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_rows = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"contrastive epoch {epoch}"), start=1):
            loss = batch_loss(batch) / args.grad_accum_steps
            scaler.scale(loss).backward()
            rows = len(batch[3])
            total_loss += float(loss.detach().cpu()) * args.grad_accum_steps * rows
            total_rows += rows
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    break
        valid_loss = evaluate()
        train_loss = total_loss / max(1, total_rows)
        epoch_metrics.append({"epoch": epoch, "train_loss": train_loss, "inner_valid_loss": valid_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} inner_valid_loss={valid_loss:.6f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            save_checkpoint(model_dir, model, tokenizer, {**base_config, "best_epoch": epoch}, torch)
        if args.max_steps and global_step >= args.max_steps:
            break

    refit_source_counts: dict[str, int] | None = None
    if args.refit_full:
        # Epoch selection above is leakage-safe but only sees the inner-training
        # terms. Start again from the pretrained encoder and fit the selected
        # number of epochs on every outer-training positive term.
        del model, optimizer, scheduler, scaler, parameter_groups
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seed_everything(args.seed + 10_000, torch)
        encoder = AutoModel.from_pretrained(args.model_name)
        model = Model(encoder, hidden_size, args.projection_dim).to(device)
        full_builder = HybridBatchBuilder(
            tokenizer, term_texts, item_texts, catalog_item_ids, negative_pool, positives_by_term,
            args.query_max_length, args.item_max_length, args.seed + 10_000,
            args.uniform_multiplier, args.use_semi_hard,
        )
        full_generator = torch.Generator()
        full_generator.manual_seed(args.seed + 10_000)
        full_loader = DataLoader(
            PositiveDataset(positives), batch_size=args.batch_size, shuffle=True,
            generator=full_generator, num_workers=args.num_workers,
            pin_memory=device.type == "cuda", collate_fn=full_builder,
        )
        refit_parameter_groups = [
            {"params": model.encoder.parameters(), "lr": args.learning_rate, "weight_decay": args.weight_decay},
            {"params": model.query_projection.parameters(), "lr": args.head_learning_rate, "weight_decay": args.weight_decay},
            {"params": model.item_projection.parameters(), "lr": args.head_learning_rate, "weight_decay": args.weight_decay},
        ]
        optimizer = torch.optim.AdamW(refit_parameter_groups)
        refit_epochs = max(1, best_epoch)
        refit_steps = math.ceil(len(full_loader) / args.grad_accum_steps) * refit_epochs
        if args.max_steps:
            refit_steps = min(refit_steps, args.max_steps)
        scheduler = get_scheduler(
            optimizer,
            num_warmup_steps=int(refit_steps * args.warmup_ratio),
            num_training_steps=refit_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        refit_global_step = 0
        for epoch in range(1, refit_epochs + 1):
            model.train()
            full_builder.epoch = epoch - 1
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(full_loader, desc=f"full outer refit {epoch}"), start=1):
                loss = batch_loss(batch) / args.grad_accum_steps
                scaler.scale(loss).backward()
                if step % args.grad_accum_steps == 0 or step == len(full_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    refit_global_step += 1
                    if args.max_steps and refit_global_step >= args.max_steps:
                        break
            if args.max_steps and refit_global_step >= args.max_steps:
                break
        save_checkpoint(
            model_dir,
            model,
            tokenizer,
            {
                **base_config,
                "best_epoch": best_epoch,
                "refit_on_all_outer_train": True,
                "refit_epochs": refit_epochs,
                "refit_positive_pairs": len(positives),
                "refit_terms": int(positives["term_id"].nunique()),
            },
            torch,
        )
        refit_source_counts = dict(sorted(full_builder.source_counts.items()))

    metrics = {
        "best_epoch": best_epoch,
        "best_inner_valid_loss": best_loss,
        "epochs": epoch_metrics,
        "training_positive_pairs": len(inner_train),
        "inner_validation_positive_pairs": len(inner_valid),
        "negative_source_counts": dict(sorted(train_builder.source_counts.items())),
        "refit_on_all_outer_train": args.refit_full,
        "refit_positive_pairs": len(positives) if args.refit_full else 0,
        "refit_terms": int(positives["term_id"].nunique()) if args.refit_full else 0,
        "refit_negative_source_counts": refit_source_counts,
        "elapsed_seconds": time.time() - started,
    }
    (model_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def encode_batches(
    ids: list[str],
    text_by_id: dict[str, str],
    tokenizer: Any,
    model: Any,
    tower: str,
    max_length: int,
    batch_size: int,
    device: Any,
    torch: Any,
    fp16: bool,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    amp_enabled = fp16 and device.type == "cuda"
    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(ids), batch_size), desc=f"encode {tower}"):
            texts = [text_by_id[value] for value in ids[start : start + batch_size]]
            batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
                embeddings = model.encode_query(batch) if tower == "query" else model.encode_item(batch)
            parts.append(embeddings.float().cpu().numpy().astype(np.float16))
    return np.concatenate(parts) if parts else np.empty((0, int(model.query_projection.out_features)), dtype=np.float16)


def load_entity_texts(data_dir: Path, entity: str, ids: set[str] | None = None) -> tuple[list[str], dict[str, str]]:
    if entity == "terms":
        frame = pd.read_csv(data_dir / "terms.csv", usecols=["term_id", "query"], dtype=str, keep_default_na=False)
        if ids is not None:
            frame = frame.loc[frame["term_id"].isin(ids)]
        frame = frame.sort_values("term_id")
        ordered = frame["term_id"].astype(str).tolist()
        return ordered, dict(zip(ordered, frame["query"].map(clean_text)))
    if entity == "items":
        frame = pd.read_csv(data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype=str, keep_default_na=False)
        if ids is not None:
            frame = frame.loc[frame["item_id"].isin(ids)]
        frame = frame.sort_values("item_id")
        ordered = frame["item_id"].astype(str).tolist()
        return ordered, {str(row.item_id): build_item_text(row) for row in frame.itertuples(index=False)}
    raise ValueError(f"Unknown entity: {entity}")


def encode(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, _, tokenizer, model, config = load_checkpoint(
        Path(args.model_dir), choose_device(require_transformer_stack()[0], args.device)
    )
    device = next(model.parameters()).device
    ids, text_by_id = load_entity_texts(Path(args.data_dir), args.entity)
    tower = "query" if args.entity == "terms" else "item"
    max_length = int(config["query_max_length"] if tower == "query" else config["item_max_length"])
    embeddings = encode_batches(
        ids, text_by_id, tokenizer, model, tower, max_length, args.batch_size, device, torch, args.fp16
    )
    id_column = "term_id" if args.entity == "terms" else "item_id"
    pd.DataFrame({id_column: ids}).to_csv(output_dir / f"{args.entity}_ids.csv", index=False)
    np.save(output_dir / f"{args.entity}_embeddings.npy", embeddings)
    manifest = {
        "entity": args.entity,
        "rows": len(ids),
        "dimension": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "checkpoint": str(Path(args.model_dir).resolve()),
        "checkpoint_config": config,
    }
    (output_dir / f"{args.entity}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def score_embedding_pairs(
    pairs: pd.DataFrame,
    term_ids: list[str],
    term_embeddings: np.ndarray,
    item_ids: list[str],
    item_embeddings: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    term_position = {value: position for position, value in enumerate(term_ids)}
    item_position = {value: position for position, value in enumerate(item_ids)}
    scores = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), chunk_size):
        stop = min(start + chunk_size, len(pairs))
        chunk = pairs.iloc[start:stop]
        query_index = np.fromiter((term_position[str(value)] for value in chunk["term_id"]), dtype=np.int64, count=len(chunk))
        item_index = np.fromiter((item_position[str(value)] for value in chunk["item_id"]), dtype=np.int64, count=len(chunk))
        query = np.asarray(term_embeddings[query_index], dtype=np.float32)
        item = np.asarray(item_embeddings[item_index], dtype=np.float32)
        scores[start:stop] = np.einsum("ij,ij->i", query, item)
    return scores


def score_pairs(args: argparse.Namespace) -> None:
    started = time.time()
    data_dir = Path(args.data_dir)
    pair_path = Path(args.pairs) if args.pairs else data_dir / "submission_pairs.csv"
    pairs = pd.read_csv(pair_path, dtype=str, keep_default_na=False, nrows=args.limit_pairs or None)
    required = {"term_id", "item_id"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"{pair_path} is missing {sorted(required - set(pairs.columns))}")
    id_column = args.id_column or ("id" if "id" in pairs else "slate_id" if "slate_id" in pairs else "")
    if not id_column:
        pairs["id"] = [f"PAIR_{index:010d}" for index in range(len(pairs))]
        id_column = "id"
    if id_column not in pairs:
        raise ValueError(f"Missing id column {id_column}")
    unique_terms = set(pairs["term_id"].astype(str))
    unique_items = set(pairs["item_id"].astype(str))
    term_ids, term_texts = load_entity_texts(data_dir, "terms", unique_terms)
    item_ids, item_texts = load_entity_texts(data_dir, "items", unique_items)
    if set(term_ids) != unique_terms or set(item_ids) != unique_items:
        raise KeyError("Some pair term/item IDs are missing from the source tables")

    torch_module = require_transformer_stack()[0]
    device = choose_device(torch_module, args.device)
    torch, _, tokenizer, model, config = load_checkpoint(Path(args.model_dir), device)
    term_embeddings = encode_batches(
        term_ids, term_texts, tokenizer, model, "query", int(config["query_max_length"]),
        args.encode_batch_size, device, torch, args.fp16,
    )
    item_embeddings = encode_batches(
        item_ids, item_texts, tokenizer, model, "item", int(config["item_max_length"]),
        args.encode_batch_size, device, torch, args.fp16,
    )
    scores = score_embedding_pairs(
        pairs, term_ids, term_embeddings, item_ids, item_embeddings, args.pair_chunk_size
    )
    output = pairs[[id_column, "term_id", "item_id"]].copy()
    output["semantic_cosine"] = scores
    output["semantic_rank_pct"] = output.groupby("term_id", sort=False)["semantic_cosine"].rank(
        method="average", pct=True
    ).astype(np.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    metrics = {
        "rows": len(output),
        "unique_ids": int(output[id_column].nunique()),
        "unique_terms": len(term_ids),
        "unique_items": len(item_ids),
        "elapsed_seconds": time.time() - started,
    }
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive MiniLM two-tower training and pair scoring.")
    sub = parser.add_subparsers(dest="operation", required=True)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--data-dir", default="data")
    train_cmd.add_argument("--negative-pool", default="outputs/hybrid_embedding/contrastive_negative_pool.csv")
    train_cmd.add_argument("--model-dir", default="outputs/hybrid_embedding/contrastive_model")
    train_cmd.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train_cmd.add_argument("--split-manifest")
    train_cmd.add_argument("--outer-fold", type=int, default=-1)
    train_cmd.add_argument("--cache-manifest")
    train_cmd.add_argument("--projection-dim", type=int, default=256)
    train_cmd.add_argument("--temperature", type=float, default=0.05)
    train_cmd.add_argument("--query-max-length", type=int, default=48)
    train_cmd.add_argument("--item-max-length", type=int, default=128)
    train_cmd.add_argument("--batch-size", type=int, default=64)
    train_cmd.add_argument("--epochs", type=int, default=2)
    train_cmd.add_argument("--inner-valid-size", type=float, default=0.10)
    train_cmd.add_argument("--uniform-multiplier", type=int, default=3)
    train_cmd.add_argument("--use-semi-hard", action=argparse.BooleanOptionalAction, default=True)
    train_cmd.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    train_cmd.add_argument("--learning-rate", type=float, default=2e-5)
    train_cmd.add_argument("--head-learning-rate", type=float, default=1e-4)
    train_cmd.add_argument("--weight-decay", type=float, default=0.01)
    train_cmd.add_argument("--warmup-ratio", type=float, default=0.06)
    train_cmd.add_argument("--max-grad-norm", type=float, default=1.0)
    train_cmd.add_argument("--grad-accum-steps", type=int, default=1)
    train_cmd.add_argument("--num-workers", type=int, default=0)
    train_cmd.add_argument("--device", default="auto")
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--limit-positives", type=int, default=0)
    train_cmd.add_argument("--max-steps", type=int, default=0)
    train_cmd.add_argument(
        "--refit-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After inner-term epoch selection, restart and refit that many epochs on all outer-training terms.",
    )
    train_cmd.set_defaults(func=train)

    encode_cmd = sub.add_parser("encode")
    encode_cmd.add_argument("--data-dir", default="data")
    encode_cmd.add_argument("--model-dir", default="outputs/hybrid_embedding/contrastive_model")
    encode_cmd.add_argument("--output-dir", default="outputs/hybrid_embedding/contrastive_embeddings")
    encode_cmd.add_argument("--entity", choices=["terms", "items"], required=True)
    encode_cmd.add_argument("--batch-size", type=int, default=512)
    encode_cmd.add_argument("--device", default="auto")
    encode_cmd.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    encode_cmd.set_defaults(func=encode)

    score_cmd = sub.add_parser("score-pairs")
    score_cmd.add_argument("--data-dir", default="data")
    score_cmd.add_argument("--model-dir", default="outputs/hybrid_embedding/contrastive_model")
    score_cmd.add_argument("--pairs")
    score_cmd.add_argument("--id-column")
    score_cmd.add_argument("--output", default="outputs/hybrid_embedding/contrastive_submission_scores.csv")
    score_cmd.add_argument("--encode-batch-size", type=int, default=512)
    score_cmd.add_argument("--pair-chunk-size", type=int, default=200_000)
    score_cmd.add_argument("--device", default="auto")
    score_cmd.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    score_cmd.add_argument("--limit-pairs", type=int, default=0)
    score_cmd.set_defaults(func=score_pairs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
