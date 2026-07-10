from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
SPACE_RE = re.compile(r"\s+")


def require_transformer_stack():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Transformer bi-encoder training/scoring requires torch and transformers. "
            "Install PyTorch in this environment or run this script on Kaggle GPU."
        ) from exc

    return torch, nn, F, DataLoader, Dataset, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return SPACE_RE.sub(" ", str(value).strip())


def build_item_text(row: Any) -> str:
    parts = [
        clean_text(getattr(row, "title", "")),
        clean_text(getattr(row, "brand", "")),
        clean_text(getattr(row, "category", "")),
        clean_text(getattr(row, "gender", "")),
        clean_text(getattr(row, "age_group", "")),
        clean_text(getattr(row, "attributes", "")),
    ]
    return " | ".join(part for part in parts if part)


def pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["term_id"].astype(str) + "\t" + frame["item_id"].astype(str)


def load_terms_texts(data_dir: Path, term_ids: set[str] | None = None) -> dict[str, str]:
    terms = pd.read_csv(data_dir / "terms.csv", dtype=str, keep_default_na=False)
    if term_ids is not None:
        terms = terms.loc[terms["term_id"].isin(term_ids)]
    return {str(row.term_id): clean_text(row.query) for row in terms.itertuples(index=False)}


def load_item_texts_for_ids(data_dir: Path, item_ids: set[str], chunksize: int) -> dict[str, str]:
    texts: dict[str, str] = {}
    reader = pd.read_csv(
        data_dir / "items.csv",
        usecols=ITEM_COLUMNS,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )
    for chunk in tqdm(reader, desc="load item texts"):
        matched = chunk.loc[chunk["item_id"].isin(item_ids)]
        for row in matched.itertuples(index=False):
            texts[str(row.item_id)] = build_item_text(row)
        if len(texts) == len(item_ids):
            break
    return texts


def load_first_item_ids(data_dir: Path, n_items: int) -> set[str]:
    if n_items <= 0:
        return set()
    items = pd.read_csv(data_dir / "items.csv", usecols=["item_id"], dtype=str, keep_default_na=False, nrows=n_items)
    return set(items["item_id"].astype(str))


def sample_labeled_pairs(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    if n_rows <= 0 or len(frame) <= n_rows:
        return frame

    pos = frame.loc[frame["label"].eq(1)]
    neg = frame.loc[frame["label"].eq(0)]
    n_pos = max(1, min(len(pos), round(n_rows * len(pos) / len(frame))))
    n_neg = max(1, min(len(neg), n_rows - n_pos))
    sampled = [
        pos.sample(n=n_pos, random_state=seed) if len(pos) > n_pos else pos,
        neg.sample(n=n_neg, random_state=seed + 1) if len(neg) > n_neg else neg,
    ]
    return pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def load_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    positives = pd.read_csv(data_dir / "training_pairs.csv", usecols=["term_id", "item_id"], dtype=str)
    positives = positives.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)
    positives["label"] = np.int8(1)

    negatives = pd.read_csv(args.negatives, usecols=["term_id", "item_id"], dtype=str, keep_default_na=False)
    negatives = negatives.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)
    keep = ~pair_key(negatives).isin(set(pair_key(positives)))
    dropped = int((~keep).sum())
    negatives = negatives.loc[keep].reset_index(drop=True)
    negatives["label"] = np.int8(0)

    if args.limit_terms:
        term_ids = set(positives["term_id"].drop_duplicates().head(args.limit_terms).astype(str))
        positives = positives.loc[positives["term_id"].isin(term_ids)]
        negatives = negatives.loc[negatives["term_id"].isin(term_ids)]

    if args.limit_items:
        item_ids = load_first_item_ids(data_dir, args.limit_items)
        positives = positives.loc[positives["item_id"].isin(item_ids)]
        negatives = negatives.loc[negatives["item_id"].isin(item_ids)]

    frame = pd.concat([positives, negatives], ignore_index=True)
    frame = sample_labeled_pairs(frame, args.limit_pairs, args.seed)
    frame = frame.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if frame["label"].nunique() != 2:
        raise ValueError("Training frame must contain both positive and negative rows after limits are applied")

    print(f"dropped_negative_positive_overlap={dropped:,}")
    print(
        f"training_rows={len(frame):,} positives={int(frame['label'].sum()):,} "
        f"positive_rate={frame['label'].mean():.4f}"
    )
    return frame


def load_slate_training_split(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.fold < 0:
        raise ValueError("--fold must be provided when --slates is used")

    slates = pd.read_csv(args.slates, dtype=str, keep_default_na=False)
    required = {"term_id", "item_id", "label", "fold"}
    missing = required - set(slates.columns)
    if missing:
        raise ValueError(f"{args.slates} is missing columns: {sorted(missing)}")
    if "slate_id" not in slates.columns:
        slates["slate_id"] = [f"SLATE_{index:010d}" for index in range(len(slates))]

    slates["fold"] = pd.to_numeric(slates["fold"], errors="raise").astype(np.int16)
    slates["label"] = pd.to_numeric(slates["label"], errors="raise").astype(np.int8)
    if not set(slates["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{args.slates} label values must be binary")
    available_folds = sorted(slates["fold"].unique().tolist())
    if args.fold not in available_folds:
        raise ValueError(f"fold={args.fold} not present in {args.slates}; available={available_folds}")

    if args.limit_terms:
        term_ids = set(slates["term_id"].drop_duplicates().head(args.limit_terms).astype(str))
        slates = slates.loc[slates["term_id"].isin(term_ids)]
    if args.limit_items:
        item_ids = load_first_item_ids(Path(args.data_dir), args.limit_items)
        slates = slates.loc[slates["item_id"].isin(item_ids)]

    train_frame = slates.loc[slates["fold"].ne(args.fold)].copy()
    valid_frame = slates.loc[slates["fold"].eq(args.fold)].copy()
    train_frame = sample_labeled_pairs(train_frame, args.limit_pairs, args.seed)
    valid_frame = sample_labeled_pairs(valid_frame, args.max_valid_pairs, args.seed + 1)
    train_frame = train_frame.sample(frac=1.0, random_state=args.seed + 2).reset_index(drop=True)
    valid_frame = valid_frame.reset_index(drop=True)

    if train_frame.empty or valid_frame.empty:
        raise ValueError("Slate fold split produced an empty training or validation frame")
    if train_frame["label"].nunique() != 2 or valid_frame["label"].nunique() != 2:
        raise ValueError("Slate training and validation frames must both contain labels 0 and 1")
    overlap = set(train_frame["term_id"].astype(str)) & set(valid_frame["term_id"].astype(str))
    if overlap:
        examples = sorted(overlap)[:5]
        raise AssertionError(f"Term leakage across slate split, examples={examples}")

    print(
        f"slate_fold={args.fold} train_rows={len(train_frame):,} valid_rows={len(valid_frame):,} "
        f"train_terms={train_frame['term_id'].nunique():,} valid_terms={valid_frame['term_id'].nunique():,}"
    )
    return train_frame, valid_frame


def split_inner_validation_by_term(
    frame: pd.DataFrame,
    valid_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < valid_size < 1.0:
        raise ValueError(f"--inner-valid-size must be between 0 and 1, got {valid_size}")
    term_ids = frame["term_id"].drop_duplicates().astype(str).to_numpy()
    if len(term_ids) < 2:
        raise ValueError("Need at least two outer-training terms for inner early stopping")
    rng = np.random.default_rng(seed)
    rng.shuffle(term_ids)
    n_valid_terms = min(len(term_ids) - 1, max(1, int(round(len(term_ids) * valid_size))))
    valid_term_ids = set(term_ids[:n_valid_terms])
    inner_valid = frame.loc[frame["term_id"].isin(valid_term_ids)].copy().reset_index(drop=True)
    inner_train = frame.loc[~frame["term_id"].isin(valid_term_ids)].copy().reset_index(drop=True)
    overlap = set(inner_train["term_id"].astype(str)) & set(inner_valid["term_id"].astype(str))
    if overlap:
        raise AssertionError("Term leakage across transformer inner validation split")
    if inner_train["label"].nunique() != 2 or inner_valid["label"].nunique() != 2:
        raise ValueError("Transformer inner training and validation frames must contain labels 0 and 1")
    return inner_train, inner_valid


def make_pair_head_class(torch, nn):
    class PairHead(nn.Module):
        def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embedding_dim * 4 + 1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, query_embedding, item_embedding):
            dot = (query_embedding * item_embedding).sum(dim=1, keepdim=True)
            features = [
                query_embedding,
                item_embedding,
                torch.abs(query_embedding - item_embedding),
                query_embedding * item_embedding,
                dot,
            ]
            return self.net(torch.cat(features, dim=1)).squeeze(1)

    return PairHead


def mean_pool(model_output, attention_mask, F):
    if hasattr(model_output, "last_hidden_state"):
        token_embeddings = model_output.last_hidden_state
    elif isinstance(model_output, dict):
        token_embeddings = model_output["last_hidden_state"]
    else:
        token_embeddings = model_output[0]
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def choose_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            return torch.device(f"cuda:{device_ids[0]}"), device_ids
        return torch.device("cpu"), []

    if requested == "cpu":
        return torch.device("cpu"), []

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {requested}, but CUDA is not available")
        if requested == "cuda":
            device_ids = list(range(torch.cuda.device_count()))
        elif "," in requested:
            device_ids = [int(value) for value in requested.replace("cuda:", "").split(",")]
        else:
            device_ids = [int(requested.split(":")[1])]

        if not device_ids:
            raise ValueError(f"No CUDA devices selected by --device {requested}")
        max_device = torch.cuda.device_count() - 1
        invalid = [device_id for device_id in device_ids if device_id < 0 or device_id > max_device]
        if invalid:
            raise ValueError(f"Invalid CUDA device ids {invalid}; available range is 0..{max_device}")
        return torch.device(f"cuda:{device_ids[0]}"), device_ids

    return torch.device(requested), []


def maybe_wrap_data_parallel(torch, module: Any, device_ids: list[int], name: str):
    if len(device_ids) > 1:
        print(f"using DataParallel for {name} on cuda devices {device_ids}")
        return torch.nn.DataParallel(module, device_ids=device_ids)
    return module


def unwrap_parallel(module: Any) -> Any:
    return module.module if hasattr(module, "module") else module


def train(args: argparse.Namespace) -> None:
    torch, nn, F, DataLoader, Dataset, AutoModel, AutoTokenizer, get_scheduler = require_transformer_stack()
    start_time = time.time()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    device, device_ids = choose_device(torch, args.device)
    print(f"device={device} device_ids={device_ids}")

    if args.slates:
        outer_train_frame, score_frame = load_slate_training_split(args)
        train_frame, valid_frame = split_inner_validation_by_term(
            outer_train_frame,
            args.inner_valid_size,
            args.seed + args.fold * 100,
        )
        frame = pd.concat([train_frame, valid_frame, score_frame], ignore_index=True)
        print(
            f"transformer model_fit_rows={len(train_frame):,} "
            f"inner_early_stop_rows={len(valid_frame):,} outer_score_rows={len(score_frame):,}"
        )
    else:
        frame = load_training_frame(args)
        rng = np.random.default_rng(args.seed)
        valid_mask = rng.random(len(frame)) < args.valid_size
        if valid_mask.all() or (~valid_mask).all():
            raise ValueError("Validation split produced an empty train or validation set")
        train_frame = frame.loc[~valid_mask].reset_index(drop=True)
        valid_frame = frame.loc[valid_mask].reset_index(drop=True)
        score_frame = valid_frame

    required_terms = set(frame["term_id"].astype(str))
    required_items = set(frame["item_id"].astype(str))
    term_texts = load_terms_texts(data_dir, required_terms)
    item_texts = load_item_texts_for_ids(data_dir, required_items, args.item_chunksize)
    missing_terms = required_terms - set(term_texts)
    missing_items = required_items - set(item_texts)
    if missing_terms or missing_items:
        raise KeyError(f"missing_terms={len(missing_terms)} missing_items={len(missing_items)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    embedding_dim = int(encoder.config.hidden_size)
    PairHead = make_pair_head_class(torch, nn)
    head = PairHead(embedding_dim=embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    encoder.to(device)
    head.to(device)
    encoder = maybe_wrap_data_parallel(torch, encoder, device_ids, "encoder")
    head = maybe_wrap_data_parallel(torch, head, device_ids, "head")

    class PairDataset(Dataset):
        def __init__(self, rows: pd.DataFrame):
            self.term_ids = rows["term_id"].astype(str).to_numpy()
            self.item_ids = rows["item_id"].astype(str).to_numpy()
            self.labels = rows["label"].astype("float32").to_numpy()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index: int):
            term_id = self.term_ids[index]
            item_id = self.item_ids[index]
            return term_texts[term_id], item_texts[item_id], float(self.labels[index])

    def collate(batch):
        query_text, item_text, labels = zip(*batch)
        query_batch = tokenizer(
            list(query_text),
            padding=True,
            truncation=True,
            max_length=args.max_query_length,
            return_tensors="pt",
        )
        item_batch = tokenizer(
            list(item_text),
            padding=True,
            truncation=True,
            max_length=args.max_item_length,
            return_tensors="pt",
        )
        return query_batch, item_batch, torch.tensor(labels, dtype=torch.float32)

    train_loader = DataLoader(
        PairDataset(train_frame),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        PairDataset(valid_frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    score_loader = DataLoader(
        PairDataset(score_frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {"params": head.parameters(), "weight_decay": args.weight_decay, "lr": args.head_learning_rate},
    ]
    optimizer = torch.optim.AdamW(grouped_params)
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    neg_count = int((train_frame["label"] == 0).sum())
    pos_count = int((train_frame["label"] == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / max(1, pos_count)], device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    def run_eval() -> tuple[float, np.ndarray]:
        encoder.eval()
        head.eval()
        total_loss = 0.0
        total_rows = 0
        probability_parts: list[np.ndarray] = []
        with torch.no_grad():
            for query_batch, item_batch, labels in tqdm(valid_loader, desc="valid", leave=False):
                query_batch = move_batch(query_batch, device)
                item_batch = move_batch(item_batch, device)
                labels = labels.to(device)
                q_emb = mean_pool(encoder(**query_batch), query_batch["attention_mask"], F)
                i_emb = mean_pool(encoder(**item_batch), item_batch["attention_mask"], F)
                logits = head(q_emb, i_emb)
                loss = criterion(logits, labels)
                total_loss += float(loss.detach().cpu()) * len(labels)
                total_rows += len(labels)
                probability_parts.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
        probabilities = np.concatenate(probability_parts) if probability_parts else np.empty(0, dtype=np.float32)
        return total_loss / max(1, total_rows), probabilities

    def predict_loader(loader: Any, desc: str) -> np.ndarray:
        encoder.eval()
        head.eval()
        probability_parts: list[np.ndarray] = []
        with torch.no_grad():
            for query_batch, item_batch, _ in tqdm(loader, desc=desc, leave=False):
                query_batch = move_batch(query_batch, device)
                item_batch = move_batch(item_batch, device)
                q_emb = mean_pool(encoder(**query_batch), query_batch["attention_mask"], F)
                i_emb = mean_pool(encoder(**item_batch), item_batch["attention_mask"], F)
                logits = head(q_emb, i_emb)
                probability_parts.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(probability_parts) if probability_parts else np.empty(0, dtype=np.float32)

    best_valid = float("inf")
    best_valid_probability: np.ndarray | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_rows = 0

        for step, (query_batch, item_batch, labels) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), start=1):
            query_batch = move_batch(query_batch, device)
            item_batch = move_batch(item_batch, device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                q_emb = mean_pool(encoder(**query_batch), query_batch["attention_mask"], F)
                i_emb = mean_pool(encoder(**item_batch), item_batch["attention_mask"], F)
                logits = head(q_emb, i_emb)
                loss = criterion(logits, labels) / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum_steps * len(labels)
            running_rows += len(labels)

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        valid_loss, valid_probability = run_eval()
        train_loss = running_loss / max(1, running_rows)
        print(f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_valid_probability = valid_probability.copy()
            best_epoch = epoch
            save_model(model_dir, encoder, tokenizer, head, args, embedding_dim)
            print(f"saved best model to {model_dir}")

    print(f"best_valid_loss={best_valid:.6f}")
    print(f"best_epoch={best_epoch}")
    if args.valid_output:
        if args.slates:
            del encoder, head, optimizer, scheduler, scaler, criterion, grouped_params
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            torch, F, tokenizer, encoder, head, _ = load_model(model_dir, device)
            encoder = maybe_wrap_data_parallel(torch, encoder, device_ids, "encoder")
            head = maybe_wrap_data_parallel(torch, head, device_ids, "head")
            output_probability = predict_loader(score_loader, "outer OOF score")
        else:
            if best_valid_probability is None:
                raise RuntimeError("Best validation probabilities were not captured correctly")
            output_probability = best_valid_probability
        if len(output_probability) != len(score_frame):
            raise RuntimeError("Best-checkpoint output probabilities do not align with the score frame")
        valid_output_path = Path(args.valid_output)
        valid_output_path.parent.mkdir(parents=True, exist_ok=True)
        output_columns = ["slate_id"] if "slate_id" in score_frame.columns else []
        if not output_columns:
            output_columns = [
                column
                for column in ["term_id", "item_id", "label"]
                if column in score_frame.columns
            ]
        valid_output = score_frame[output_columns].copy()
        valid_output["transformer_prob"] = output_probability
        valid_output.to_csv(valid_output_path, index=False)
        print(f"wrote {valid_output_path} rows={len(valid_output):,}")
    print(f"elapsed_minutes={(time.time() - start_time) / 60:.1f}")


def save_model(model_dir: Path, encoder: Any, tokenizer: Any, head: Any, args: argparse.Namespace, embedding_dim: int) -> None:
    import torch

    encoder_to_save = unwrap_parallel(encoder)
    head_to_save = unwrap_parallel(head)
    encoder_to_save.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(head_to_save.state_dict(), model_dir / "pair_head.pt")
    config = {
        "base_model": args.model_name,
        "embedding_dim": embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_query_length": args.max_query_length,
        "max_item_length": args.max_item_length,
    }
    (model_dir / "biencoder_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_model(model_dir: Path, device: Any):
    torch, nn, F, _, _, AutoModel, AutoTokenizer, _ = require_transformer_stack()
    config = json.loads((model_dir / "biencoder_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    PairHead = make_pair_head_class(torch, nn)
    head = PairHead(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    head.load_state_dict(torch.load(model_dir / "pair_head.pt", map_location=device))
    encoder.to(device)
    head.to(device)
    encoder.eval()
    head.eval()
    return torch, F, tokenizer, encoder, head, config


def encode_text_map(
    ids: list[str],
    text_by_id: dict[str, str],
    tokenizer: Any,
    encoder: Any,
    F: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    max_length: int,
    dtype: str,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(ids), batch_size), desc="encode texts"):
            batch_ids = ids[start : start + batch_size]
            texts = [text_by_id[item_id] for item_id in batch_ids]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = move_batch(encoded, device)
            embeddings = mean_pool(encoder(**encoded), encoded["attention_mask"], F)
            array = embeddings.detach().cpu().numpy()
            arrays.append(array.astype(np.float16 if dtype == "float16" else np.float32))
    return np.vstack(arrays)


def predict(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch, _, _, _, _, _, _, _ = require_transformer_stack()
    device, device_ids = choose_device(torch, args.device)
    print(f"device={device} device_ids={device_ids}")
    torch, F, tokenizer, encoder, head, config = load_model(model_dir, device)
    encoder = maybe_wrap_data_parallel(torch, encoder, device_ids, "encoder")
    head = maybe_wrap_data_parallel(torch, head, device_ids, "head")

    pairs_for_ids = pd.read_csv(
        data_dir / "submission_pairs.csv",
        usecols=["term_id", "item_id"],
        dtype=str,
        keep_default_na=False,
        nrows=args.limit_pairs or None,
    )
    term_ids = sorted(set(pairs_for_ids["term_id"].astype(str)))
    item_ids = sorted(set(pairs_for_ids["item_id"].astype(str)))
    print(f"unique_terms={len(term_ids):,} unique_items={len(item_ids):,}")

    term_texts = load_terms_texts(data_dir, set(term_ids))
    item_texts = load_item_texts_for_ids(data_dir, set(item_ids), args.item_chunksize)
    missing_terms = set(term_ids) - set(term_texts)
    missing_items = set(item_ids) - set(item_texts)
    if missing_terms or missing_items:
        raise KeyError(f"missing_terms={len(missing_terms)} missing_items={len(missing_items)}")

    term_embeddings = encode_text_map(
        term_ids,
        term_texts,
        tokenizer,
        encoder,
        F,
        torch,
        device,
        args.encode_batch_size,
        int(config["max_query_length"]),
        args.embedding_dtype,
    )
    item_embeddings = encode_text_map(
        item_ids,
        item_texts,
        tokenizer,
        encoder,
        F,
        torch,
        device,
        args.encode_batch_size,
        int(config["max_item_length"]),
        args.embedding_dtype,
    )
    term_pos = {term_id: i for i, term_id in enumerate(term_ids)}
    item_pos = {item_id: i for i, item_id in enumerate(item_ids)}
    del term_texts, item_texts, pairs_for_ids
    gc.collect()

    written = 0
    reader = pd.read_csv(
        data_dir / "submission_pairs.csv",
        usecols=["id", "term_id", "item_id"],
        dtype=str,
        keep_default_na=False,
        chunksize=args.pair_chunksize,
        nrows=args.limit_pairs or None,
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "term_id", "prob"])
        with torch.no_grad():
            for chunk in tqdm(reader, desc="score pairs"):
                q_idx = np.fromiter((term_pos[str(value)] for value in chunk["term_id"]), dtype=np.int64, count=len(chunk))
                i_idx = np.fromiter((item_pos[str(value)] for value in chunk["item_id"]), dtype=np.int64, count=len(chunk))
                probs: list[np.ndarray] = []
                for start in range(0, len(chunk), args.score_batch_size):
                    stop = min(start + args.score_batch_size, len(chunk))
                    q_emb = torch.from_numpy(term_embeddings[q_idx[start:stop]]).to(device=device, dtype=torch.float32)
                    i_emb = torch.from_numpy(item_embeddings[i_idx[start:stop]]).to(device=device, dtype=torch.float32)
                    logits = head(q_emb, i_emb)
                    probs.append(torch.sigmoid(logits).detach().cpu().numpy())
                prob_values = np.concatenate(probs)
                writer.writerows(
                    (row_id, term_id, f"{prob:.8f}")
                    for row_id, term_id, prob in zip(chunk["id"], chunk["term_id"], prob_values)
                )
                written += len(chunk)

    print(f"wrote {output_path} rows={written:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score a transformer bi-encoder expert.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--data-dir", default="data")
    train_cmd.add_argument("--negatives", default="outputs/train_term_negatives.csv")
    train_cmd.add_argument(
        "--slates",
        help="Grouped OOF validation slate CSV. When set, --fold selects held-out terms.",
    )
    train_cmd.add_argument("--fold", type=int, default=-1)
    train_cmd.add_argument("--valid-output", help="Write best-epoch held-out probabilities to this CSV.")
    train_cmd.add_argument("--model-dir", default="outputs/transformer_biencoder")
    train_cmd.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train_cmd.add_argument("--epochs", type=int, default=2)
    train_cmd.add_argument("--batch-size", type=int, default=64)
    train_cmd.add_argument("--grad-accum-steps", type=int, default=1)
    train_cmd.add_argument("--learning-rate", type=float, default=2e-5)
    train_cmd.add_argument("--head-learning-rate", type=float, default=1e-4)
    train_cmd.add_argument("--weight-decay", type=float, default=0.01)
    train_cmd.add_argument("--warmup-ratio", type=float, default=0.06)
    train_cmd.add_argument("--max-grad-norm", type=float, default=1.0)
    train_cmd.add_argument("--hidden-dim", type=int, default=256)
    train_cmd.add_argument("--dropout", type=float, default=0.1)
    train_cmd.add_argument("--valid-size", type=float, default=0.05)
    train_cmd.add_argument(
        "--inner-valid-size",
        type=float,
        default=0.10,
        help="With --slates, reserve this fraction of outer-training terms for checkpoint selection.",
    )
    train_cmd.add_argument("--max-query-length", type=int, default=48)
    train_cmd.add_argument("--max-item-length", type=int, default=192)
    train_cmd.add_argument("--item-chunksize", type=int, default=100_000)
    train_cmd.add_argument("--num-workers", type=int, default=0)
    train_cmd.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or cuda:0,1")
    train_cmd.add_argument("--fp16", action="store_true")
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--limit-pairs", type=int, default=0)
    train_cmd.add_argument("--max-valid-pairs", type=int, default=0)
    train_cmd.add_argument("--limit-terms", type=int, default=0)
    train_cmd.add_argument("--limit-items", type=int, default=0)
    train_cmd.set_defaults(func=train)

    pred_cmd = sub.add_parser("predict")
    pred_cmd.add_argument("--data-dir", default="data")
    pred_cmd.add_argument("--model-dir", default="outputs/transformer_biencoder")
    pred_cmd.add_argument("--output", default="outputs/transformer_biencoder_scores.csv")
    pred_cmd.add_argument("--encode-batch-size", type=int, default=512)
    pred_cmd.add_argument("--score-batch-size", type=int, default=8192)
    pred_cmd.add_argument("--pair-chunksize", type=int, default=200_000)
    pred_cmd.add_argument("--item-chunksize", type=int, default=100_000)
    pred_cmd.add_argument("--embedding-dtype", choices=["float32", "float16"], default="float16")
    pred_cmd.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or cuda:0,1")
    pred_cmd.add_argument("--limit-pairs", type=int, default=0)
    pred_cmd.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
