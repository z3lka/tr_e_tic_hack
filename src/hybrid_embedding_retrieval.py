"""Cache lexical and semantic retrieval, mine negatives, and build validation slates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_SCHEMA_VERSION = 1
TEXT_FORMAT_VERSION = "title|brand|category|gender|age_group|attributes:v1"
ITEM_COLUMNS = [
    "item_id",
    "title",
    "category",
    "brand",
    "gender",
    "age_group",
    "attributes",
]
NEGATIVE_COLUMNS = [
    "term_id",
    "item_id",
    "negative_source",
    "source_rank",
    "source_score",
]
BASE_SLATE_COLUMNS = [
    "slate_id",
    "term_id",
    "item_id",
    "fold",
    "label",
    "candidate_count",
    "is_retrieved",
    "retrieval_rank",
    "retrieval_score",
]
SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9çğıöşü]+")


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return SPACE_RE.sub(" ", str(value).strip())


def normalize_text(value: object) -> str:
    text = clean_text(value).lower().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return SPACE_RE.sub(" ", NON_WORD_RE.sub(" ", text)).strip()


def build_item_text(row: Any) -> str:
    parts = [clean_text(getattr(row, column, "")) for column in ITEM_COLUMNS[1:]]
    return " | ".join(part for part in parts if part)


def item_search_text(items: pd.DataFrame) -> pd.Series:
    columns = [
        column
        for column in ["title", "brand", "category", "attributes"]
        if column in items
    ]
    output = pd.Series("", index=items.index, dtype=object)
    for column in columns:
        values = items[column].map(normalize_text)
        output = output.str.cat(values, sep=" ")
    return output.str.strip()


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def file_fingerprint(path: Path, sample_bytes: int = 1 << 20) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }


def canonical_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "manifest": cache_dir / "manifest.json",
        "item_ids": cache_dir / "item_ids.csv",
        "term_ids": cache_dir / "term_ids.csv",
        "item_embeddings": cache_dir / "item_embeddings.npy",
        "term_embeddings": cache_dir / "term_embeddings.npy",
        "item_tfidf": cache_dir / "item_tfidf.npz",
        "term_tfidf": cache_dir / "term_tfidf.npz",
        "vectorizer": cache_dir / "tfidf_vectorizer.joblib",
        "retrieval": cache_dir / "retrieval_topk.npz",
    }


def load_manifest(cache_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    path = cache_paths(cache_dir)["manifest"]
    if not path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cache schema in {path}: {manifest.get('schema_version')}"
        )
    if require_complete and not manifest.get("complete"):
        raise ValueError(f"Cache is incomplete: {path}")
    return manifest


def cache_is_valid(cache_dir: Path, expected_key: str | None = None) -> bool:
    try:
        manifest = load_manifest(cache_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    if expected_key is not None and manifest.get("cache_key") != expected_key:
        return False
    paths = cache_paths(cache_dir)
    required = [
        paths["item_ids"],
        paths["term_ids"],
        paths["item_embeddings"],
        paths["term_embeddings"],
        paths["item_tfidf"],
        paths["term_tfidf"],
        paths["retrieval"],
    ]
    return all(path.exists() for path in required)


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
    return torch.device(requested)


def mean_pool(last_hidden_state: Any, attention_mask: Any, torch: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def encode_pretrained_texts(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("build-cache requires torch and transformers") from exc

    device = choose_device(torch, device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    arrays: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(texts), batch_size), desc="frozen MiniLM encoding"
        ):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=torch.float16
            ):
                output = model(**batch)
                embedding = mean_pool(
                    output.last_hidden_state, batch["attention_mask"], torch
                )
            arrays.append(embedding.float().cpu().numpy().astype(np.float16))
    if not arrays:
        return np.empty((0, int(model.config.hidden_size)), dtype=np.float16)
    return np.concatenate(arrays, axis=0)


def _deterministic_topk(
    scores: np.ndarray, indices: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape != indices.shape:
        raise ValueError("scores and indices must have identical shapes")
    take = min(k, scores.shape[1])
    out_scores = np.empty((scores.shape[0], take), dtype=np.float32)
    out_indices = np.empty((scores.shape[0], take), dtype=np.int64)
    for row in range(scores.shape[0]):
        if take < scores.shape[1]:
            # Argpartition alone may choose arbitrary members of a tie at the
            # cutoff. Keep all strictly better scores, then the smallest item
            # indices at the boundary so CPU/GPU block sizes cannot alter IDs.
            threshold = np.partition(scores[row], -take)[-take]
            strict = np.flatnonzero(scores[row] > threshold)
            boundary = np.flatnonzero(scores[row] == threshold)
            boundary_order = np.argsort(indices[row, boundary], kind="stable")
            needed = take - len(strict)
            selected = np.concatenate([strict, boundary[boundary_order[:needed]]])
        else:
            selected = np.arange(scores.shape[1])
        order = np.lexsort((indices[row, selected], -scores[row, selected]))
        selected = selected[order]
        out_scores[row] = scores[row, selected]
        out_indices[row] = indices[row, selected]
    return out_indices, out_scores


def exact_cosine_topk(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    k: int,
    query_block_size: int = 64,
    item_block_size: int = 16_384,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic exact cosine top-k over already L2-normalized arrays."""
    if query_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")
    if query_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError("Query/item embedding dimensions differ")
    if not 0 < k <= len(item_embeddings):
        raise ValueError(f"k must be in 1..{len(item_embeddings)}, got {k}")

    use_torch = device != "cpu"
    torch = None
    torch_device = None
    if use_torch:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("GPU retrieval requires torch") from exc
        torch = torch_module
        torch_device = choose_device(torch, device)

    all_indices = np.empty((len(query_embeddings), k), dtype=np.int32)
    all_scores = np.empty((len(query_embeddings), k), dtype=np.float32)
    for q_start in tqdm(
        range(0, len(query_embeddings), query_block_size), desc="exact cosine top-k"
    ):
        q_stop = min(q_start + query_block_size, len(query_embeddings))
        query = np.asarray(query_embeddings[q_start:q_stop], dtype=np.float32)
        kept_indices = np.empty((len(query), 0), dtype=np.int64)
        kept_scores = np.empty((len(query), 0), dtype=np.float32)
        if use_torch:
            query_tensor = torch.from_numpy(query).to(torch_device)
        for i_start in range(0, len(item_embeddings), item_block_size):
            i_stop = min(i_start + item_block_size, len(item_embeddings))
            item = np.asarray(item_embeddings[i_start:i_stop], dtype=np.float32)
            if use_torch:
                item_tensor = torch.from_numpy(item).to(torch_device)
                block_scores = (query_tensor @ item_tensor.T).float().cpu().numpy()
                del item_tensor
            else:
                block_scores = query @ item.T
            block_indices = np.broadcast_to(
                np.arange(i_start, i_stop, dtype=np.int64)[None, :], block_scores.shape
            )
            merged_scores = np.concatenate([kept_scores, block_scores], axis=1)
            merged_indices = np.concatenate([kept_indices, block_indices], axis=1)
            kept_indices, kept_scores = _deterministic_topk(
                merged_scores, merged_indices, k
            )
        all_indices[q_start:q_stop] = kept_indices.astype(np.int32)
        all_scores[q_start:q_stop] = kept_scores
    return all_indices, all_scores


def sparse_cosine_topk(
    term_matrix: sparse.csr_matrix,
    item_matrix: sparse.csr_matrix,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < k <= item_matrix.shape[0]:
        raise ValueError(f"k must be in 1..{item_matrix.shape[0]}, got {k}")
    all_indices = np.empty((term_matrix.shape[0], k), dtype=np.int32)
    all_scores = np.empty((term_matrix.shape[0], k), dtype=np.float32)
    item_matrix_t = item_matrix.T.tocsr()
    for row in tqdm(range(term_matrix.shape[0]), desc="exact TF-IDF top-k"):
        product = (term_matrix.getrow(row) @ item_matrix_t).tocsr()
        dense_indices = product.indices.astype(np.int64)
        dense_scores = product.data.astype(np.float32)
        if len(dense_indices):
            order = np.lexsort((dense_indices, -dense_scores))
            dense_indices = dense_indices[order]
            dense_scores = dense_scores[order]
        take = min(k, len(dense_indices))
        selected_indices = dense_indices[:take].tolist()
        selected_scores = dense_scores[:take].tolist()
        if take < k:
            selected_set = set(selected_indices)
            for position in range(item_matrix.shape[0]):
                if position in selected_set:
                    continue
                selected_indices.append(position)
                selected_scores.append(0.0)
                if len(selected_indices) == k:
                    break
        all_indices[row] = np.asarray(selected_indices, dtype=np.int32)
        all_scores[row] = np.asarray(selected_scores, dtype=np.float32)
    return all_indices, all_scores


def save_retrieval_pool(
    path: Path,
    lexical_indices: np.ndarray,
    lexical_scores: np.ndarray,
    embedding_indices: np.ndarray,
    embedding_scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        lexical_indices=np.asarray(lexical_indices, dtype=np.int32),
        lexical_scores=np.asarray(lexical_scores, dtype=np.float16),
        embedding_indices=np.asarray(embedding_indices, dtype=np.int32),
        embedding_scores=np.asarray(embedding_scores, dtype=np.float16),
    )


def load_retrieval_pool(cache_dir: Path) -> dict[str, np.ndarray]:
    manifest = load_manifest(cache_dir)
    pool = np.load(cache_paths(cache_dir)["retrieval"], mmap_mode="r")
    required = {
        "lexical_indices",
        "lexical_scores",
        "embedding_indices",
        "embedding_scores",
    }
    missing = required - set(pool.files)
    if missing:
        raise ValueError(f"Retrieval pool is missing arrays: {sorted(missing)}")
    topk = int(manifest["retrieval_topk"])
    if any(
        pool[name].shape[1] < topk for name in ["lexical_indices", "embedding_indices"]
    ):
        raise ValueError("Retrieval pool shape does not match its manifest")
    return {name: pool[name] for name in required}


def write_embedding_cache_fixture(
    cache_dir: Path,
    item_ids: list[str],
    term_ids: list[str],
    item_embeddings: np.ndarray,
    term_embeddings: np.ndarray,
    lexical_indices: np.ndarray,
    lexical_scores: np.ndarray,
    embedding_indices: np.ndarray,
    embedding_scores: np.ndarray,
) -> None:
    """Write a tiny complete cache for deterministic tests and smoke runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    pd.DataFrame({"item_id": item_ids}).to_csv(paths["item_ids"], index=False)
    pd.DataFrame({"term_id": term_ids}).to_csv(paths["term_ids"], index=False)
    np.save(paths["item_embeddings"], np.asarray(item_embeddings, dtype=np.float16))
    np.save(paths["term_embeddings"], np.asarray(term_embeddings, dtype=np.float16))
    identity_items = sparse.eye(len(item_ids), dtype=np.float32, format="csr")
    identity_terms = sparse.csr_matrix((len(term_ids), len(item_ids)), dtype=np.float32)
    sparse.save_npz(paths["item_tfidf"], identity_items)
    sparse.save_npz(paths["term_tfidf"], identity_terms)
    save_retrieval_pool(
        paths["retrieval"],
        lexical_indices,
        lexical_scores,
        embedding_indices,
        embedding_scores,
    )
    payload: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": "synthetic-fixture",
        "text_format": TEXT_FORMAT_VERSION,
        "query_max_length": 48,
        "item_max_length": 128,
        "item_count": len(item_ids),
        "term_count": len(term_ids),
        "embedding_dim": int(item_embeddings.shape[1]),
        "embedding_dtype": "float16",
        "retrieval_topk": int(lexical_indices.shape[1]),
        "source_data": {},
        "complete": True,
    }
    payload["cache_key"] = canonical_fingerprint(payload)
    paths["manifest"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_cache(args: argparse.Namespace) -> None:
    started = time.time()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    config: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": args.model_name,
        "text_format": TEXT_FORMAT_VERSION,
        "query_max_length": args.query_max_length,
        "item_max_length": args.item_max_length,
        "retrieval_topk": args.retrieval_topk,
        "tfidf_max_features": args.tfidf_max_features,
        "tfidf_min_df": args.tfidf_min_df,
        "source_data": {
            filename: file_fingerprint(data_dir / filename)
            for filename in ["items.csv", "terms.csv"]
        },
        "limits": {"terms": args.limit_terms, "items": args.limit_items},
    }
    cache_key = canonical_fingerprint(config)
    if not args.force and cache_is_valid(cache_dir, cache_key):
        print(f"resume: valid cache already exists at {cache_dir} key={cache_key[:12]}")
        return
    if paths["manifest"].exists() and not args.force:
        previous = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if previous.get("cache_key") != cache_key:
            raise ValueError(
                "Cache configuration changed "
                f"({previous.get('cache_key', '')[:12]} -> {cache_key[:12]}). "
                "Use a new --cache-dir or pass --force."
            )
        print(f"resume incomplete cache at {cache_dir} key={cache_key[:12]}")

    incomplete = {
        **config,
        "cache_key": cache_key,
        "complete": False,
        "started_at": time.time(),
    }
    paths["manifest"].write_text(json.dumps(incomplete, indent=2), encoding="utf-8")
    terms = (
        pd.read_csv(
            data_dir / "terms.csv",
            usecols=["term_id", "query"],
            dtype=str,
            keep_default_na=False,
            nrows=args.limit_terms or None,
        )
        .sort_values("term_id")
        .reset_index(drop=True)
    )
    items = (
        pd.read_csv(
            data_dir / "items.csv",
            usecols=ITEM_COLUMNS,
            dtype=str,
            keep_default_na=False,
            nrows=args.limit_items or None,
        )
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    if len(items) < args.retrieval_topk:
        raise ValueError("Catalog is smaller than retrieval-topk")
    term_ids = terms["term_id"].astype(str).tolist()
    item_ids = items["item_id"].astype(str).tolist()
    term_texts = terms["query"].map(clean_text).tolist()
    item_texts = [build_item_text(row) for row in items.itertuples(index=False)]

    embedding_stage_complete = all(
        paths[name].exists()
        for name in ["term_ids", "item_ids", "term_embeddings", "item_embeddings"]
    )
    if embedding_stage_complete and not args.force:
        cached_term_ids = pd.read_csv(paths["term_ids"], dtype=str)["term_id"].tolist()
        cached_item_ids = pd.read_csv(paths["item_ids"], dtype=str)["item_id"].tolist()
        if cached_term_ids == term_ids and cached_item_ids == item_ids:
            term_embeddings = np.load(paths["term_embeddings"], mmap_mode="r")
            item_embeddings = np.load(paths["item_embeddings"], mmap_mode="r")
            print("resume completed embedding stage")
        else:
            embedding_stage_complete = False
    if not embedding_stage_complete or args.force:
        term_embeddings = encode_pretrained_texts(
            term_texts,
            args.model_name,
            args.query_max_length,
            args.encode_batch_size,
            args.device,
        )
        item_embeddings = encode_pretrained_texts(
            item_texts,
            args.model_name,
            args.item_max_length,
            args.encode_batch_size,
            args.device,
        )
        pd.DataFrame({"term_id": term_ids}).to_csv(paths["term_ids"], index=False)
        pd.DataFrame({"item_id": item_ids}).to_csv(paths["item_ids"], index=False)
        np.save(paths["term_embeddings"], term_embeddings)
        np.save(paths["item_embeddings"], item_embeddings)

    term_norm = terms["query"].map(normalize_text)
    item_norm = item_search_text(items)
    tfidf_stage_complete = all(
        paths[name].exists() for name in ["vectorizer", "term_tfidf", "item_tfidf"]
    )
    if tfidf_stage_complete and not args.force:
        term_tfidf = sparse.load_npz(paths["term_tfidf"]).tocsr()
        item_tfidf = sparse.load_npz(paths["item_tfidf"]).tocsr()
        if term_tfidf.shape[0] == len(terms) and item_tfidf.shape[0] == len(items):
            print("resume completed TF-IDF stage")
        else:
            tfidf_stage_complete = False
    if not tfidf_stage_complete or args.force:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=args.tfidf_min_df,
            max_features=args.tfidf_max_features,
            dtype=np.float32,
            norm="l2",
            sublinear_tf=True,
        )
        vectorizer.fit(pd.concat([term_norm, item_norm], ignore_index=True))
        term_tfidf = vectorizer.transform(term_norm).tocsr()
        item_tfidf = vectorizer.transform(item_norm).tocsr()
        joblib.dump(vectorizer, paths["vectorizer"])
        sparse.save_npz(paths["term_tfidf"], term_tfidf)
        sparse.save_npz(paths["item_tfidf"], item_tfidf)

    if paths["retrieval"].exists() and not args.force:
        existing_pool = np.load(paths["retrieval"])
        expected_shape = (len(terms), args.retrieval_topk)
        retrieval_stage_complete = all(
            existing_pool[name].shape == expected_shape
            for name in [
                "lexical_indices",
                "lexical_scores",
                "embedding_indices",
                "embedding_scores",
            ]
        )
    else:
        retrieval_stage_complete = False
    if retrieval_stage_complete:
        print("resume completed exact retrieval stage")
    else:
        embedding_indices, embedding_scores = exact_cosine_topk(
            term_embeddings,
            item_embeddings,
            args.retrieval_topk,
            args.query_block_size,
            args.item_block_size,
            args.device,
        )
        lexical_indices, lexical_scores = sparse_cosine_topk(
            term_tfidf, item_tfidf, args.retrieval_topk
        )
        save_retrieval_pool(
            paths["retrieval"],
            lexical_indices,
            lexical_scores,
            embedding_indices,
            embedding_scores,
        )
    manifest = {
        **config,
        "cache_key": cache_key,
        "complete": True,
        "item_count": len(items),
        "term_count": len(terms),
        "embedding_dim": int(item_embeddings.shape[1]),
        "embedding_dtype": "float16",
        "elapsed_seconds": time.time() - started,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote frozen retrieval cache {cache_dir} key={cache_key[:12]}")


def load_positive_pairs(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        data_dir / "training_pairs.csv",
        usecols=lambda column: column in {"term_id", "item_id", "label"},
        dtype=str,
        keep_default_na=False,
    )
    if "label" in frame:
        frame = frame.loc[frame["label"].ne("0")]
    output = frame[["term_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    if output.empty:
        raise ValueError("No positive training pairs found")
    return output


class ConservativeNegativeFilter:
    """Reject known positives and near-duplicate items during negative mining."""

    def __init__(self, items: pd.DataFrame, positives: pd.DataFrame):
        self.item_ids = items["item_id"].astype(str).to_numpy()
        self.position_by_id = {
            item_id: position for position, item_id in enumerate(self.item_ids)
        }
        self.keys = np.asarray(
            [
                f"{normalize_text(row.title)}\t{normalize_text(row.brand)}"
                for row in items.itertuples(index=False)
            ],
            dtype=object,
        )
        self.tokens = [
            frozenset(normalize_text(value).split()) for value in items["title"]
        ]
        self.positive_ids: dict[str, set[str]] = {}
        self.positive_keys: dict[str, set[str]] = {}
        self.positive_tokens: dict[str, list[frozenset[str]]] = {}
        for term_id, group in positives.groupby("term_id", sort=False):
            ids = set(group["item_id"].astype(str))
            positions = [
                self.position_by_id[item_id]
                for item_id in ids
                if item_id in self.position_by_id
            ]
            term = str(term_id)
            self.positive_ids[term] = ids
            self.positive_keys[term] = {
                str(self.keys[position]) for position in positions
            }
            self.positive_tokens[term] = [
                self.tokens[position] for position in positions
            ]

    @staticmethod
    def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def valid(self, term_id: str, position: int) -> bool:
        item_id = str(self.item_ids[position])
        if item_id in self.positive_ids.get(term_id, set()):
            return False
        if str(self.keys[position]) in self.positive_keys.get(term_id, set()):
            return False
        tokens = self.tokens[position]
        return all(
            self.jaccard(tokens, positive) < 0.90
            for positive in self.positive_tokens.get(term_id, [])
        )


def _rank_band_candidates(
    term_id: str,
    ranked_positions: np.ndarray,
    ranked_scores: np.ndarray,
    low_rank: int,
    high_rank: int,
    negative_filter: ConservativeNegativeFilter,
    selected: set[int],
) -> list[tuple[int, int, float]]:
    output: list[tuple[int, int, float]] = []
    stop = min(high_rank, len(ranked_positions))
    for zero_index in range(low_rank - 1, stop):
        position = int(ranked_positions[zero_index])
        if position in selected or not negative_filter.valid(term_id, position):
            continue
        output.append((position, zero_index + 1, float(ranked_scores[zero_index])))
    return output


def _sample_rank_band(
    candidates: list[tuple[int, int, float]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, float]]:
    if len(candidates) <= count:
        return candidates
    chosen = np.sort(rng.choice(len(candidates), size=count, replace=False))
    return [candidates[int(index)] for index in chosen]


def _sample_uniform_positions(
    term_id: str,
    count: int,
    negative_filter: ConservativeNegativeFilter,
    selected: set[int],
    rng: np.random.Generator,
) -> list[int]:
    output: list[int] = []
    attempts = 0
    maximum = max(1_000, count * 500)
    while len(output) < count and attempts < maximum:
        position = int(rng.integers(0, len(negative_filter.item_ids)))
        attempts += 1
        if position in selected or not negative_filter.valid(term_id, position):
            continue
        selected.add(position)
        output.append(position)
    if len(output) < count:
        offset = stable_seed(0, term_id, "fallback") % len(negative_filter.item_ids)
        for step in range(len(negative_filter.item_ids)):
            position = int((offset + step) % len(negative_filter.item_ids))
            if position in selected or not negative_filter.valid(term_id, position):
                continue
            selected.add(position)
            output.append(position)
            if len(output) == count:
                break
    if len(output) != count:
        raise ValueError(
            f"Could not sample {count} valid unique negatives for term {term_id}"
        )
    return output


def mine_negative_rows(
    term_ids: Iterable[str],
    term_position: dict[str, int],
    pool: dict[str, np.ndarray],
    negative_filter: ConservativeNegativeFilter,
    seed: int,
    recipe: str,
) -> list[dict[str, object]]:
    if recipe == "contrastive":
        source_specs = [
            ("tfidf_semi_hard", "lexical", 101, 500, 10),
            ("embedding_semi_hard", "embedding", 101, 500, 10),
        ]
        uniform_count = 0
    elif recipe == "reranker":
        source_specs = [
            ("tfidf_near_hard", "lexical", 21, 200, 15),
            ("embedding_near_hard", "embedding", 21, 200, 15),
        ]
        uniform_count = 20
    else:
        raise ValueError(f"Unknown negative recipe: {recipe}")

    rows: list[dict[str, object]] = []
    for term_id in tqdm(
        sorted(set(str(value) for value in term_ids)), desc=f"mine {recipe} negatives"
    ):
        if term_id not in term_position:
            raise KeyError(f"Term {term_id} is absent from the retrieval cache")
        row_position = term_position[term_id]
        selected: set[int] = set()
        term_rng = np.random.default_rng(stable_seed(seed, recipe, term_id))
        if uniform_count:
            for position in _sample_uniform_positions(
                term_id, uniform_count, negative_filter, selected, term_rng
            ):
                rows.append(
                    {
                        "term_id": term_id,
                        "item_id": negative_filter.item_ids[position],
                        "negative_source": "uniform_random",
                        "source_rank": 0,
                        "source_score": 0.0,
                    }
                )

        for source_name, array_prefix, low_rank, high_rank, count in source_specs:
            ranked_positions = pool[f"{array_prefix}_indices"][row_position]
            ranked_scores = pool[f"{array_prefix}_scores"][row_position]
            eligible = _rank_band_candidates(
                term_id,
                ranked_positions,
                ranked_scores,
                low_rank,
                high_rank,
                negative_filter,
                selected,
            )
            chosen = _sample_rank_band(eligible, count, term_rng)
            for position, rank, score in chosen:
                selected.add(position)
                rows.append(
                    {
                        "term_id": term_id,
                        "item_id": negative_filter.item_ids[position],
                        "negative_source": source_name,
                        "source_rank": rank,
                        "source_score": score,
                    }
                )
            shortage = count - len(chosen)
            if shortage:
                for position in _sample_uniform_positions(
                    term_id, shortage, negative_filter, selected, term_rng
                ):
                    rows.append(
                        {
                            "term_id": term_id,
                            "item_id": negative_filter.item_ids[position],
                            "negative_source": f"uniform_backfill_{array_prefix}",
                            "source_rank": 0,
                            "source_score": 0.0,
                        }
                    )
    return rows


def write_negative_output(
    rows: list[dict[str, object]],
    output: Path,
    positives: pd.DataFrame,
    recipe: str,
    seed: int,
) -> None:
    frame = pd.DataFrame(rows, columns=NEGATIVE_COLUMNS)
    if frame.duplicated(["term_id", "item_id"]).any():
        raise AssertionError("Duplicate mined negative pairs")
    positive_keys = set(
        positives["term_id"].astype(str) + "\t" + positives["item_id"].astype(str)
    )
    mined_keys = frame["term_id"].astype(str) + "\t" + frame["item_id"].astype(str)
    if mined_keys.isin(positive_keys).any():
        raise AssertionError("A mined negative overlaps a known positive")
    ranked = frame.loc[frame["source_rank"].astype(int).gt(0)]
    if (
        recipe == "contrastive"
        and not ranked["source_rank"].astype(int).between(101, 500).all()
    ):
        raise AssertionError("Contrastive negative rank outside 101..500")
    if (
        recipe == "reranker"
        and not ranked["source_rank"].astype(int).between(21, 200).all()
    ):
        raise AssertionError("Reranker negative rank outside 21..200")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    source_counts = Counter(frame["negative_source"].astype(str))
    audit = {
        "recipe": recipe,
        "seed": seed,
        "rows": len(frame),
        "terms": int(frame["term_id"].nunique()),
        "duplicates": int(frame.duplicated(["term_id", "item_id"]).sum()),
        "positive_overlaps": int(mined_keys.isin(positive_keys).sum()),
        "source_counts": dict(sorted(source_counts.items())),
        "source_ratios": {
            key: value / max(1, len(frame))
            for key, value in sorted(source_counts.items())
        },
    }
    output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"wrote {output} rows={len(frame):,}")


def load_cache_ids(cache_dir: Path) -> tuple[list[str], list[str]]:
    paths = cache_paths(cache_dir)
    item_ids = (
        pd.read_csv(paths["item_ids"], dtype=str, keep_default_na=False)["item_id"]
        .astype(str)
        .tolist()
    )
    term_ids = (
        pd.read_csv(paths["term_ids"], dtype=str, keep_default_na=False)["term_id"]
        .astype(str)
        .tolist()
    )
    return item_ids, term_ids


def run_miner(args: argparse.Namespace, recipe: str) -> None:
    started = time.time()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    manifest = load_manifest(cache_dir)
    required_rank = 500 if recipe == "contrastive" else 200
    if int(manifest["retrieval_topk"]) < required_rank:
        raise ValueError(f"{recipe} mining requires retrieval_topk >= {required_rank}")
    item_ids, term_ids = load_cache_ids(cache_dir)
    term_position = {term_id: position for position, term_id in enumerate(term_ids)}
    positives = load_positive_pairs(data_dir)
    positives = positives.loc[positives["term_id"].isin(term_position)].reset_index(
        drop=True
    )
    if args.limit_terms:
        keep_terms = sorted(positives["term_id"].unique())[: args.limit_terms]
        positives = positives.loc[positives["term_id"].isin(keep_terms)].reset_index(
            drop=True
        )
    items = pd.read_csv(
        data_dir / "items.csv",
        usecols=["item_id", "title", "brand"],
        dtype=str,
        keep_default_na=False,
    )
    items = items.set_index("item_id").reindex(item_ids).reset_index()
    if items[["title", "brand"]].isna().any().any():
        raise KeyError("Some cached catalog IDs are missing from items.csv")
    negative_filter = ConservativeNegativeFilter(items, positives)
    pool = load_retrieval_pool(cache_dir)
    rows = mine_negative_rows(
        positives["term_id"].unique(),
        term_position,
        pool,
        negative_filter,
        args.seed,
        recipe,
    )
    output_path = Path(args.output)
    write_negative_output(rows, output_path, positives, recipe, args.seed)
    audit_path = output_path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["cache_fingerprint"] = manifest["cache_key"]
    audit["elapsed_seconds"] = time.time() - started
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def make_term_split_manifest(
    positives: pd.DataFrame, mode: str, seed: int
) -> pd.DataFrame:
    counts = (
        positives.groupby("term_id", sort=True)
        .size()
        .rename("n_positives")
        .reset_index()
    )
    counts["fold"] = np.int16(-1)
    if mode == "pilot":
        if len(counts) < 2:
            raise ValueError("Pilot split requires at least two terms")
        n_bins = max(1, min(10, len(counts) // 5))
        bins = pd.qcut(
            counts["n_positives"].rank(method="first"),
            q=n_bins,
            labels=False,
            duplicates="drop",
        ).astype(int)
        n_holdout = min(len(counts) - 1, max(1, int(round(0.20 * len(counts)))))
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=n_holdout, random_state=seed
            )
            train_index, holdout_index = next(
                splitter.split(np.zeros(len(counts)), bins)
            )
        except ValueError:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(counts))
            holdout_index, train_index = order[:n_holdout], order[n_holdout:]
        counts.loc[train_index, "fold"] = np.int16(1)
        counts.loc[holdout_index, "fold"] = np.int16(0)
        counts["split"] = np.where(counts["fold"].eq(0), "outer_holdout", "outer_train")
    elif mode == "confirm":
        if len(counts) < 5:
            raise ValueError("Confirm split requires at least five terms")
        n_bins = max(1, min(10, len(counts) // 5))
        bins = pd.qcut(
            counts["n_positives"].rank(method="first"),
            q=n_bins,
            labels=False,
            duplicates="drop",
        ).astype(int)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (_, valid_index) in enumerate(
            splitter.split(np.zeros(len(counts)), bins)
        ):
            counts.loc[valid_index, "fold"] = np.int16(fold)
        counts["split"] = "oof"
    elif mode == "final":
        counts["fold"] = np.int16(-1)
        counts["split"] = "all_train"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if counts["fold"].lt(-1).any():
        raise AssertionError("Invalid term fold assignment")
    return counts[["term_id", "fold", "split", "n_positives"]]


def build_slates(args: argparse.Namespace) -> None:
    started = time.time()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache_dir)
    if int(manifest["retrieval_topk"]) < 100:
        raise ValueError("Hybrid slates require retrieval_topk >= 100 for Recall@100")
    item_ids, term_ids = load_cache_ids(cache_dir)
    item_position = {item_id: position for position, item_id in enumerate(item_ids)}
    term_position = {term_id: position for position, term_id in enumerate(term_ids)}
    positives = load_positive_pairs(data_dir)
    positives = positives.loc[
        positives["term_id"].isin(term_position)
        & positives["item_id"].isin(item_position)
    ].reset_index(drop=True)
    if args.limit_terms:
        kept = sorted(positives["term_id"].unique())[: args.limit_terms]
        positives = positives.loc[positives["term_id"].isin(kept)].reset_index(
            drop=True
        )
    split = make_term_split_manifest(positives, args.mode, args.seed)
    split_path = output_dir / "split_manifest.csv"
    split.to_csv(split_path, index=False)
    fold_by_term = split.set_index("term_id")["fold"].astype(int).to_dict()
    split_by_term = split.set_index("term_id")["split"].astype(str).to_dict()

    positive_positions = {
        str(term_id): [
            item_position[item_id] for item_id in group["item_id"].astype(str)
        ]
        for term_id, group in positives.groupby("term_id", sort=False)
    }
    pool = load_retrieval_pool(cache_dir)
    term_embeddings = np.load(cache_paths(cache_dir)["term_embeddings"], mmap_mode="r")
    item_embeddings = np.load(cache_paths(cache_dir)["item_embeddings"], mmap_mode="r")
    rows: list[dict[str, object]] = []
    retrieval_counts = Counter()
    total_positives = len(positives)

    for term_id in tqdm(sorted(positive_positions), desc="build hybrid slates"):
        tpos = term_position[term_id]
        lexical = [int(value) for value in pool["lexical_indices"][tpos, :40]]
        embedding = [int(value) for value in pool["embedding_indices"][tpos, :40]]
        lexical_meta = {
            position: (rank, float(pool["lexical_scores"][tpos, rank - 1]))
            for rank, position in enumerate(lexical, 1)
        }
        embedding_meta = {
            position: (rank, float(pool["embedding_scores"][tpos, rank - 1]))
            for rank, position in enumerate(embedding, 1)
        }
        positives_for_term = set(positive_positions[term_id])
        selected = set(lexical) | set(embedding)
        rng = np.random.default_rng(stable_seed(args.seed, args.mode, "slate", term_id))
        uniform: list[int] = []
        while len(uniform) < 20:
            position = int(rng.integers(0, len(item_ids)))
            if position in selected or position in positives_for_term:
                continue
            selected.add(position)
            uniform.append(position)
        base_positions = list(dict.fromkeys([*lexical, *embedding, *uniform]))
        appended = sorted(positives_for_term - set(base_positions))
        candidates = base_positions + appended
        cosine = np.asarray(item_embeddings[candidates], dtype=np.float32) @ np.asarray(
            term_embeddings[tpos], dtype=np.float32
        )
        semantic_pct = (
            pd.Series(cosine)
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float32)
        )
        lexical_100 = set(int(value) for value in pool["lexical_indices"][tpos, :100])
        embedding_100 = set(
            int(value) for value in pool["embedding_indices"][tpos, :100]
        )
        retrieval_counts["tfidf_hits"] += len(positives_for_term & lexical_100)
        retrieval_counts["embedding_hits"] += len(positives_for_term & embedding_100)
        retrieval_counts["hybrid_hits"] += len(
            positives_for_term & (lexical_100 | embedding_100)
        )

        for local_index, position in enumerate(candidates):
            sources: list[str] = []
            if position in lexical_meta:
                sources.append("tfidf")
            if position in embedding_meta:
                sources.append("embedding")
            if position in uniform:
                sources.append("uniform")
            if position in appended:
                sources.append("positive_union")
            lexical_rank, lexical_score = lexical_meta.get(position, (0, 0.0))
            embedding_rank, _ = embedding_meta.get(position, (0, 0.0))
            rows.append(
                {
                    "slate_id": f"HYBRID_{len(rows):010d}",
                    "term_id": term_id,
                    "item_id": item_ids[position],
                    "fold": fold_by_term[term_id],
                    "label": int(position in positives_for_term),
                    "candidate_count": len(candidates),
                    "is_retrieved": int(bool(lexical_rank or embedding_rank)),
                    "retrieval_rank": lexical_rank,
                    "retrieval_score": lexical_score,
                    "semantic_cosine": float(cosine[local_index]),
                    "semantic_rank_pct": float(semantic_pct[local_index]),
                    "candidate_sources": "+".join(sources),
                    "tfidf_rank": lexical_rank,
                    "embedding_rank": embedding_rank,
                    "split": split_by_term[term_id],
                }
            )
    slate = pd.DataFrame(rows)
    if int(slate["label"].sum()) != total_positives:
        raise AssertionError(
            "Every known positive must occur exactly once in its validation slate"
        )
    slate_path = output_dir / "hybrid_validation_slates.csv"
    slate.to_csv(slate_path, index=False)
    metrics = {
        "mode": args.mode,
        "seed": args.seed,
        "rows": len(slate),
        "terms": int(slate["term_id"].nunique()),
        "positives": total_positives,
        "tfidf_recall_at_100": retrieval_counts["tfidf_hits"] / max(1, total_positives),
        "frozen_embedding_recall_at_100": retrieval_counts["embedding_hits"]
        / max(1, total_positives),
        "hybrid_recall_at_100": retrieval_counts["hybrid_hits"]
        / max(1, total_positives),
        "cache_fingerprint": manifest["cache_key"],
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"wrote {slate_path}")


def add_common_mining_arguments(
    parser: argparse.ArgumentParser, default_output: str
) -> None:
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="outputs/hybrid_embedding/frozen_cache")
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-terms", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cached exact hybrid retrieval and conservative negative mining."
    )
    sub = parser.add_subparsers(dest="operation", required=True)

    cache_cmd = sub.add_parser("build-cache")
    cache_cmd.add_argument("--data-dir", default="data")
    cache_cmd.add_argument(
        "--cache-dir", default="outputs/hybrid_embedding/frozen_cache"
    )
    cache_cmd.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    cache_cmd.add_argument("--query-max-length", type=int, default=48)
    cache_cmd.add_argument("--item-max-length", type=int, default=128)
    cache_cmd.add_argument("--encode-batch-size", type=int, default=512)
    cache_cmd.add_argument("--retrieval-topk", type=int, default=500)
    cache_cmd.add_argument("--query-block-size", type=int, default=64)
    cache_cmd.add_argument("--item-block-size", type=int, default=16_384)
    cache_cmd.add_argument("--tfidf-max-features", type=int, default=250_000)
    cache_cmd.add_argument("--tfidf-min-df", type=int, default=2)
    cache_cmd.add_argument("--device", default="auto")
    cache_cmd.add_argument("--limit-terms", type=int, default=0)
    cache_cmd.add_argument("--limit-items", type=int, default=0)
    cache_cmd.add_argument("--force", action="store_true")
    cache_cmd.set_defaults(func=build_cache)

    contrastive_cmd = sub.add_parser("mine-contrastive")
    add_common_mining_arguments(
        contrastive_cmd, "outputs/hybrid_embedding/contrastive_negative_pool.csv"
    )
    contrastive_cmd.set_defaults(func=lambda args: run_miner(args, "contrastive"))

    reranker_cmd = sub.add_parser("mine-reranker")
    add_common_mining_arguments(
        reranker_cmd, "outputs/hybrid_embedding/reranker_negatives.csv"
    )
    reranker_cmd.set_defaults(func=lambda args: run_miner(args, "reranker"))

    slate_cmd = sub.add_parser("build-slates")
    slate_cmd.add_argument("--data-dir", default="data")
    slate_cmd.add_argument(
        "--cache-dir", default="outputs/hybrid_embedding/frozen_cache"
    )
    slate_cmd.add_argument("--output-dir", default="outputs/hybrid_embedding/pilot")
    slate_cmd.add_argument(
        "--mode", choices=["pilot", "confirm", "final"], default="pilot"
    )
    slate_cmd.add_argument("--seed", type=int, default=42)
    slate_cmd.add_argument("--limit-terms", type=int, default=0)
    slate_cmd.set_defaults(func=build_slates)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
