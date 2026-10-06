"""Learn query-to-category priors for category-aware relevance models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from lexical_baseline import normalize_text

CATEGORY_LEVELS = ("cat_l1", "cat_l2")
CATEGORY_TOPK = 5


def split_category(value: object) -> tuple[str, str, str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        parts = ["unknown"]
    else:
        parts = [normalize_text(part) for part in str(value).split("/")]
        parts = [part for part in parts if part]
        if not parts:
            parts = ["unknown"]

    cat_l1 = parts[0]
    cat_l2 = "/".join(parts[:2]) if len(parts) >= 2 else cat_l1
    cat_l3 = "/".join(parts[:3]) if len(parts) >= 3 else cat_l2
    cat_leaf = "/".join(parts)
    return cat_l1, cat_l2, cat_l3, cat_leaf


def add_category_levels(
    frame: pd.DataFrame, category_col: str = "category"
) -> pd.DataFrame:
    levels = pd.DataFrame(
        frame[category_col].map(split_category).tolist(),
        columns=["cat_l1", "cat_l2", "cat_l3", "cat_leaf"],
        index=frame.index,
    )
    return pd.concat([frame, levels], axis=1)


def build_category_examples(
    terms: pd.DataFrame,
    items: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    max_examples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    required_term_ids = set(positive_pairs["term_id"].astype(str))
    required_item_ids = set(positive_pairs["item_id"].astype(str))
    term_subset = terms.loc[
        terms["term_id"].astype(str).isin(required_term_ids), ["term_id", "query"]
    ]
    item_subset = items.loc[
        items["item_id"].astype(str).isin(required_item_ids), ["item_id", "category"]
    ]

    examples = (
        positive_pairs[["term_id", "item_id"]]
        .merge(term_subset, on="term_id", how="left", validate="many_to_one")
        .merge(item_subset, on="item_id", how="left", validate="many_to_one")
        .dropna(subset=["query", "category"])
    )
    examples = add_category_levels(examples)
    examples["query_norm"] = examples["query"].map(normalize_text)

    # A broad query can have hundreds of positive products in the same category.
    # Keep one query/category path so it does not dominate the category classifier.
    examples = examples.drop_duplicates(["query_norm", "cat_l2"])
    if max_examples is not None and len(examples) > max_examples:
        examples = examples.sample(n=max_examples, random_state=seed)

    columns = ["query_norm", "cat_l1", "cat_l2", "cat_l3", "cat_leaf"]
    return examples[columns].reset_index(drop=True)


@dataclass
class ConstantCategoryModel:
    """Return a fixed category distribution when training has one class."""

    label: str

    def __post_init__(self) -> None:
        self.classes_ = np.asarray([self.label], dtype=object)

    def predict_proba(self, matrix: Any) -> np.ndarray:
        return np.ones((matrix.shape[0], 1), dtype=np.float32)


class CategorySignal:
    """Fit TF-IDF category classifiers and return each query's top categories."""

    def __init__(
        self,
        levels: tuple[str, ...] = CATEGORY_LEVELS,
        topk: int = CATEGORY_TOPK,
        max_features: int = 120_000,
        max_iter: int = 80,
        min_class_count: int = 2,
        seed: int = 42,
    ):
        self.levels = levels
        self.topk = topk
        self.max_features = max_features
        self.max_iter = max_iter
        self.min_class_count = min_class_count
        self.seed = seed
        self.vectorizer: TfidfVectorizer | None = None
        self.models: dict[str, Any] = {}

    def fit(self, examples: pd.DataFrame) -> "CategorySignal":
        if examples.empty:
            raise ValueError("CategorySignal requires at least one training example")

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2 if len(examples) >= 2 else 1,
            max_features=self.max_features,
            dtype=np.float32,
            sublinear_tf=True,
        )
        x_all = self.vectorizer.fit_transform(examples["query_norm"])
        print(f"category tfidf shape={x_all.shape}")

        for level in self.levels:
            counts = examples[level].value_counts()
            keep_labels = counts.index[counts >= self.min_class_count]
            if len(keep_labels) == 0:
                keep_labels = counts.index[:1]
            keep = examples[level].isin(keep_labels).to_numpy()
            y = examples.loc[keep, level].astype(str).to_numpy()
            x = x_all[keep]
            classes = np.unique(y)

            start = time.time()
            if len(classes) == 1:
                model: Any = ConstantCategoryModel(str(classes[0]))
            else:
                model = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=2e-5,
                    max_iter=self.max_iter,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=self.seed,
                    n_jobs=-1,
                )
                model.fit(x, y)
            self.models[level] = model
            elapsed = time.time() - start
            print(
                f"{level}: classes={len(classes):,} examples={len(y):,} seconds={elapsed:.1f}"
            )
        return self

    def predict_terms(
        self, terms: pd.DataFrame, chunk_size: int = 25_000
    ) -> pd.DataFrame:
        if self.vectorizer is None:
            raise RuntimeError("fit CategorySignal before predicting")
        if terms["term_id"].duplicated().any():
            raise ValueError("terms contains duplicate term_id values")

        term_ids = terms["term_id"].astype(str).to_numpy()
        queries = terms["query"].map(normalize_text).to_numpy()
        chunks: list[pd.DataFrame] = []

        for start in tqdm(range(0, len(terms), chunk_size), desc="category topk"):
            end = min(start + chunk_size, len(terms))
            matrix = self.vectorizer.transform(queries[start:end])
            out: dict[str, object] = {"term_id": term_ids[start:end]}

            for level, model in self.models.items():
                proba = model.predict_proba(matrix)
                k = min(self.topk, proba.shape[1])
                if k == proba.shape[1]:
                    indices = np.argsort(-proba, axis=1)[:, :k]
                else:
                    indices = np.argpartition(proba, -k, axis=1)[:, -k:]
                    local_order = np.argsort(
                        -np.take_along_axis(proba, indices, axis=1), axis=1
                    )
                    indices = np.take_along_axis(indices, local_order, axis=1)
                scores = np.take_along_axis(proba, indices, axis=1).astype(np.float32)
                labels = model.classes_[indices]

                for rank in range(k):
                    out[f"{level}_top{rank + 1}"] = labels[:, rank]
                    out[f"{level}_prob{rank + 1}"] = scores[:, rank]

            chunks.append(pd.DataFrame(out))

        if not chunks:
            return pd.DataFrame(columns=["term_id"])
        return pd.concat(chunks, ignore_index=True)
