"""Run leakage-aware term-grouped validation for tree and transformer models."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import time
from pathlib import Path
from typing import Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from tqdm import tqdm

from category_signal import CategorySignal, build_category_examples
from pu_catboost import (
    BASE_FEATURE_NAMES,
    CAT_FEATURES,
    CATEGORY_FEATURE_NAMES,
    build_item_meta_lookup,
    build_term_category_lookup,
    rows_to_frame,
)
from lexical_baseline import load_items, load_terms, normalize_text
from train_term_negatives import item_search_text, topk_sparse_row

SLATE_COLUMNS = [
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
OPTIONAL_SEMANTIC_FEATURES = ["semantic_cosine", "semantic_rank_pct"]


def load_positive_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "label" in frame.columns:
        frame = frame.loc[frame["label"].astype(str).ne("0")]
    frame = frame[["term_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No positive pairs found in {path}")
    return frame


def make_term_folds(positives: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    counts = (
        positives.groupby("term_id", sort=True)
        .size()
        .rename("n_positives")
        .reset_index()
    )
    if len(counts) < n_splits:
        raise ValueError(f"Need at least {n_splits} terms, found {len(counts)}")

    # Stratifying quantiles of query breadth distributes both head and tail queries
    # across folds while term_id grouping prevents any query leakage.
    n_bins = max(1, min(10, len(counts) // n_splits))
    bins = pd.qcut(
        counts["n_positives"].rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)
    counts["fold"] = np.int16(-1)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(counts), dtype=np.int8)
    for fold, (_, valid_index) in enumerate(splitter.split(dummy, bins)):
        counts.loc[valid_index, "fold"] = np.int16(fold)

    if counts["fold"].lt(0).any():
        raise AssertionError("Some terms were not assigned to a fold")
    return counts[["term_id", "fold", "n_positives"]]


def sample_zero_score_items(
    item_count: int,
    selected: set[int],
    needed: int,
    rng: np.random.Generator,
) -> list[int]:
    output: list[int] = []
    max_attempts = max(1_000, needed * 100)
    attempts = 0
    while len(output) < needed and attempts < max_attempts:
        position = int(rng.integers(0, item_count))
        attempts += 1
        if position in selected:
            continue
        selected.add(position)
        output.append(position)
    if len(output) != needed:
        for position in range(item_count):
            if position in selected:
                continue
            selected.add(position)
            output.append(position)
            if len(output) == needed:
                break
    if len(output) != needed:
        raise ValueError(f"Could not fill a {needed}-item zero-score retrieval tail")
    return output


def restrict_debug_catalog(
    items: pd.DataFrame,
    positives: pd.DataFrame,
    limit_items: int,
) -> pd.DataFrame:
    if limit_items <= 0 or len(items) <= limit_items:
        return items
    required = set(positives["item_id"].astype(str))
    keep = items["item_id"].astype(str).isin(required)
    head_positions = np.arange(len(items)) < limit_items
    restricted = (
        items.loc[keep | head_positions]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )
    print(
        f"debug catalog restricted from {len(items):,} to {len(restricted):,}; "
        f"retained_positive_items={len(required):,}"
    )
    return restricted


def build_slates(args: argparse.Namespace) -> None:
    started = time.time()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = load_positive_pairs(data_dir / "training_pairs.csv")
    terms = pd.read_csv(data_dir / "terms.csv", dtype=str, keep_default_na=False)
    train_terms = terms.loc[
        terms["term_id"].isin(set(positives["term_id"])), ["term_id", "query"]
    ].copy()
    train_terms = train_terms.sort_values("term_id").reset_index(drop=True)
    if args.limit_terms:
        train_terms = train_terms.head(args.limit_terms).copy()
        positives = positives.loc[
            positives["term_id"].isin(set(train_terms["term_id"]))
        ].reset_index(drop=True)

    missing_terms = set(positives["term_id"]) - set(train_terms["term_id"])
    if missing_terms:
        raise KeyError(f"Missing {len(missing_terms)} positive term ids from terms.csv")

    folds = make_term_folds(positives, args.n_splits, args.seed)
    folds_path = output_dir / "term_folds.csv"
    folds.to_csv(folds_path, index=False)
    fold_by_term = folds.set_index("term_id")["fold"].astype(int).to_dict()

    items = pd.read_csv(
        data_dir / "items.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["item_id", "title", "category", "brand", "attributes"],
    )
    items = restrict_debug_catalog(items, positives, args.limit_items)
    if len(items) < args.base_candidates:
        raise ValueError(
            f"Catalog has {len(items)} items, fewer than base_candidates={args.base_candidates}"
        )

    item_ids = items["item_id"].astype(str).to_numpy()
    item_to_position = {item_id: position for position, item_id in enumerate(item_ids)}
    missing_items = set(positives["item_id"]) - set(item_to_position)
    if missing_items:
        raise KeyError(f"Missing {len(missing_items)} positive item ids from items.csv")

    train_terms["query_norm"] = train_terms["query"].map(normalize_text)
    item_text = item_search_text(items)
    corpus = pd.concat([train_terms["query_norm"], item_text], ignore_index=True)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, args.word_ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
        dtype=np.float32,
        norm="l2",
        sublinear_tf=True,
    )
    vectorizer.fit(corpus)
    joblib.dump(vectorizer, output_dir / "retrieval_vectorizer.joblib")
    term_matrix = vectorizer.transform(train_terms["query_norm"])
    item_matrix_t = vectorizer.transform(item_text).T.tocsr()
    print(
        f"terms={len(train_terms):,} catalog_items={len(items):,} positives={len(positives):,} "
        f"vocab={len(vectorizer.vocabulary_):,}"
    )

    positives_by_term: dict[str, list[int]] = {}
    for term_id, group in positives.groupby("term_id", sort=False):
        positives_by_term[str(term_id)] = [
            item_to_position[item_id] for item_id in group["item_id"].astype(str)
        ]

    slate_path = output_dir / "validation_slates.csv"
    rng = np.random.default_rng(args.seed)
    row_id = 0
    positive_rows = 0
    retrieved_positive_rows = 0
    rows_by_fold = np.zeros(args.n_splits, dtype=np.int64)

    with slate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SLATE_COLUMNS)

        for start in tqdm(
            range(0, len(train_terms), args.chunk_size), desc="build validation slates"
        ):
            stop = min(start + args.chunk_size, len(train_terms))
            score_chunk = (term_matrix[start:stop] @ item_matrix_t).tocsr()

            for row_offset, term_id in enumerate(
                train_terms["term_id"].iloc[start:stop].astype(str)
            ):
                indices, scores = topk_sparse_row(
                    score_chunk.getrow(row_offset), args.base_candidates
                )
                retrieved_positions = [
                    int(value) for value in indices[: args.base_candidates]
                ]
                retrieved_scores = [
                    float(value) for value in scores[: args.base_candidates]
                ]
                selected = set(retrieved_positions)
                missing = args.base_candidates - len(retrieved_positions)
                if missing:
                    fill = sample_zero_score_items(len(items), selected, missing, rng)
                    retrieved_positions.extend(fill)
                    retrieved_scores.extend([0.0] * len(fill))

                positive_positions = positives_by_term[term_id]
                positive_set = set(positive_positions)
                appended_positions = sorted(positive_set - set(retrieved_positions))
                candidate_positions = retrieved_positions + appended_positions
                candidate_count = len(candidate_positions)
                fold = int(fold_by_term[term_id])

                retrieval_metadata = {
                    position: (rank, score)
                    for rank, (position, score) in enumerate(
                        zip(retrieved_positions, retrieved_scores),
                        start=1,
                    )
                }
                for position in candidate_positions:
                    label = int(position in positive_set)
                    rank, retrieval_score = retrieval_metadata.get(position, (0, 0.0))
                    is_retrieved = int(rank > 0)
                    writer.writerow(
                        [
                            f"OOF_{row_id:010d}",
                            term_id,
                            item_ids[position],
                            fold,
                            label,
                            candidate_count,
                            is_retrieved,
                            rank,
                            f"{retrieval_score:.8f}",
                        ]
                    )
                    row_id += 1
                    positive_rows += label
                    retrieved_positive_rows += label * is_retrieved
                    rows_by_fold[fold] += 1

    if positive_rows != len(positives):
        raise AssertionError(
            f"Expected {len(positives):,} positive slate rows, wrote {positive_rows:,}"
        )

    summary = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "base_candidates": args.base_candidates,
        "retrieval_max_features": args.max_features,
        "retrieval_min_df": args.min_df,
        "retrieval_word_ngram_max": args.word_ngram_max,
        "terms": len(train_terms),
        "catalog_items": len(items),
        "positive_pairs": len(positives),
        "slate_rows": row_id,
        "positive_rate": positive_rows / row_id,
        "retrieved_positive_recall": retrieved_positive_rows / positive_rows,
        "rows_by_fold": rows_by_fold.tolist(),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (output_dir / "slate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {folds_path}")
    print(f"wrote {slate_path}")


def sample_rows(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    positive = frame.loc[frame["label"].eq(1)]
    negative = frame.loc[frame["label"].eq(0)]
    positive_share = len(positive) / len(frame)
    n_positive = min(len(positive), max(1, int(round(maximum * positive_share))))
    n_negative = min(len(negative), maximum - n_positive)
    parts = [
        (
            positive.sample(n=n_positive, random_state=seed)
            if len(positive) > n_positive
            else positive
        ),
        (
            negative.sample(n=n_negative, random_state=seed + 1)
            if len(negative) > n_negative
            else negative
        ),
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed + 2)
        .reset_index(drop=True)
    )


def fit_category_signal_for_fold(
    data_dir: Path,
    output_dir: Path,
    train_term_ids: set[str],
    slate_term_ids: set[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    terms = pd.read_csv(
        data_dir / "terms.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["term_id", "query"],
    )
    items = pd.read_csv(
        data_dir / "items.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["item_id", "category"],
    )
    positives = load_positive_pairs(data_dir / "training_pairs.csv")
    positives = positives.loc[positives["term_id"].isin(train_term_ids)].reset_index(
        drop=True
    )
    examples = build_category_examples(
        terms,
        items,
        positives,
        max_examples=args.category_max_examples or None,
        seed=args.seed,
    )
    signal = CategorySignal(
        topk=args.category_topk,
        max_features=args.category_max_features,
        max_iter=args.category_max_iter,
        min_class_count=args.category_min_class_count,
        seed=args.seed,
    ).fit(examples)
    prediction_terms = terms.loc[
        terms["term_id"].isin(slate_term_ids), ["term_id", "query"]
    ].copy()
    topk = signal.predict_terms(prediction_terms, chunk_size=args.category_chunk_size)
    topk_path = output_dir / "term_category_topk.csv"
    topk.to_csv(topk_path, index=False)
    joblib.dump(signal, output_dir / "category_signal.joblib")
    print(f"wrote {topk_path}")
    return topk


def train_catboost(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_inner_valid: pd.DataFrame | None,
    y_inner_valid: np.ndarray | None,
    x_score: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[CatBoostClassifier, np.ndarray]:
    params: dict[str, object] = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": args.catboost_iterations,
        "learning_rate": args.catboost_learning_rate,
        "depth": args.catboost_depth,
        "l2_leaf_reg": args.catboost_l2_leaf_reg,
        "random_strength": args.catboost_random_strength,
        "bootstrap_type": "Bernoulli",
        "subsample": args.catboost_subsample,
        "auto_class_weights": "Balanced",
        "random_seed": args.seed,
        "thread_count": args.threads,
        "allow_writing_files": False,
        "verbose": args.verbose,
        "task_type": args.task_type,
        "devices": args.devices,
    }
    if args.task_type == "CPU":
        params["rsm"] = args.catboost_rsm

    train_pool = Pool(x_fit, y_fit, cat_features=CAT_FEATURES)
    score_pool = Pool(x_score, cat_features=CAT_FEATURES)
    model = CatBoostClassifier(**params)
    if x_inner_valid is not None and y_inner_valid is not None:
        inner_valid_pool = Pool(x_inner_valid, y_inner_valid, cat_features=CAT_FEATURES)
        model.fit(
            train_pool,
            eval_set=inner_valid_pool,
            use_best_model=True,
            early_stopping_rounds=args.early_stopping_rounds,
        )
    else:
        model.fit(train_pool)
    probability = model.predict_proba(score_pool)[:, 1].astype(np.float32)
    model.save_model(str(output_dir / "catboost.cbm"))
    return model, probability


def train_lgbm(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_inner_valid: pd.DataFrame | None,
    y_inner_valid: np.ndarray | None,
    x_score: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    numeric_features = BASE_FEATURE_NAMES + CATEGORY_FEATURE_NAMES
    numeric_features.extend(
        feature for feature in OPTIONAL_SEMANTIC_FEATURES if feature in x_fit.columns
    )
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=args.lgbm_min_child_samples,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        reg_lambda=args.lgbm_reg_lambda,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.threads,
    )
    callbacks: list[object] = [lgb.log_evaluation(period=args.verbose)]
    fit_kwargs: dict[str, object] = {"callbacks": callbacks}
    if x_inner_valid is not None and y_inner_valid is not None:
        if args.early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(args.early_stopping_rounds, verbose=True)
            )
        fit_kwargs["eval_set"] = [(x_inner_valid[numeric_features], y_inner_valid)]
        fit_kwargs["eval_metric"] = "binary_logloss"
    model.fit(x_fit[numeric_features], y_fit, **fit_kwargs)
    probability = model.predict_proba(x_score[numeric_features])[:, 1].astype(
        np.float32
    )
    joblib.dump(
        {"model": model, "features": numeric_features}, output_dir / "lgbm.joblib"
    )
    return model, probability


def inner_group_split(
    rows: pd.DataFrame,
    valid_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_size <= 0.0:
        return np.arange(len(rows), dtype=np.int64), np.empty(0, dtype=np.int64)
    if not 0.0 < valid_size < 1.0:
        raise ValueError(f"inner-valid-size must be in [0, 1), got {valid_size}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    fit_index, inner_valid_index = next(
        splitter.split(rows, rows["label"], groups=rows["term_id"])
    )
    if set(rows.iloc[fit_index]["term_id"]) & set(
        rows.iloc[inner_valid_index]["term_id"]
    ):
        raise AssertionError("Term leakage across inner early-stopping split")
    return fit_index, inner_valid_index


def run_fold(args: argparse.Namespace, fold: int | None = None) -> Path:
    started = time.time()
    selected_fold = int(args.fold if fold is None else fold)
    data_dir = Path(args.data_dir)
    root_output_dir = Path(args.output_dir)
    output_dir = root_output_dir / f"fold_{selected_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    slates = pd.read_csv(
        args.slates or root_output_dir / "validation_slates.csv",
        dtype={"slate_id": str, "term_id": str, "item_id": str},
        keep_default_na=False,
    )
    required = set(SLATE_COLUMNS)
    missing = required - set(slates.columns)
    if missing:
        raise ValueError(f"Slate file is missing columns: {sorted(missing)}")
    for column in [
        "fold",
        "label",
        "candidate_count",
        "is_retrieved",
        "retrieval_rank",
    ]:
        slates[column] = pd.to_numeric(slates[column], errors="raise").astype(np.int32)
    slates["retrieval_score"] = pd.to_numeric(
        slates["retrieval_score"], errors="raise"
    ).astype(np.float32)

    available_folds = sorted(slates["fold"].unique().tolist())
    if selected_fold not in available_folds:
        raise ValueError(
            f"fold={selected_fold} not present; available={available_folds}"
        )
    train_rows = slates.loc[slates["fold"].ne(selected_fold)].copy()
    valid_rows = slates.loc[slates["fold"].eq(selected_fold)].copy()
    train_rows = sample_rows(
        train_rows, args.max_train_rows, args.seed + selected_fold * 10
    )
    valid_rows = sample_rows(
        valid_rows, args.max_valid_rows, args.seed + selected_fold * 10 + 1
    )
    if train_rows.empty or valid_rows.empty:
        raise ValueError("Fold split produced empty training or validation rows")
    print(
        f"fold={selected_fold} train_rows={len(train_rows):,} valid_rows={len(valid_rows):,} "
        f"train_terms={train_rows['term_id'].nunique():,} "
        f"valid_terms={valid_rows['term_id'].nunique():,}"
    )

    train_term_ids = set(train_rows["term_id"].astype(str))
    slate_term_ids = set(slates["term_id"].astype(str))
    topk = fit_category_signal_for_fold(
        data_dir,
        output_dir,
        train_term_ids,
        slate_term_ids,
        args,
    )
    term_category_lookup = build_term_category_lookup(topk)

    terms = load_terms(data_dir / "terms.csv")
    items = load_items(data_dir / "items.csv")
    item_meta_lookup = build_item_meta_lookup(data_dir / "items.csv")
    x_train = rows_to_frame(
        train_rows,
        terms,
        items,
        term_category_lookup,
        item_meta_lookup,
        f"fold {selected_fold} train features",
    )
    x_valid = rows_to_frame(
        valid_rows,
        terms,
        items,
        term_category_lookup,
        item_meta_lookup,
        f"fold {selected_fold} valid features",
    )
    semantic_features: list[str] = []
    if args.semantic_features:
        semantic_features = [
            feature
            for feature in OPTIONAL_SEMANTIC_FEATURES
            if feature in train_rows.columns and feature in valid_rows.columns
        ]
        for feature in semantic_features:
            x_train[feature] = pd.to_numeric(
                train_rows[feature], errors="raise"
            ).to_numpy(dtype=np.float32)
            x_valid[feature] = pd.to_numeric(
                valid_rows[feature], errors="raise"
            ).to_numpy(dtype=np.float32)
        if not semantic_features:
            print("semantic features requested, but the slate contains none")
    y_train = train_rows["label"].to_numpy(dtype=np.int8)
    y_valid = valid_rows["label"].to_numpy(dtype=np.int8)
    fit_index, inner_valid_index = inner_group_split(
        train_rows,
        args.inner_valid_size,
        args.seed + selected_fold * 100,
    )
    x_fit = x_train.iloc[fit_index]
    y_fit = y_train[fit_index]
    if len(inner_valid_index):
        x_inner_valid: pd.DataFrame | None = x_train.iloc[inner_valid_index]
        y_inner_valid: np.ndarray | None = y_train[inner_valid_index]
    else:
        x_inner_valid = None
        y_inner_valid = None
    print(
        f"fold={selected_fold} model_fit_rows={len(fit_index):,} "
        f"inner_early_stop_rows={len(inner_valid_index):,} outer_score_rows={len(valid_rows):,}"
    )

    score_output = valid_rows[SLATE_COLUMNS].copy().reset_index(drop=True)
    score_output["lexical_score"] = x_valid["lexical_score"].to_numpy(dtype=np.float32)
    for feature in semantic_features:
        score_output[feature] = x_valid[feature].to_numpy(dtype=np.float32)
    models = set(args.models)
    if "catboost" in models:
        _, catboost_probability = train_catboost(
            x_fit,
            y_fit,
            x_inner_valid,
            y_inner_valid,
            x_valid,
            output_dir,
            args,
        )
        score_output["catboost_prob"] = catboost_probability
    if "lgbm" in models:
        _, lgbm_probability = train_lgbm(
            x_fit,
            y_fit,
            x_inner_valid,
            y_inner_valid,
            x_valid,
            output_dir,
            args,
        )
        score_output["lgbm_prob"] = lgbm_probability

    score_path = output_dir / "oof_scores.csv"
    score_output.to_csv(score_path, index=False)
    metadata = {
        "fold": selected_fold,
        "models": sorted(models),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_terms": len(train_term_ids),
        "valid_terms": valid_rows["term_id"].nunique(),
        "model_fit_rows": len(fit_index),
        "inner_early_stop_rows": len(inner_valid_index),
        "semantic_features": semantic_features,
        "train_positive_rate": float(y_train.mean()),
        "valid_positive_rate": float(y_valid.mean()),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"wrote {score_path}")

    del x_train, x_valid, terms, items, item_meta_lookup
    gc.collect()
    return score_path


def combine_fold_scores(
    output_dir: Path, n_splits: int, output: Path | None = None
) -> Path:
    paths = [output_dir / f"fold_{fold}" / "oof_scores.csv" for fold in range(n_splits)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fold score files: {missing}")
    frames: list[pd.DataFrame] = []
    for fold, path in enumerate(paths):
        frame = pd.read_csv(
            path, dtype={"slate_id": str, "term_id": str, "item_id": str}
        )
        transformer_path = output_dir / f"fold_{fold}" / "transformer_oof_scores.csv"
        if transformer_path.exists():
            transformer = pd.read_csv(transformer_path, dtype={"slate_id": str})
            required = {"slate_id", "transformer_prob"}
            missing_columns = required - set(transformer.columns)
            if missing_columns:
                raise ValueError(
                    f"{transformer_path} is missing {sorted(missing_columns)}"
                )
            frame = frame.merge(
                transformer[["slate_id", "transformer_prob"]],
                on="slate_id",
                how="left",
                validate="one_to_one",
            )
            if frame["transformer_prob"].isna().any():
                raise ValueError(
                    f"{transformer_path} does not cover every row in fold {fold}"
                )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["slate_id"].duplicated().any():
        examples = (
            combined.loc[combined["slate_id"].duplicated(), "slate_id"].head().tolist()
        )
        raise ValueError(f"Duplicate OOF slate ids, examples={examples}")
    combined = combined.sort_values("slate_id").reset_index(drop=True)
    output_path = output or output_dir / "oof_scores.csv"
    combined.to_csv(output_path, index=False)
    print(f"wrote {output_path} rows={len(combined):,}")
    return output_path


def run_all(args: argparse.Namespace) -> None:
    for fold in range(args.n_splits):
        run_fold(args, fold=fold)
    combine_fold_scores(Path(args.output_dir), args.n_splits)


def simplex_weights(n_components: int, step: float) -> Iterable[tuple[float, ...]]:
    if n_components <= 0:
        raise ValueError("At least one score component is required")
    units = int(round(1.0 / step))
    if units <= 0 or not np.isclose(units * step, 1.0, atol=1e-8):
        raise ValueError(
            "weight-step must divide 1.0 exactly, for example 0.25, 0.20, or 0.10"
        )
    for separators in itertools.combinations(
        range(units + n_components - 1), n_components - 1
    ):
        boundaries = (-1,) + separators + (units + n_components - 1,)
        counts = tuple(
            boundaries[i + 1] - boundaries[i] - 1 for i in range(n_components)
        )
        yield tuple(count / units for count in counts)


def macro_f1_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive_f1 = 2 * tp / positive_denominator if positive_denominator else 0.0
    negative_f1 = 2 * tn / negative_denominator if negative_denominator else 0.0
    return 0.5 * (positive_f1 + negative_f1)


def evaluate_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    term_ids: np.ndarray,
    rates: list[float],
    constraint_base: int,
) -> list[tuple[float, float, int]]:
    n_rows = len(scores)
    total_positive = int(labels.sum())
    mandatory = np.zeros(n_rows, dtype=bool)
    if constraint_base > 0:
        frame = pd.DataFrame({"term_id": term_ids, "score": scores})
        group_size = frame.groupby("term_id", sort=False)["score"].transform("size")
        minimum = (group_size - constraint_base).clip(lower=0)
        rank = frame.groupby("term_id", sort=False)["score"].rank(
            method="first", ascending=False
        )
        mandatory = rank.le(minimum).to_numpy()

    mandatory_count = int(mandatory.sum())
    mandatory_positive = int(labels[mandatory].sum())
    eligible_positions = np.flatnonzero(~mandatory)
    order = np.argsort(-scores[eligible_positions], kind="stable")
    ordered_labels = labels[eligible_positions[order]].astype(np.int64)
    cumulative_positive = np.cumsum(ordered_labels)

    output: list[tuple[float, float, int]] = []
    for rate in rates:
        predicted_positive = int(round(n_rows * rate))
        if predicted_positive < mandatory_count or predicted_positive > n_rows:
            continue
        additional = predicted_positive - mandatory_count
        additional_tp = int(cumulative_positive[additional - 1]) if additional else 0
        tp = mandatory_positive + additional_tp
        fp = predicted_positive - tp
        fn = total_positive - tp
        tn = n_rows - tp - fp - fn
        output.append((rate, macro_f1_from_counts(tp, fp, fn, tn), mandatory_count))
    return output


def merge_extra_scores(frame: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    merged = frame
    for raw_path in paths:
        path = Path(raw_path)
        extra = pd.read_csv(path, dtype={"slate_id": str})
        if "slate_id" not in extra.columns:
            raise ValueError(f"{path} must contain slate_id")
        value_columns = [column for column in extra.columns if column != "slate_id"]
        duplicates = set(value_columns) & set(merged.columns)
        if duplicates:
            raise ValueError(
                f"{path} duplicates existing columns: {sorted(duplicates)}"
            )
        merged = merged.merge(extra, on="slate_id", how="left", validate="one_to_one")
        if merged[value_columns].isna().any().any():
            raise ValueError(f"{path} does not cover every OOF slate_id")
    return merged


def optimize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.oof_scores, dtype={"slate_id": str, "term_id": str})
    frame = merge_extra_scores(frame, args.extra_scores)
    for column in ["label", "fold"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int16)

    if args.components:
        components = args.components
    else:
        preferred = [
            "catboost_prob",
            "lgbm_prob",
            "transformer_prob",
            "semantic_cosine",
            "semantic_rank_pct",
            "lexical_score",
        ]
        components = [column for column in preferred if column in frame.columns]
    if not components:
        raise ValueError("No score components were found; pass --components explicitly")
    missing = set(components) - set(frame.columns)
    if missing:
        raise ValueError(f"OOF score file is missing components: {sorted(missing)}")

    rank_columns: list[np.ndarray] = []
    for component in components:
        numeric = pd.to_numeric(frame[component], errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"Component {component} contains missing/non-numeric values"
            )
        fold_rank = numeric.groupby(frame["fold"], sort=False).rank(
            method="average", pct=True
        )
        rank_columns.append(fold_rank.to_numpy(dtype=np.float32))

    labels = frame["label"].to_numpy(dtype=np.int8)
    term_ids = frame["term_id"].astype(str).to_numpy()
    rates = sorted(set(float(rate) for rate in args.rates))
    constraints = sorted(set(int(base) for base in args.constraint_bases))
    results: list[dict[str, float | int]] = []

    weight_grid = list(simplex_weights(len(components), args.weight_step))
    print(
        f"optimizing rows={len(frame):,} components={components} "
        f"weight_combinations={len(weight_grid)} rates={len(rates)} constraints={constraints}"
    )
    for weights in tqdm(weight_grid, desc="OOF blend grid"):
        blend = np.zeros(len(frame), dtype=np.float32)
        for values, weight in zip(rank_columns, weights):
            if weight:
                blend += values * np.float32(weight)
        for constraint_base in constraints:
            for rate, score, mandatory_count in evaluate_rates(
                blend,
                labels,
                term_ids,
                rates,
                constraint_base,
            ):
                row: dict[str, float | int] = {
                    "macro_f1": score,
                    "rate": rate,
                    "constraint_base": constraint_base,
                    "mandatory_count": mandatory_count,
                }
                for component, weight in zip(components, weights):
                    row[f"weight_{component}"] = weight
                results.append(row)

    result_frame = (
        pd.DataFrame(results)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    result_path = output_dir / "oof_optimization.csv"
    result_frame.to_csv(result_path, index=False)
    best = result_frame.iloc[0].to_dict()
    report = {
        "rows": len(frame),
        "positive_rate": float(labels.mean()),
        "components": components,
        "weight_step": args.weight_step,
        "best": best,
    }
    report_path = output_dir / "oof_best.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result_frame.head(args.show_top).to_string(index=False))
    print(f"wrote {result_path}")
    print(f"wrote {report_path}")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/grouped_oof")
    parser.add_argument("--slates")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["catboost", "lgbm"],
        default=["catboost", "lgbm"],
    )
    parser.add_argument(
        "--max-train-rows", type=int, default=0, help="Debug only; 0 uses all rows."
    )
    parser.add_argument(
        "--max-valid-rows", type=int, default=0, help="Debug only; 0 uses all rows."
    )
    parser.add_argument("--category-topk", type=int, default=5)
    parser.add_argument("--category-max-examples", type=int, default=0)
    parser.add_argument("--category-max-features", type=int, default=120_000)
    parser.add_argument("--category-max-iter", type=int, default=80)
    parser.add_argument("--category-min-class-count", type=int, default=2)
    parser.add_argument("--category-chunk-size", type=int, default=25_000)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--inner-valid-size",
        type=float,
        default=0.10,
        help=(
            "Fraction of outer-training terms reserved for early stopping; "
            "outer fold labels are never used."
        ),
    )
    parser.add_argument("--catboost-iterations", type=int, default=2_000)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.035)
    parser.add_argument("--catboost-depth", type=int, default=7)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--catboost-random-strength", type=float, default=0.8)
    parser.add_argument("--catboost-subsample", type=float, default=0.85)
    parser.add_argument("--catboost-rsm", type=float, default=0.95)
    parser.add_argument("--lgbm-estimators", type=int, default=4_000)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-num-leaves", type=int, default=128)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=80)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--semantic-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use semantic_cosine/semantic_rank_pct when those optional columns "
            "exist in the slate."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe term-grouped candidate-slate OOF validation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    slate_parser = subparsers.add_parser("build-slates")
    slate_parser.add_argument("--data-dir", default="data")
    slate_parser.add_argument("--output-dir", default="outputs/grouped_oof")
    slate_parser.add_argument("--n-splits", type=int, default=5)
    slate_parser.add_argument("--base-candidates", type=int, default=100)
    slate_parser.add_argument("--chunk-size", type=int, default=128)
    slate_parser.add_argument("--max-features", type=int, default=250_000)
    slate_parser.add_argument("--min-df", type=int, default=2)
    slate_parser.add_argument("--word-ngram-max", type=int, default=2)
    slate_parser.add_argument("--seed", type=int, default=42)
    slate_parser.add_argument("--limit-terms", type=int, default=0, help="Debug only.")
    slate_parser.add_argument("--limit-items", type=int, default=0, help="Debug only.")
    slate_parser.set_defaults(func=build_slates)

    fold_parser = subparsers.add_parser("run-fold")
    add_training_arguments(fold_parser)
    fold_parser.add_argument("--fold", type=int, required=True)
    fold_parser.set_defaults(func=run_fold)

    all_parser = subparsers.add_parser("run-all")
    add_training_arguments(all_parser)
    all_parser.add_argument("--n-splits", type=int, default=5)
    all_parser.set_defaults(func=run_all)

    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--output-dir", default="outputs/grouped_oof")
    combine_parser.add_argument("--n-splits", type=int, default=5)
    combine_parser.add_argument("--output")
    combine_parser.set_defaults(
        func=lambda args: combine_fold_scores(
            Path(args.output_dir),
            args.n_splits,
            Path(args.output) if args.output else None,
        )
    )

    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument(
        "--oof-scores", default="outputs/grouped_oof/oof_scores.csv"
    )
    optimize_parser.add_argument("--output-dir", default="outputs/grouped_oof")
    optimize_parser.add_argument("--extra-scores", nargs="*", default=[])
    optimize_parser.add_argument("--components", nargs="*")
    optimize_parser.add_argument("--weight-step", type=float, default=0.25)
    optimize_parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[round(value, 3) for value in np.arange(0.15, 0.351, 0.005)],
    )
    optimize_parser.add_argument(
        "--constraint-bases", nargs="+", type=int, default=[0, 100]
    )
    optimize_parser.add_argument("--show-top", type=int, default=20)
    optimize_parser.set_defaults(func=optimize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
