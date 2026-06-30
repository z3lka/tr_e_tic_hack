from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lexical_baseline import load_items, load_terms, normalize_text, pair_features


BASE_FEATURE_NAMES = [
    "n_query_tokens",
    "title_cov",
    "category_cov",
    "brand_cov",
    "attr_cov",
    "full_cov",
    "title_ratio",
    "category_ratio",
    "brand_ratio",
    "query_in_title",
    "query_in_category",
    "query_in_brand",
    "full_complete",
    "full_high",
    "title_complete",
    "category_complete",
    "brand_complete",
    "short_complete",
    "gender_mismatch",
    "age_mismatch",
    "color_mismatch",
    "weak_match",
    "title_len",
    "category_len",
    "brand_len",
    "attr_len",
    "lexical_score",
]

CATEGORY_LEVELS = ("cat_l1", "cat_l2")
CATEGORY_TOPK = 5
CATEGORY_FEATURE_NAMES: list[str] = []
for level in CATEGORY_LEVELS:
    CATEGORY_FEATURE_NAMES.extend(
        [
            f"{level}_top1_match",
            f"{level}_top3_match",
            f"{level}_top5_match",
            f"{level}_candidate_prob",
            f"{level}_top1_prob",
            f"{level}_rank",
        ]
    )

ITEM_CAT_FEATURES = ["item_cat_l1", "item_cat_l2", "item_gender", "item_age_group"]
FEATURE_NAMES = BASE_FEATURE_NAMES + CATEGORY_FEATURE_NAMES + ITEM_CAT_FEATURES
CAT_FEATURES = ITEM_CAT_FEATURES


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


def add_category_levels(frame: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    levels = pd.DataFrame(
        frame[category_col].map(split_category).tolist(),
        columns=["cat_l1", "cat_l2", "cat_l3", "cat_leaf"],
        index=frame.index,
    )
    return pd.concat([frame, levels], axis=1)


def build_term_category_lookup(
    topk: pd.DataFrame,
    levels: tuple[str, ...] = CATEGORY_LEVELS,
    k: int = CATEGORY_TOPK,
) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for row in topk.itertuples(index=False):
        row_dict = row._asdict()
        entry: dict[str, object] = {}
        for level in levels:
            scores: dict[str, float] = {}
            for rank in range(1, k + 1):
                label = row_dict.get(f"{level}_top{rank}")
                score = row_dict.get(f"{level}_prob{rank}")
                if isinstance(label, str) and label:
                    scores[label] = float(score)
            entry[level] = scores
            entry[f"{level}_top1_prob"] = next(iter(scores.values()), 0.0)
        lookup[row_dict["term_id"]] = entry
    return lookup


def build_item_meta_lookup(items_path: Path) -> dict[str, dict[str, str]]:
    items = pd.read_csv(items_path, usecols=["item_id", "category", "gender", "age_group"])
    items = add_category_levels(items)
    lookup: dict[str, dict[str, str]] = {}
    for row in tqdm(items.itertuples(index=False), total=len(items), desc="item metadata"):
        lookup[row.item_id] = {
            "cat_l1": row.cat_l1,
            "cat_l2": row.cat_l2,
            "item_cat_l1": row.cat_l1 or "unknown",
            "item_cat_l2": row.cat_l2 or "unknown",
            "item_gender": normalize_text(row.gender) or "unknown",
            "item_age_group": normalize_text(row.age_group) or "unknown",
        }
    return lookup


def category_pair_features(
    term_id: str,
    item_id: str,
    term_category_lookup: dict[str, dict[str, object]],
    item_meta_lookup: dict[str, dict[str, str]],
) -> dict[str, float | str]:
    term_entry = term_category_lookup.get(term_id, {})
    item_entry = item_meta_lookup.get(item_id, {})
    out: dict[str, float | str] = {
        "item_cat_l1": item_entry.get("item_cat_l1", "unknown"),
        "item_cat_l2": item_entry.get("item_cat_l2", "unknown"),
        "item_gender": item_entry.get("item_gender", "unknown"),
        "item_age_group": item_entry.get("item_age_group", "unknown"),
    }

    for level in CATEGORY_LEVELS:
        scores = term_entry.get(level, {})
        if not isinstance(scores, dict):
            scores = {}
        ordered_labels = list(scores.keys())
        candidate = item_entry.get(level, "")
        rank = ordered_labels.index(candidate) + 1 if candidate in scores else 0

        out[f"{level}_top1_match"] = float(rank == 1)
        out[f"{level}_top3_match"] = float(1 <= rank <= 3)
        out[f"{level}_top5_match"] = float(1 <= rank <= 5)
        out[f"{level}_candidate_prob"] = float(scores.get(candidate, 0.0))
        out[f"{level}_top1_prob"] = float(term_entry.get(f"{level}_top1_prob", 0.0))
        out[f"{level}_rank"] = float(rank)

    return out


def rows_to_frame(
    rows: pd.DataFrame,
    terms: dict[str, dict[str, object]],
    items: dict[str, tuple[str, str, str, str, str, str]],
    term_category_lookup: dict[str, dict[str, object]],
    item_meta_lookup: dict[str, dict[str, str]],
    desc: str,
) -> pd.DataFrame:
    numeric_names = BASE_FEATURE_NAMES + CATEGORY_FEATURE_NAMES
    numeric = np.empty((len(rows), len(numeric_names)), dtype=np.float32)
    cat_values = {name: np.empty(len(rows), dtype=object) for name in ITEM_CAT_FEATURES}

    iterator = tqdm(rows.itertuples(index=False), total=len(rows), desc=desc)
    for i, row in enumerate(iterator):
        features: dict[str, float | str] = pair_features(terms[row.term_id], items[row.item_id])
        features.update(category_pair_features(row.term_id, row.item_id, term_category_lookup, item_meta_lookup))
        numeric[i] = [float(features.get(name, 0.0)) for name in numeric_names]
        for name in ITEM_CAT_FEATURES:
            cat_values[name][i] = features.get(name, "unknown") or "unknown"

    frame = pd.DataFrame(numeric, columns=numeric_names)
    for name in ITEM_CAT_FEATURES:
        frame[name] = cat_values[name]
    return frame[FEATURE_NAMES]


def sample_frame(frame: pd.DataFrame, mask: pd.Series, n: int, seed: int) -> pd.DataFrame:
    subset = frame.loc[mask]
    if len(subset) <= n:
        return subset
    return subset.sample(n=n, random_state=seed)


def build_unlabeled_negatives(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    pairs = pd.read_csv(data_dir / "submission_pairs.csv", usecols=["id", "term_id", "item_id"])
    scores = pd.read_csv(args.lexical_scores, usecols=["score"])
    if len(pairs) != len(scores):
        raise ValueError("submission_pairs and lexical_scores row counts differ")

    pairs["score"] = scores["score"].astype("float32").to_numpy()
    parts = [
        sample_frame(pairs, pairs["score"] < args.low_max, args.n_low, args.seed),
        sample_frame(
            pairs,
            (pairs["score"] >= args.low_max) & (pairs["score"] < args.mid_max),
            args.n_mid,
            args.seed + 1,
        ),
        sample_frame(
            pairs,
            (pairs["score"] >= args.mid_max) & (pairs["score"] < args.hard_max),
            args.n_hard,
            args.seed + 2,
        ),
    ]
    negatives = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    print("negative score distribution")
    print(negatives["score"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]).to_string())
    return negatives[["term_id", "item_id"]]


def load_feature_context(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    terms = load_terms(data_dir / "terms.csv")
    items = load_items(data_dir / "items.csv")
    item_meta_lookup = build_item_meta_lookup(data_dir / "items.csv")

    topk_path = Path(args.term_category_topk)
    if not topk_path.exists():
        raise FileNotFoundError(
            f"term category top-k file not found: {topk_path}. "
            "Run the category-aware notebook first or pass --term-category-topk."
        )
    term_category_topk = pd.read_csv(topk_path)
    term_category_lookup = build_term_category_lookup(term_category_topk)
    return terms, items, term_category_lookup, item_meta_lookup


def train(args: argparse.Namespace) -> None:
    start_time = time.time()
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    terms, items, term_category_lookup, item_meta_lookup = load_feature_context(args)
    positives = pd.read_csv(Path(args.data_dir) / "training_pairs.csv", usecols=["term_id", "item_id"])
    negatives = build_unlabeled_negatives(args)

    if args.n_pos and len(positives) > args.n_pos:
        positives = positives.sample(n=args.n_pos, random_state=args.seed)

    x_pos = rows_to_frame(positives, terms, items, term_category_lookup, item_meta_lookup, "positive features")
    x_neg = rows_to_frame(negatives, terms, items, term_category_lookup, item_meta_lookup, "negative features")
    x = pd.concat([x_pos, x_neg], ignore_index=True)
    y = np.concatenate([np.ones(len(x_pos), dtype=np.int8), np.zeros(len(x_neg), dtype=np.int8)])
    print(f"matrix={x.shape} positives={int(y.sum()):,} positive_rate={y.mean():.4f}")

    del x_pos, x_neg
    gc.collect()

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=y,
    )
    del x
    gc.collect()

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bootstrap_type="Bernoulli",
        subsample=args.subsample,
        rsm=args.rsm,
        auto_class_weights="Balanced",
        random_seed=args.seed,
        thread_count=args.thread_count,
        allow_writing_files=False,
        verbose=args.verbose,
    )

    train_pool = Pool(x_train, y_train, cat_features=CAT_FEATURES)
    valid_pool = Pool(x_valid, y_valid, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=False)

    valid_prob = model.predict_proba(valid_pool)[:, 1]
    best = (0.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (valid_prob >= threshold).astype(np.int8)
        score = f1_score(y_valid, pred, average="macro")
        if score > best[0]:
            best = (float(score), float(threshold))
    print(f"synthetic valid macro_f1={best[0]:.5f} threshold={best[1]:.3f}")

    importances = pd.Series(model.get_feature_importance(valid_pool), index=FEATURE_NAMES).sort_values(ascending=False)
    print(importances.head(35).to_string())

    model.save_model(str(model_path))
    joblib.dump(
        {
            "features": FEATURE_NAMES,
            "cat_features": CAT_FEATURES,
            "threshold": best[1],
            "iterations": args.iterations,
            "term_category_topk": str(args.term_category_topk),
        },
        model_path.with_suffix(".meta.joblib"),
    )
    elapsed = time.time() - start_time
    print(f"wrote {model_path}")
    print(f"elapsed_minutes={elapsed / 60:.1f}")


def predict(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = CatBoostClassifier()
    model.load_model(str(args.model))
    terms, items, term_category_lookup, item_meta_lookup = load_feature_context(args)

    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(["id", "term_id", "prob"])

        reader = pd.read_csv(data_dir / "submission_pairs.csv", chunksize=args.chunk_size)
        for chunk in tqdm(reader, desc="predict chunks"):
            matrix = rows_to_frame(chunk, terms, items, term_category_lookup, item_meta_lookup, "chunk features")
            pool = Pool(matrix, cat_features=CAT_FEATURES)
            prob = model.predict_proba(pool)[:, 1]
            writer.writerows(
                (row_id, term_id, f"{score:.8f}")
                for row_id, term_id, score in zip(chunk["id"], chunk["term_id"], prob)
            )
            written += len(chunk)
            del matrix, pool, prob, chunk
            gc.collect()

    print(f"wrote {output_path} rows={written:,}")


def make_submission(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores)
    if args.rate is not None:
        threshold = scores["prob"].quantile(1.0 - args.rate)
    else:
        threshold = args.threshold
    pred = (scores["prob"] >= threshold).astype("int8")
    output = pd.DataFrame({"id": scores["id"], "prediction": pred})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"threshold={threshold:.8f}")
    print(f"wrote {out_path}")
    print(f"rows={len(output):,} positives={int(pred.sum()):,} rate={pred.mean():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data")
    common.add_argument("--term-category-topk", default="outputs/category_aware_notebook/term_category_topk.csv")

    train_cmd = sub.add_parser("train", parents=[common])
    train_cmd.add_argument("--lexical-scores", default="outputs/lexical_scores.csv")
    train_cmd.add_argument("--model", default="outputs/catboost_category_aware.cbm")
    train_cmd.add_argument("--n-pos", type=int, default=0)
    train_cmd.add_argument("--n-low", type=int, default=350_000)
    train_cmd.add_argument("--n-mid", type=int, default=350_000)
    train_cmd.add_argument("--n-hard", type=int, default=150_000)
    train_cmd.add_argument("--low-max", type=float, default=2.0)
    train_cmd.add_argument("--mid-max", type=float, default=4.5)
    train_cmd.add_argument("--hard-max", type=float, default=6.5)
    train_cmd.add_argument("--valid-size", type=float, default=0.15)
    train_cmd.add_argument("--iterations", type=int, default=5_000)
    train_cmd.add_argument("--learning-rate", type=float, default=0.025)
    train_cmd.add_argument("--depth", type=int, default=6)
    train_cmd.add_argument("--l2-leaf-reg", type=float, default=6.0)
    train_cmd.add_argument("--random-strength", type=float, default=0.5)
    train_cmd.add_argument("--subsample", type=float, default=0.85)
    train_cmd.add_argument("--rsm", type=float, default=0.95)
    train_cmd.add_argument("--thread-count", type=int, default=-1)
    train_cmd.add_argument("--verbose", type=int, default=100)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.set_defaults(func=train)

    pred_cmd = sub.add_parser("predict", parents=[common])
    pred_cmd.add_argument("--model", default="outputs/catboost_category_aware.cbm")
    pred_cmd.add_argument("--output", default="outputs/catboost_category_aware_scores.csv")
    pred_cmd.add_argument("--chunk-size", type=int, default=150_000)
    pred_cmd.set_defaults(func=predict)

    submit_cmd = sub.add_parser("submit")
    submit_cmd.add_argument("--scores", default="outputs/catboost_category_aware_scores.csv")
    submit_cmd.add_argument("--output", default="outputs/catboost_category_aware.csv")
    submit_cmd.add_argument("--threshold", type=float, default=0.5)
    submit_cmd.add_argument("--rate", type=float)
    submit_cmd.set_defaults(func=make_submission)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
