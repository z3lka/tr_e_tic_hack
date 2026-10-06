"""Train and score a lexical positive-unlabeled LightGBM model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lexical_baseline import load_items, load_terms, pair_features

FEATURE_NAMES = [
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


def rows_to_matrix(
    rows: pd.DataFrame, terms: dict, items: dict, desc: str
) -> np.ndarray:
    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(
        tqdm(rows.itertuples(index=False), total=len(rows), desc=desc)
    ):
        features = pair_features(terms[row.term_id], items[row.item_id])
        matrix[i] = [features[name] for name in FEATURE_NAMES]
    return matrix


def sample_frame(
    frame: pd.DataFrame, mask: pd.Series, n: int, seed: int
) -> pd.DataFrame:
    subset = frame.loc[mask]
    if len(subset) <= n:
        return subset
    return subset.sample(n=n, random_state=seed)


def pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["term_id"].astype(str) + "\t" + frame["item_id"].astype(str)


def load_train_term_negatives(path: Path, positives: pd.DataFrame) -> pd.DataFrame:
    negatives = pd.read_csv(
        path, usecols=["term_id", "item_id"], dtype=str, keep_default_na=False
    )
    negatives = negatives.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)
    positive_keys = set(pair_key(positives))
    keep = ~pair_key(negatives).isin(positive_keys)
    dropped = int((~keep).sum())
    negatives = negatives.loc[keep].reset_index(drop=True)
    if negatives.empty:
        raise ValueError(f"No usable negatives found in {path}")
    if dropped:
        print(f"dropped {dropped:,} negatives that overlap known positives")
    print(f"using train-term negatives from {path} rows={len(negatives):,}")
    return negatives


def build_unlabeled_negatives(args: argparse.Namespace) -> pd.DataFrame:
    negative_path = Path(args.negatives) if args.negatives else None
    if negative_path and negative_path.exists():
        positives = pd.read_csv(
            Path(args.data_dir) / "training_pairs.csv",
            usecols=["term_id", "item_id"],
            dtype=str,
        )
        return load_train_term_negatives(negative_path, positives)

    if not args.allow_submission_negatives:
        raise FileNotFoundError(
            f"Train-term negatives were not found at {negative_path}. "
            "Create train-term negatives first, or pass "
            "`--allow-submission-negatives` to use the older submission-pair fallback."
        )

    data_dir = Path(args.data_dir)
    pairs = pd.read_csv(data_dir / "submission_pairs.csv")
    scores = pd.read_csv(args.lexical_scores, usecols=["score"])
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
    negatives = pd.concat(parts, ignore_index=True)
    negatives = negatives.drop_duplicates("id")
    print("negative score distribution")
    print(negatives["score"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]).to_string())
    return negatives[["term_id", "item_id"]]


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    terms = load_terms(data_dir / "terms.csv")
    items = load_items(data_dir / "items.csv")

    positives = pd.read_csv(
        data_dir / "training_pairs.csv", usecols=["term_id", "item_id"]
    )
    negatives = build_unlabeled_negatives(args)

    if args.n_pos and len(positives) > args.n_pos:
        positives = positives.sample(n=args.n_pos, random_state=args.seed)

    x_pos = rows_to_matrix(positives, terms, items, "pos")
    x_neg = rows_to_matrix(negatives, terms, items, "neg")
    x = np.vstack([x_pos, x_neg])
    y = np.concatenate(
        [np.ones(len(x_pos), dtype=np.int8), np.zeros(len(x_neg), dtype=np.int8)]
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=args.valid_size, random_state=args.seed, stratify=y
    )

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.log_evaluation(period=50)],
    )

    valid_prob = clf.predict_proba(x_valid)[:, 1]
    best = (0.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (valid_prob >= threshold).astype(np.int8)
        score = f1_score(y_valid, pred, average="macro")
        if score > best[0]:
            best = (score, float(threshold))
    print(f"synthetic valid macro_f1={best[0]:.5f} threshold={best[1]:.3f}")

    importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(
        ascending=False
    )
    print(importances.head(20).to_string())
    joblib.dump(
        {"model": clf, "features": FEATURE_NAMES, "threshold": best[1]}, model_path
    )
    print(f"wrote {model_path}")


def predict(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    model_bundle = joblib.load(args.model)
    clf = model_bundle["model"]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    terms = load_terms(data_dir / "terms.csv")
    items = load_items(data_dir / "items.csv")

    with (data_dir / "submission_pairs.csv").open(
        newline="", encoding="utf-8"
    ) as in_handle, out_path.open("w", newline="", encoding="utf-8") as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.writer(out_handle)
        writer.writerow(["id", "term_id", "prob"])

        ids: list[str] = []
        term_ids: list[str] = []
        rows: list[tuple[str, str]] = []
        for row in tqdm(reader, desc="read/predict"):
            ids.append(row["id"])
            term_ids.append(row["term_id"])
            rows.append((row["term_id"], row["item_id"]))
            if len(rows) >= args.chunk_size:
                write_predictions(writer, clf, ids, term_ids, rows, terms, items)
                ids, term_ids, rows = [], [], []
        if rows:
            write_predictions(writer, clf, ids, term_ids, rows, terms, items)
    print(f"wrote {out_path}")


def write_predictions(writer, clf, ids, term_ids, rows, terms, items) -> None:
    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (term_id, item_id) in enumerate(rows):
        features = pair_features(terms[term_id], items[item_id])
        matrix[i] = [features[name] for name in FEATURE_NAMES]
    probs = clf.predict_proba(matrix)[:, 1]
    writer.writerows(
        (row_id, term_id, f"{prob:.8f}")
        for row_id, term_id, prob in zip(ids, term_ids, probs)
    )


def submit_file(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores)
    if args.rate is not None:
        threshold = scores["prob"].quantile(1.0 - args.rate)
    else:
        threshold = args.threshold
    pred = (scores["prob"] >= threshold).astype("int8")
    output = pd.DataFrame({"id": scores["id"], "prediction": pred})
    output.to_csv(args.output, index=False)
    print(f"threshold={threshold:.8f}")
    print(f"wrote {args.output}")
    print(f"rows={len(output)} positives={int(pred.sum())} rate={pred.mean():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--data-dir", default="data")
    train_cmd.add_argument("--lexical-scores", default="outputs/lexical_scores.csv")
    train_cmd.add_argument("--negatives", default="outputs/train_term_negatives.csv")
    train_cmd.add_argument("--allow-submission-negatives", action="store_true")
    train_cmd.add_argument("--model", default="outputs/pu_lgbm.joblib")
    train_cmd.add_argument("--n-pos", type=int, default=0)
    train_cmd.add_argument("--n-low", type=int, default=350_000)
    train_cmd.add_argument("--n-mid", type=int, default=350_000)
    train_cmd.add_argument("--n-hard", type=int, default=150_000)
    train_cmd.add_argument("--low-max", type=float, default=2.0)
    train_cmd.add_argument("--mid-max", type=float, default=4.5)
    train_cmd.add_argument("--hard-max", type=float, default=6.5)
    train_cmd.add_argument("--valid-size", type=float, default=0.15)
    train_cmd.add_argument("--n-estimators", type=int, default=700)
    train_cmd.add_argument("--learning-rate", type=float, default=0.035)
    train_cmd.add_argument("--num-leaves", type=int, default=64)
    train_cmd.add_argument("--min-child-samples", type=int, default=80)
    train_cmd.add_argument("--subsample", type=float, default=0.9)
    train_cmd.add_argument("--colsample-bytree", type=float, default=0.9)
    train_cmd.add_argument("--reg-lambda", type=float, default=2.0)
    train_cmd.add_argument("--n-jobs", type=int, default=-1)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.set_defaults(func=train)

    pred_cmd = sub.add_parser("predict")
    pred_cmd.add_argument("--data-dir", default="data")
    pred_cmd.add_argument("--model", default="outputs/pu_lgbm.joblib")
    pred_cmd.add_argument("--output", default="outputs/pu_lgbm_scores.csv")
    pred_cmd.add_argument("--chunk-size", type=int, default=100_000)
    pred_cmd.set_defaults(func=predict)

    submit_cmd = sub.add_parser("submit")
    submit_cmd.add_argument("--scores", default="outputs/pu_lgbm_scores.csv")
    submit_cmd.add_argument("--output", default="outputs/submission_pu_lgbm.csv")
    submit_cmd.add_argument("--threshold", type=float, default=0.5)
    submit_cmd.add_argument("--rate", type=float)
    submit_cmd.set_defaults(func=submit_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
