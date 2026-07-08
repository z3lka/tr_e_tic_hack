from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]


def pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["term_id"].astype(str) + "\t" + frame["item_id"].astype(str)


def assign_score_buckets(scores: pd.Series, n_bins: int) -> pd.Series:
    if len(scores) == 0 or n_bins <= 1 or scores.nunique(dropna=False) <= 1:
        return pd.Series(["all"] * len(scores), index=scores.index)

    bins = min(n_bins, len(scores), int(scores.nunique(dropna=False)))
    codes = pd.qcut(scores.rank(method="first"), q=bins, labels=False, duplicates="drop")
    return codes.fillna(0).astype(int).map(lambda value: f"q{value + 1}")


def stratified_sample(frame: pd.DataFrame, n_per_band: int, score_bins: int, seed: int) -> pd.DataFrame:
    sampled: list[pd.DataFrame] = []
    sample_seed = seed

    for band, band_frame in frame.groupby("negative_band", sort=True):
        band_frame = band_frame.copy()
        band_frame["score_bucket"] = assign_score_buckets(band_frame["tfidf_score"], score_bins)
        buckets = list(band_frame["score_bucket"].drop_duplicates())
        bucket_target = max(1, math.ceil(n_per_band / max(1, len(buckets))))
        band_parts: list[pd.DataFrame] = []

        for _, bucket_frame in band_frame.groupby("score_bucket", sort=True):
            n_rows = min(len(bucket_frame), bucket_target)
            band_parts.append(bucket_frame.sample(n=n_rows, random_state=sample_seed))
            sample_seed += 1

        band_sample = pd.concat(band_parts, ignore_index=True)
        if len(band_sample) > n_per_band:
            band_sample = band_sample.sample(n=n_per_band, random_state=sample_seed)
            sample_seed += 1
        sampled.append(band_sample)

    if not sampled:
        return frame.head(0).copy()
    return pd.concat(sampled, ignore_index=True)


def load_items_for_ids(data_dir: Path, item_ids: set[str], chunksize: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        data_dir / "items.csv",
        usecols=ITEM_COLUMNS,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        matched = chunk.loc[chunk["item_id"].isin(item_ids)]
        if not matched.empty:
            parts.append(matched)

    if not parts:
        raise ValueError("No sampled item_ids were found in items.csv")
    return pd.concat(parts, ignore_index=True).drop_duplicates("item_id")


def audit(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    negatives = pd.read_csv(args.negatives, dtype={"term_id": str, "item_id": str}, keep_default_na=False)
    required = {"term_id", "item_id"}
    missing = required - set(negatives.columns)
    if missing:
        raise ValueError(f"Negative file is missing columns: {sorted(missing)}")

    if "negative_band" not in negatives.columns:
        negatives["negative_band"] = "unknown"
    if "tfidf_score" not in negatives.columns:
        negatives["tfidf_score"] = 0.0
    negatives["tfidf_score"] = pd.to_numeric(negatives["tfidf_score"], errors="coerce").fillna(0.0)
    negatives = negatives.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)

    positives = pd.read_csv(data_dir / "training_pairs.csv", usecols=["term_id", "item_id"], dtype=str)
    overlap_mask = pair_key(negatives).isin(set(pair_key(positives)))
    overlap = int(overlap_mask.sum())
    if overlap and not args.allow_overlap:
        examples = negatives.loc[overlap_mask, ["term_id", "item_id"]].head(10).to_dict("records")
        raise ValueError(f"Found {overlap:,} negatives overlapping positives; examples={examples}")

    sampled = stratified_sample(negatives, args.n_per_band, args.score_bins, args.seed)
    terms = pd.read_csv(data_dir / "terms.csv", dtype=str, keep_default_na=False)
    terms = terms.loc[terms["term_id"].isin(set(sampled["term_id"])), ["term_id", "query"]]
    items = load_items_for_ids(data_dir, set(sampled["item_id"].astype(str)), args.chunksize)

    out = sampled.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
    out = out.sort_values(["negative_band", "score_bucket", "tfidf_score"], ascending=[True, True, False])
    out.to_csv(output_path, index=False)

    if args.jsonl_output:
        jsonl_path = Path(args.jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in out.to_dict("records"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"loaded_negatives={len(negatives):,}")
    print(f"positive_overlap={overlap:,}")
    print(f"wrote {output_path} rows={len(out):,}")
    print(out["negative_band"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample mined negatives for LLM/manual audit.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--negatives", default="outputs/train_term_negatives.csv")
    parser.add_argument("--output", default="outputs/negative_audit_sample.csv")
    parser.add_argument("--jsonl-output")
    parser.add_argument("--n-per-band", type=int, default=200)
    parser.add_argument("--score-bins", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-overlap", action="store_true")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
