"""Mine deterministic TF-IDF negatives from training queries and catalog items."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from lexical_baseline import normalize_text


def item_search_text(items: pd.DataFrame) -> pd.Series:
    title = items["title"].map(normalize_text)
    category = items["category"].map(normalize_text)
    brand = items["brand"].map(normalize_text)
    attrs = items["attributes"].map(normalize_text)
    return (
        title + " " + title + " " + brand + " " + brand + " " + category + " " + attrs
    ).str.strip()


def pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["term_id"].astype(str) + "\t" + frame["item_id"].astype(str)


def topk_sparse_row(row: sparse.spmatrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    row = row.tocsr()
    indices = row.indices
    data = row.data
    if len(data) == 0:
        return indices, data

    if len(data) > k:
        selected = np.argpartition(data, -k)[-k:]
        selected = selected[np.argsort(-data[selected], kind="mergesort")]
    else:
        selected = np.argsort(-data, kind="mergesort")
    return indices[selected], data[selected]


def sample_easy_items(
    item_count: int,
    excluded: set[int],
    n_items: int,
    rng: np.random.Generator,
) -> list[int]:
    if n_items <= 0:
        return []

    selected: list[int] = []
    attempts = 0
    max_attempts = max(100, n_items * 50)
    while len(selected) < n_items and attempts < max_attempts:
        item_pos = int(rng.integers(0, item_count))
        attempts += 1
        if item_pos in excluded:
            continue
        excluded.add(item_pos)
        selected.append(item_pos)
    return selected


def load_positive_pairs(path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "label" in pairs.columns:
        pairs = pairs.loc[pairs["label"].astype(str).ne("0")]
    pairs = pairs[["term_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    if pairs.empty:
        raise ValueError(f"No positive training pairs found in {path}")
    return pairs


def mine_negatives(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    terms = pd.read_csv(data_dir / "terms.csv", dtype=str, keep_default_na=False)
    items = pd.read_csv(
        data_dir / "items.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["item_id", "title", "category", "brand", "attributes"],
    )
    positives = load_positive_pairs(data_dir / "training_pairs.csv")

    train_term_ids = positives["term_id"].drop_duplicates().to_numpy()
    if args.limit_terms:
        train_term_ids = train_term_ids[: args.limit_terms]
        positives = positives.loc[
            positives["term_id"].isin(train_term_ids)
        ].reset_index(drop=True)
    if args.limit_items:
        items = items.head(args.limit_items).reset_index(drop=True)

    terms = terms.loc[
        terms["term_id"].isin(train_term_ids), ["term_id", "query"]
    ].copy()
    missing_terms = set(train_term_ids) - set(terms["term_id"])
    if missing_terms:
        examples = sorted(missing_terms)[:5]
        raise KeyError(
            f"{len(missing_terms)} training term_ids are missing from terms.csv, "
            f"examples={examples}"
        )

    terms["query_norm"] = terms["query"].map(normalize_text)
    terms = terms.set_index("term_id").loc[train_term_ids].reset_index()
    item_ids = items["item_id"].astype(str).to_numpy()
    item_to_pos = {item_id: i for i, item_id in enumerate(item_ids)}

    positive_by_term: dict[str, set[int]] = {}
    for term_id, group in positives.groupby("term_id", sort=False):
        positions = {
            item_to_pos[item_id]
            for item_id in group["item_id"].astype(str)
            if item_id in item_to_pos
        }
        positive_by_term[str(term_id)] = positions

    item_text = item_search_text(items)
    corpus = pd.concat([terms["query_norm"], item_text], ignore_index=True)
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
    term_matrix = vectorizer.transform(terms["query_norm"])
    item_matrix = vectorizer.transform(item_text)
    item_matrix_t = item_matrix.T.tocsr()
    print(
        f"terms={len(terms):,} items={len(items):,} positives={len(positives):,} "
        f"vocab={len(vectorizer.vocabulary_):,}"
    )

    rng = np.random.default_rng(args.seed)
    counts = {"hard": 0, "mid": 0, "easy": 0}
    total = 0
    positive_keys = set(pair_key(positives))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["term_id", "item_id", "negative_band", "tfidf_score"])

        for start in tqdm(
            range(0, len(terms), args.chunk_size), desc="mine train-term negatives"
        ):
            stop = min(start + args.chunk_size, len(terms))
            scores = term_matrix[start:stop] @ item_matrix_t
            scores = scores.tocsr()

            for row_offset, term_id in enumerate(
                terms["term_id"].iloc[start:stop].astype(str)
            ):
                pos_items = set(positive_by_term.get(term_id, set()))
                excluded = set(pos_items)
                row_indices, row_scores = topk_sparse_row(
                    scores.getrow(row_offset), args.retrieve_topk
                )

                candidates: list[tuple[int, float]] = []
                for item_pos, score in zip(row_indices, row_scores):
                    item_pos = int(item_pos)
                    if item_pos in excluded:
                        continue
                    excluded.add(item_pos)
                    candidates.append((item_pos, float(score)))

                hard = candidates[: args.hard_per_term]
                mid_pool = candidates[args.hard_per_term :]
                if len(mid_pool) > args.mid_per_term:
                    chosen = rng.choice(
                        len(mid_pool), size=args.mid_per_term, replace=False
                    )
                    mid = [mid_pool[int(i)] for i in chosen]
                else:
                    mid = mid_pool

                rows: list[tuple[str, str, str, float]] = []
                for item_pos, score in hard:
                    rows.append((term_id, item_ids[item_pos], "hard", score))
                for item_pos, score in mid:
                    rows.append((term_id, item_ids[item_pos], "mid", score))

                easy_positions = sample_easy_items(
                    len(item_ids), excluded, args.easy_per_term, rng
                )
                for item_pos in easy_positions:
                    rows.append((term_id, item_ids[item_pos], "easy", 0.0))

                for out_term_id, item_id, band, score in rows:
                    if f"{out_term_id}\t{item_id}" in positive_keys:
                        continue
                    writer.writerow([out_term_id, item_id, band, f"{score:.8f}"])
                    counts[band] += 1
                    total += 1

    print(f"wrote {output_path}")
    print(
        f"negatives={total:,} hard={counts['hard']:,} "
        f"mid={counts['mid']:,} easy={counts['easy']:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/train_term_negatives.csv")
    parser.add_argument("--hard-per-term", type=int, default=25)
    parser.add_argument("--mid-per-term", type=int, default=15)
    parser.add_argument("--easy-per-term", type=int, default=10)
    parser.add_argument("--retrieve-topk", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=250_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-terms",
        type=int,
        help="Debug only: mine negatives for the first N train terms.",
    )
    parser.add_argument(
        "--limit-items", type=int, help="Debug only: restrict the item catalog."
    )
    args = parser.parse_args()
    mine_negatives(args)


if __name__ == "__main__":
    main()
