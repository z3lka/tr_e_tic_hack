"""Build Turkish-aware lexical relevance features, scores, and submissions."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from text_unidecode import unidecode
from tqdm import tqdm

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "ve",
    "ile",
    "icin",
    "bir",
    "adet",
    "ad",
    "li",
    "set",
    "seti",
    "takim",
    "takimi",
    "urun",
    "model",
    "uyumlu",
    "orjinal",
    "orijinal",
}

COLORS = {
    "beyaz",
    "siyah",
    "kirmizi",
    "mavi",
    "yesil",
    "sari",
    "pembe",
    "mor",
    "gri",
    "lacivert",
    "kahverengi",
    "bej",
    "krem",
    "turuncu",
    "bordo",
    "gold",
    "altin",
    "gumus",
    "haki",
    "taba",
    "lila",
    "ekru",
    "vizon",
    "füme",
}

GENDER_ALLOWED = {
    "kadin": {"kadin", "unisex", "unknown"},
    "bayan": {"kadin", "unisex", "unknown"},
    "erkek": {"erkek", "unisex", "unknown"},
    "unisex": {"kadin", "erkek", "unisex", "unknown"},
}

AGE_ALLOWED = {
    "bebek": {"bebek", "unknown"},
    "cocuk": {"cocuk", "bebek", "unknown"},
    "kiz": {"cocuk", "bebek", "unknown"},
}


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unidecode(str(value).lower())
    text = text.replace("&", " ")
    return re.sub(r"\s+", " ", text).strip()


def stem_token(token: str) -> str:
    if len(token) <= 4:
        return token

    for suffix in (
        "larin",
        "lerin",
        "lari",
        "leri",
        "lara",
        "lere",
        "larda",
        "lerde",
        "ndan",
        "nden",
        "dan",
        "den",
        "nin",
        "nun",
        "ler",
        "lar",
        "lik",
        "luk",
        "cuk",
        "cik",
        "ci",
        "cu",
        "si",
        "su",
        "u",
        "i",
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


@lru_cache(maxsize=1_000_000)
def token_set(text: str) -> frozenset[str]:
    out: set[str] = set()
    for token in TOKEN_RE.findall(text):
        if token in STOPWORDS:
            continue
        out.add(token)
        stemmed = stem_token(token)
        if stemmed not in STOPWORDS:
            out.add(stemmed)
    return frozenset(out)


def query_tokens(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for token in TOKEN_RE.findall(text):
        if token in STOPWORDS:
            continue
        stemmed = stem_token(token)
        for candidate in (token, stemmed):
            if candidate and candidate not in STOPWORDS and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
                break
    return tuple(out)


def coverage(query: tuple[str, ...], field: frozenset[str]) -> float:
    if not query:
        return 0.0
    return sum(1 for token in query if token in field) / len(query)


def color_tokens(query: tuple[str, ...]) -> set[str]:
    return {token for token in query if token in COLORS}


def load_terms(path: Path) -> dict[str, dict[str, object]]:
    terms = pd.read_csv(path)
    records: dict[str, dict[str, object]] = {}
    for row in terms.itertuples(index=False):
        norm_query = normalize_text(row.query)
        q_tokens = query_tokens(norm_query)
        records[row.term_id] = {
            "query": norm_query,
            "tokens": q_tokens,
            "colors": color_tokens(q_tokens),
        }
    return records


def load_items(path: Path) -> dict[str, tuple[str, str, str, str, str, str]]:
    items: dict[str, tuple[str, str, str, str, str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc="items"):
            item_id = row["item_id"]
            items[item_id] = (
                normalize_text(row.get("title")),
                normalize_text(row.get("category")),
                normalize_text(row.get("brand")),
                normalize_text(row.get("gender")) or "unknown",
                normalize_text(row.get("age_group")) or "unknown",
                normalize_text(row.get("attributes")),
            )
    return items


def pair_features(
    term: dict[str, object], item: tuple[str, str, str, str, str, str]
) -> dict[str, float]:
    query = term["query"]  # type: ignore[index]
    q_tokens = term["tokens"]  # type: ignore[index]
    q_colors = term["colors"]  # type: ignore[index]
    title, category, brand, gender, age_group, attrs = item

    title_tokens = token_set(title)
    category_tokens = token_set(category)
    brand_tokens = token_set(brand)
    attr_tokens = token_set(attrs)
    full_tokens = title_tokens | category_tokens | brand_tokens | attr_tokens

    title_cov = coverage(q_tokens, title_tokens)
    category_cov = coverage(q_tokens, category_tokens)
    brand_cov = coverage(q_tokens, brand_tokens)
    attr_cov = coverage(q_tokens, attr_tokens)
    full_cov = coverage(q_tokens, full_tokens)

    title_ratio = fuzz.token_set_ratio(query, title) / 100.0
    category_ratio = fuzz.token_set_ratio(query, category) / 100.0
    brand_ratio = max(
        fuzz.ratio(query, brand) / 100.0,
        fuzz.partial_ratio(query, brand) / 100.0 if brand else 0.0,
    )

    query_in_title = float(bool(query and query in title))
    query_in_category = float(bool(query and query in category))
    query_in_brand = float(bool(query and query in brand))
    full_complete = float(full_cov >= 0.999)
    full_high = float(0.75 <= full_cov < 0.999)
    title_complete = float(title_cov >= 0.999)
    category_complete = float(category_cov >= 0.999)
    brand_complete = float(brand_cov >= 0.999 and bool(brand))
    short_complete = float(len(q_tokens) <= 2 and full_cov >= 0.999)

    gender_mismatch = 0.0
    for gender_word, allowed in GENDER_ALLOWED.items():
        if (
            gender_word in q_tokens
            and gender not in allowed
            and gender_word not in full_tokens
        ):
            gender_mismatch = 1.0
            break

    age_mismatch = 0.0
    for age_word, allowed in AGE_ALLOWED.items():
        if (
            age_word in q_tokens
            and age_group not in allowed
            and age_word not in full_tokens
        ):
            age_mismatch = 1.0
            break

    color_mismatch = float(bool(q_colors and full_tokens.isdisjoint(q_colors)))
    weak_match = float(
        full_cov < 0.34 and max(title_ratio, category_ratio, brand_ratio) < 0.55
    )

    features = {
        "n_query_tokens": float(len(q_tokens)),
        "title_cov": title_cov,
        "category_cov": category_cov,
        "brand_cov": brand_cov,
        "attr_cov": attr_cov,
        "full_cov": full_cov,
        "title_ratio": title_ratio,
        "category_ratio": category_ratio,
        "brand_ratio": brand_ratio,
        "query_in_title": query_in_title,
        "query_in_category": query_in_category,
        "query_in_brand": query_in_brand,
        "full_complete": full_complete,
        "full_high": full_high,
        "title_complete": title_complete,
        "category_complete": category_complete,
        "brand_complete": brand_complete,
        "short_complete": short_complete,
        "gender_mismatch": gender_mismatch,
        "age_mismatch": age_mismatch,
        "color_mismatch": color_mismatch,
        "weak_match": weak_match,
        "title_len": float(len(title_tokens)),
        "category_len": float(len(category_tokens)),
        "brand_len": float(len(brand_tokens)),
        "attr_len": float(len(attr_tokens)),
    }

    features["lexical_score"] = lexical_score_from_features(features)
    return features


def lexical_score_from_features(features: dict[str, float]) -> float:
    title_cov = features["title_cov"]
    category_cov = features["category_cov"]
    brand_cov = features["brand_cov"]
    attr_cov = features["attr_cov"]
    full_cov = features["full_cov"]
    title_ratio = features["title_ratio"]
    category_ratio = features["category_ratio"]
    brand_ratio = features["brand_ratio"]

    score = 0.0
    score += 2.1 * full_cov
    score += 1.4 * title_cov
    score += 1.0 * category_cov
    score += 1.2 * brand_cov
    score += 0.55 * attr_cov
    score += 0.75 * title_ratio
    score += 0.45 * category_ratio
    score += 0.55 * brand_ratio

    if features["query_in_title"]:
        score += 1.0
    if features["query_in_category"]:
        score += 0.8
    if features["query_in_brand"]:
        score += 1.1
    if features["full_complete"]:
        score += 1.2
    elif features["full_high"]:
        score += 0.55
    if features["title_complete"]:
        score += 0.9
    if features["category_complete"]:
        score += 0.7
    if features["brand_complete"]:
        score += 0.9
    if features["short_complete"]:
        score += 0.55

    if features["gender_mismatch"]:
        score -= 0.85
    if features["age_mismatch"]:
        score -= 0.75
    if features["color_mismatch"]:
        score -= 0.7
    if features["weak_match"]:
        score -= 0.9

    return score


def score_pair(
    term: dict[str, object], item: tuple[str, str, str, str, str, str]
) -> float:
    return pair_features(term, item)["lexical_score"]


def score_pairs(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    terms = load_terms(data_dir / "terms.csv")
    items = load_items(data_dir / "items.csv")

    in_path = data_dir / args.pairs
    total = args.limit
    if total is None:
        with in_path.open(newline="", encoding="utf-8") as handle:
            total = sum(1 for _ in handle) - 1

    with in_path.open(newline="", encoding="utf-8") as in_handle, out_path.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.writer(out_handle)
        writer.writerow(["id", "term_id", "score"])
        for i, row in enumerate(tqdm(reader, total=total, desc="pairs")):
            if args.limit is not None and i >= args.limit:
                break
            item = items[row["item_id"]]
            term = terms[row["term_id"]]
            writer.writerow(
                [row["id"], row["term_id"], f"{score_pair(term, item):.6f}"]
            )


def summarize_scores(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores)
    print(
        scores["score"].describe(
            percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )
    for threshold in args.thresholds:
        pred = scores["score"] >= threshold
        print(
            f"threshold={threshold:.3f} positives={int(pred.sum())} rate={pred.mean():.5f}"
        )

    by_term = scores.groupby("term_id")["score"].agg(["count", "max", "mean"])
    print(by_term.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))


def make_submission(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores)
    pred = scores["score"] >= args.threshold

    if args.min_per_term > 0 or args.max_per_term > 0:
        scores = scores.assign(pred=pred.astype("int8"))
        parts = []
        for _, group in tqdm(scores.groupby("term_id", sort=False), desc="terms"):
            group = group.copy()
            order = group["score"].sort_values(ascending=False).index
            if (
                args.min_per_term > 0
                and group.loc[order[: args.min_per_term], "score"].max()
                >= args.min_score
            ):
                group.loc[order[: args.min_per_term], "pred"] = 1
            if args.max_per_term > 0:
                positives = group.index[group["pred"] == 1]
                if len(positives) > args.max_per_term:
                    keep = (
                        group.loc[positives]
                        .sort_values("score", ascending=False)
                        .head(args.max_per_term)
                        .index
                    )
                    group.loc[positives, "pred"] = 0
                    group.loc[keep, "pred"] = 1
            parts.append(group[["id", "pred"]])
        scored = pd.concat(parts, ignore_index=True)
        pred_values = scored["pred"].astype("int8")
        ids = scored["id"]
    else:
        pred_values = pred.astype("int8")
        ids = scores["id"]

    output = pd.DataFrame({"id": ids, "prediction": pred_values})
    output.to_csv(args.output, index=False)
    counts = Counter(output["prediction"])
    print(f"wrote {args.output}")
    print(
        f"rows={len(output)} positives={counts[1]} rate={counts[1] / len(output):.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    score = sub.add_parser("score")
    score.add_argument("--data-dir", default="data")
    score.add_argument("--pairs", default="submission_pairs.csv")
    score.add_argument("--output", default="outputs/lexical_scores.csv")
    score.add_argument("--limit", type=int)
    score.set_defaults(func=score_pairs)

    summary = sub.add_parser("summary")
    summary.add_argument("--scores", default="outputs/lexical_scores.csv")
    summary.add_argument(
        "--thresholds", type=float, nargs="+", default=[3.0, 3.5, 4.0, 4.5, 5.0]
    )
    summary.set_defaults(func=summarize_scores)

    submit = sub.add_parser("submit")
    submit.add_argument("--scores", default="outputs/lexical_scores.csv")
    submit.add_argument("--output", default="outputs/submission_lexical.csv")
    submit.add_argument("--threshold", type=float, required=True)
    submit.add_argument("--min-per-term", type=int, default=0)
    submit.add_argument("--max-per-term", type=int, default=0)
    submit.add_argument("--min-score", type=float, default=0.0)
    submit.set_defaults(func=make_submission)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
