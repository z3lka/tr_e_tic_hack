from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = ("prob", "score")


def score_col(frame: pd.DataFrame, path: str | Path) -> str:
    for column in SCORE_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(f"{path} must contain one of {SCORE_COLUMNS}")


def load_score(path: str | Path, name: str, nrows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str, "term_id": str}, keep_default_na=False, nrows=nrows)
    required = {"id", "term_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    column = score_col(frame, path)
    out = frame[["id", "term_id", column]].copy()
    out = out.rename(columns={column: name})
    out[name] = pd.to_numeric(out[name], errors="coerce").astype("float32")

    if out["id"].duplicated().any():
        examples = out.loc[out["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"{path} contains duplicate ids, examples={examples}")
    if out[name].isna().any():
        raise ValueError(f"{path} contains null/non-numeric scores")
    return out


def rank_normalize(frame: pd.DataFrame, column: str, scope: str) -> pd.Series:
    if scope == "global":
        return frame[column].rank(method="average", pct=True).astype("float32")
    if scope == "term":
        return frame.groupby("term_id", sort=False)[column].rank(method="average", pct=True).astype("float32")
    raise ValueError(f"Unknown rank scope: {scope}")


def align_scores(catboost: pd.DataFrame, lgbm: pd.DataFrame, transformer: pd.DataFrame) -> pd.DataFrame:
    merged = catboost.merge(lgbm[["id", "term_id", "lgbm"]], on="id", how="inner", suffixes=("", "_lgbm"))
    if "term_id_lgbm" in merged.columns:
        mismatch = merged["term_id"].ne(merged["term_id_lgbm"])
        if mismatch.any():
            raise ValueError(f"CatBoost/LGBM term_id mismatch rows={int(mismatch.sum())}")
        merged = merged.drop(columns=["term_id_lgbm"])

    merged = merged.merge(
        transformer[["id", "term_id", "transformer"]],
        on="id",
        how="inner",
        suffixes=("", "_transformer"),
    )
    if "term_id_transformer" in merged.columns:
        mismatch = merged["term_id"].ne(merged["term_id_transformer"])
        if mismatch.any():
            raise ValueError(f"GBDT/transformer term_id mismatch rows={int(mismatch.sum())}")
        merged = merged.drop(columns=["term_id_transformer"])

    expected = len(catboost)
    if len(merged) != expected:
        raise ValueError(f"Score files do not align on id: expected={expected:,} merged={len(merged):,}")
    return merged


def normalized_weighted_sum(columns: list[pd.Series], weights: list[float]) -> pd.Series:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive value")
    output = np.zeros(len(columns[0]), dtype=np.float32)
    for column, weight in zip(columns, weights):
        output += column.to_numpy(dtype=np.float32) * np.float32(weight / total)
    return pd.Series(output, index=columns[0].index, dtype="float32")


def report_correlations(frame: pd.DataFrame) -> None:
    rank_cols = ["catboost_rank", "lgbm_rank", "transformer_rank", "gbdt_rank", "prob"]
    corr = frame[rank_cols].corr(method="pearson")
    print("rank correlation matrix")
    print(corr.to_string(float_format=lambda value: f"{value:.4f}"))


def blend(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrows = args.limit_rows or None

    catboost = load_score(args.catboost, "catboost", nrows=nrows)
    lgbm = load_score(args.lgbm, "lgbm", nrows=nrows)
    transformer = load_score(args.transformer, "transformer", nrows=nrows)
    merged = align_scores(catboost, lgbm, transformer)

    merged["catboost_rank"] = rank_normalize(merged, "catboost", args.rank_scope)
    merged["lgbm_rank"] = rank_normalize(merged, "lgbm", args.rank_scope)
    merged["transformer_rank"] = rank_normalize(merged, "transformer", args.rank_scope)
    merged["gbdt_score"] = normalized_weighted_sum(
        [merged["catboost_rank"], merged["lgbm_rank"]],
        [args.catboost_weight, args.lgbm_weight],
    )
    merged["gbdt_rank"] = rank_normalize(merged, "gbdt_score", args.rank_scope)
    merged["prob"] = normalized_weighted_sum(
        [merged["gbdt_rank"], merged["transformer_rank"]],
        [args.gbdt_weight, args.transformer_weight],
    )

    columns = ["id", "term_id", "prob"]
    if args.include_components:
        columns.extend(["gbdt_score", "catboost_rank", "lgbm_rank", "transformer_rank"])
    merged[columns].to_csv(output_path, index=False)

    print(f"wrote {output_path} rows={len(merged):,}")
    print(
        "weights "
        f"gbdt=({args.catboost_weight:.3f} catboost, {args.lgbm_weight:.3f} lgbm) "
        f"final=({args.gbdt_weight:.3f} gbdt, {args.transformer_weight:.3f} transformer)"
    )
    if args.report_correlations:
        report_correlations(merged)


def rate_suffix(rate: float) -> str:
    return f"r{int(round(rate * 100)):02d}"


def make_prediction(scores: pd.Series, rate: float) -> np.ndarray:
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be between 0 and 1, got {rate}")
    n_pos = int(round(len(scores) * rate))
    prediction = np.zeros(len(scores), dtype=np.int8)
    if n_pos <= 0:
        return prediction
    selected = scores.nlargest(n_pos).index
    prediction[selected] = 1
    return prediction


def submit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores, dtype={"id": str}, keep_default_na=False, nrows=args.limit_rows or None)
    scores = scores.reset_index(drop=True)
    column = args.score_col or score_col(scores, args.scores)
    scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if scores[column].isna().any():
        raise ValueError(f"{args.scores} contains null/non-numeric values in {column}")

    for rate in args.rates:
        prediction = make_prediction(scores[column], rate)
        output = pd.DataFrame({"id": scores["id"], "prediction": prediction})
        output_path = output_dir / f"{args.prefix}_{rate_suffix(rate)}.csv"
        output.to_csv(output_path, index=False)
        print(f"{output_path}: rows={len(output):,} positives={int(prediction.sum()):,} rate={prediction.mean():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank-normalized two-level GBDT + transformer blending.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    blend_cmd = sub.add_parser("blend")
    blend_cmd.add_argument("--catboost", default="outputs/catboost_category_aware_2026-06-30_scores.csv")
    blend_cmd.add_argument("--lgbm", default="outputs/pu_lgbm_v1_scores.csv")
    blend_cmd.add_argument("--transformer", default="outputs/transformer_biencoder_scores.csv")
    blend_cmd.add_argument("--output", default="outputs/moe_gbdt_rankblend_scores.csv")
    blend_cmd.add_argument("--catboost-weight", type=float, default=0.70)
    blend_cmd.add_argument("--lgbm-weight", type=float, default=0.30)
    blend_cmd.add_argument("--gbdt-weight", type=float, default=0.50)
    blend_cmd.add_argument("--transformer-weight", type=float, default=0.50)
    blend_cmd.add_argument("--rank-scope", choices=["global", "term"], default="global")
    blend_cmd.add_argument("--include-components", action="store_true")
    blend_cmd.add_argument("--report-correlations", action="store_true")
    blend_cmd.add_argument("--limit-rows", type=int, default=0)
    blend_cmd.set_defaults(func=blend)

    submit_cmd = sub.add_parser("submit")
    submit_cmd.add_argument("--scores", default="outputs/moe_gbdt_rankblend_scores.csv")
    submit_cmd.add_argument("--output-dir", default="outputs")
    submit_cmd.add_argument("--prefix", default="submission_moe_gbdt_rankblend")
    submit_cmd.add_argument("--rates", type=float, nargs="+", default=[0.18, 0.20, 0.22, 0.24])
    submit_cmd.add_argument("--score-col")
    submit_cmd.add_argument("--limit-rows", type=int, default=0)
    submit_cmd.set_defaults(func=submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
