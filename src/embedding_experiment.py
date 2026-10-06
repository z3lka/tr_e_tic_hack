"""Compare embedding recipes and turn the selected score blend into a submission."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_COMPONENTS = ["catboost_prob", "lgbm_prob", "lexical_score"]


def macro_f1(labels: np.ndarray, prediction: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    prediction = prediction.astype(np.int8)
    tp = int(((labels == 1) & (prediction == 1)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    tn = int(((labels == 0) & (prediction == 0)).sum())
    pos_denominator = 2 * tp + fp + fn
    neg_denominator = 2 * tn + fp + fn
    pos_f1 = 2 * tp / pos_denominator if pos_denominator else 0.0
    neg_f1 = 2 * tn / neg_denominator if neg_denominator else 0.0
    return 0.5 * (pos_f1 + neg_f1)


def simplex_weights(n_components: int, step: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    if n_components <= 0 or units <= 0 or not np.isclose(units * step, 1.0):
        raise ValueError("weight-step must divide one exactly")
    output: list[tuple[float, ...]] = []
    for separators in itertools.combinations(
        range(units + n_components - 1), n_components - 1
    ):
        boundaries = (-1,) + separators + (units + n_components - 1,)
        counts = tuple(
            boundaries[index + 1] - boundaries[index] - 1
            for index in range(n_components)
        )
        output.append(tuple(count / units for count in counts))
    return output


def rank_by_fold(values: pd.Series, folds: pd.Series) -> np.ndarray:
    return (
        values.groupby(folds, sort=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def best_rate(
    scores: np.ndarray, labels: np.ndarray, rates: list[float]
) -> tuple[float, float]:
    order = np.argsort(-scores, kind="stable")
    best = (-1.0, 0.0)
    for rate in rates:
        n_positive = min(len(scores), max(0, int(round(len(scores) * rate))))
        prediction = np.zeros(len(scores), dtype=np.int8)
        prediction[order[:n_positive]] = 1
        value = macro_f1(labels, prediction)
        if value > best[0]:
            best = (value, rate)
    return best


def load_score_file(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"slate_id": str, "id": str}, keep_default_na=False)
    key = "slate_id" if "slate_id" in frame else "id" if "id" in frame else ""
    if not key:
        raise ValueError(f"{path} must contain slate_id or id")
    if "semantic_cosine" not in frame:
        raise ValueError(f"{path} must contain semantic_cosine")
    output = frame[[key, "semantic_cosine"]].rename(
        columns={key: "slate_id", "semantic_cosine": value_name}
    )
    if output["slate_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    output[value_name] = pd.to_numeric(output[value_name], errors="raise").astype(
        np.float32
    )
    return output


def optimize_recipe(
    frame: pd.DataFrame,
    components: list[str],
    rates: list[float],
    weight_step: float,
    required_positive: set[str] | None = None,
) -> dict[str, object]:
    labels = frame["label"].to_numpy(dtype=np.int8)
    folds = frame["fold"]
    ranks = [
        rank_by_fold(pd.to_numeric(frame[column], errors="raise"), folds)
        for column in components
    ]
    best: dict[str, object] | None = None
    for weights in simplex_weights(len(components), weight_step):
        if required_positive and any(
            weights[components.index(component)] <= 0.0
            for component in required_positive
        ):
            continue
        blend = np.zeros(len(frame), dtype=np.float32)
        for values, weight in zip(ranks, weights):
            if weight:
                blend += values * np.float32(weight)
        score, rate = best_rate(blend, labels, rates)
        row: dict[str, object] = {
            "macro_f1": score,
            "positive_rate": rate,
            "weights": {
                component: weight for component, weight in zip(components, weights)
            },
        }
        if best is None or float(row["macro_f1"]) > float(best["macro_f1"]):
            best = row
    if best is None:
        raise RuntimeError("Empty optimization grid")
    return best


def optimize_ablations(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        args.baseline_scores,
        dtype={"slate_id": str, "term_id": str},
        keep_default_na=False,
    )
    required = {"slate_id", "term_id", "label", "fold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{args.baseline_scores} is missing {sorted(missing)}")
    if frame["slate_id"].duplicated().any():
        raise ValueError("Baseline OOF scores contain duplicate slate IDs")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int8)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int16)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("OOF labels must be binary")
    if frame.groupby("term_id", sort=False)["fold"].nunique().gt(1).any():
        raise AssertionError("A validation term appears in more than one outer fold")
    baseline_components = [column for column in BASELINE_COMPONENTS if column in frame]
    if not baseline_components:
        raise ValueError("No baseline score components were found")

    recipes: list[tuple[str, list[str]]] = [
        ("current_gbdt_baseline", baseline_components)
    ]
    if args.frozen_scores:
        frozen = load_score_file(Path(args.frozen_scores), "frozen_semantic_cosine")
        frame = frame.merge(frozen, on="slate_id", how="left", validate="one_to_one")
        recipes.append(
            ("frozen_cosine_feature", [*baseline_components, "frozen_semantic_cosine"])
        )
    elif "semantic_cosine" in frame:
        frame["frozen_semantic_cosine"] = pd.to_numeric(
            frame["semantic_cosine"], errors="raise"
        )
        recipes.append(
            ("frozen_cosine_feature", [*baseline_components, "frozen_semantic_cosine"])
        )
    if args.contrastive_scores:
        contrastive = load_score_file(
            Path(args.contrastive_scores), "contrastive_semantic_cosine"
        )
        frame = frame.merge(
            contrastive, on="slate_id", how="left", validate="one_to_one"
        )
        recipes.append(
            (
                "contrastive_cosine_feature",
                [*baseline_components, "contrastive_semantic_cosine"],
            )
        )
    if args.hybrid_scores:
        hybrid = load_score_file(Path(args.hybrid_scores), "hybrid_semantic_cosine")
        frame = frame.merge(hybrid, on="slate_id", how="left", validate="one_to_one")
        recipes.append(
            (
                "contrastive_plus_hybrid_negatives",
                [*baseline_components, "hybrid_semantic_cosine"],
            )
        )
    if frame.isna().any().any():
        null_columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(
            f"A score file does not cover the identical OOF split: {null_columns}"
        )

    retrieval_metrics: dict[str, object] = {}
    if args.retrieval_metrics:
        retrieval_metrics = json.loads(
            Path(args.retrieval_metrics).read_text(encoding="utf-8")
        )
    default_recall = float(retrieval_metrics.get("hybrid_recall_at_100", 0.0))
    recall_by_recipe = {
        "current_gbdt_baseline": float(
            retrieval_metrics.get("tfidf_recall_at_100", 0.0)
        ),
        "frozen_cosine_feature": float(
            retrieval_metrics.get("frozen_embedding_recall_at_100", 0.0)
        ),
        "contrastive_cosine_feature": args.contrastive_recall_at_100,
        "contrastive_plus_hybrid_negatives": args.hybrid_contrastive_recall_at_100
        or default_recall,
    }
    rates = sorted(set(float(value) for value in args.rates))
    detailed: list[dict[str, object]] = []
    for name, components in recipes:
        added = set(components) - set(baseline_components)
        best = optimize_recipe(
            frame, components, rates, args.weight_step, required_positive=added
        )
        detailed.append(
            {
                "recipe": name,
                "macro_f1": best["macro_f1"],
                "retrieval_recall_at_100": recall_by_recipe.get(name, 0.0),
                "positive_rate": best["positive_rate"],
                "components": components,
                "weights": best["weights"],
            }
        )
    detailed.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["retrieval_recall_at_100"]),
            str(row["recipe"]),
        )
    )
    selected = {
        **detailed[0],
        "selection_rule": "highest Macro-F1; Recall@100 tie-breaker",
        "split_rows": len(frame),
        "split_terms": int(frame["term_id"].nunique()),
        "elapsed_seconds": time.time() - started,
    }
    serializable = pd.DataFrame(
        [
            {
                **row,
                "components": json.dumps(row["components"]),
                "weights": json.dumps(row["weights"], sort_keys=True),
            }
            for row in detailed
        ]
    )
    serializable.to_csv(output_dir / "ablation_results.csv", index=False)
    (output_dir / "selected_blend.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    print(serializable.to_string(index=False))
    print(json.dumps(selected, indent=2))


def parse_component(specification: str) -> tuple[str, Path, str]:
    try:
        name, raw_path, column = specification.split("=", 1)[0], *specification.split(
            "=", 1
        )[1].rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("Components use NAME=PATH:COLUMN syntax") from exc
    return name, Path(raw_path), column


def load_submission_component(name: str, path: Path, column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str, "term_id": str}, keep_default_na=False)
    missing = {"id", "term_id", column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    output = frame[["id", "term_id", column]].rename(columns={column: name})
    if output["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate ids")
    output[name] = pd.to_numeric(output[name], errors="raise").astype(np.float32)
    return output


def make_submission(args: argparse.Namespace) -> None:
    selected = json.loads(Path(args.selected_blend).read_text(encoding="utf-8"))
    selected_weights = {
        str(key): float(value)
        for key, value in selected["weights"].items()
        if float(value) > 0
    }
    component_frames: dict[str, pd.DataFrame] = {}
    for specification in args.component:
        name, path, column = parse_component(specification)
        component_frames[name] = load_submission_component(name, path, column)
    missing = set(selected_weights) - set(component_frames)
    if missing:
        raise ValueError(
            f"Selected blend needs missing final components: {sorted(missing)}"
        )
    ordered_names = list(selected_weights)
    merged = component_frames[ordered_names[0]]
    for name in ordered_names[1:]:
        merged = merged.merge(
            component_frames[name],
            on=["id", "term_id"],
            how="inner",
            validate="one_to_one",
        )
    expected = args.expected_rows or len(component_frames[ordered_names[0]])
    if len(merged) != expected:
        raise ValueError(
            f"Final score alignment has {len(merged):,} rows, expected {expected:,}"
        )
    blend = np.zeros(len(merged), dtype=np.float32)
    weight_sum = sum(selected_weights.values())
    for name, weight in selected_weights.items():
        ranks = (
            merged.groupby("term_id", sort=False)[name]
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float32)
        )
        blend += ranks * np.float32(weight / weight_sum)
    positive_rate = (
        args.positive_rate
        if args.positive_rate is not None
        else float(selected["positive_rate"])
    )
    n_positive = int(round(len(merged) * positive_rate))
    order = np.argsort(-blend, kind="stable")
    prediction = np.zeros(len(merged), dtype=np.int8)
    prediction[order[:n_positive]] = 1
    if not set(np.unique(prediction)).issubset({0, 1}):
        raise AssertionError("Predictions are not binary")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": merged["id"], "prediction": prediction}).to_csv(
        output_path, index=False
    )
    score_path = output_path.with_name(output_path.stem + "_scores.csv")
    pd.DataFrame(
        {"id": merged["id"], "term_id": merged["term_id"], "prob": blend}
    ).to_csv(score_path, index=False)
    report = {
        "rows": len(merged),
        "unique_ids": int(merged["id"].nunique()),
        "binary_predictions": True,
        "positives": int(prediction.sum()),
        "positive_rate": float(prediction.mean()),
        "selected_recipe": selected["recipe"],
        "weights": selected_weights,
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embedding ablations, blend selection, and final submissions."
    )
    sub = parser.add_subparsers(dest="operation", required=True)

    optimize_cmd = sub.add_parser("optimize-ablations")
    optimize_cmd.add_argument("--baseline-scores", required=True)
    optimize_cmd.add_argument("--frozen-scores")
    optimize_cmd.add_argument("--contrastive-scores")
    optimize_cmd.add_argument("--hybrid-scores")
    optimize_cmd.add_argument("--retrieval-metrics")
    optimize_cmd.add_argument("--output-dir", required=True)
    optimize_cmd.add_argument("--weight-step", type=float, default=0.10)
    optimize_cmd.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[round(value, 3) for value in np.arange(0.15, 0.351, 0.005)],
    )
    optimize_cmd.add_argument("--contrastive-recall-at-100", type=float, default=0.0)
    optimize_cmd.add_argument(
        "--hybrid-contrastive-recall-at-100", type=float, default=0.0
    )
    optimize_cmd.set_defaults(func=optimize_ablations)

    submit_cmd = sub.add_parser("make-submission")
    submit_cmd.add_argument("--selected-blend", required=True)
    submit_cmd.add_argument(
        "--component",
        action="append",
        required=True,
        help="NAME=PATH:COLUMN; repeat for every component",
    )
    submit_cmd.add_argument("--output", required=True)
    submit_cmd.add_argument("--positive-rate", type=float)
    submit_cmd.add_argument("--expected-rows", type=int, default=3_359_679)
    submit_cmd.set_defaults(func=make_submission)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
