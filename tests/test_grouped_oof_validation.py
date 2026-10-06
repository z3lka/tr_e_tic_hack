from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ensemble_scores import make_prediction
from grouped_oof_validation import evaluate_rates, make_term_folds
from transformer_biencoder import split_inner_validation_by_term


def test_make_term_folds_keeps_each_term_in_one_fold() -> None:
    rows: list[tuple[str, str]] = []
    for term_index in range(25):
        for item_index in range(1 + term_index % 7):
            rows.append(
                (f"term_{term_index:02d}", f"item_{term_index:02d}_{item_index:02d}")
            )
    positives = pd.DataFrame(rows, columns=["term_id", "item_id"])

    folds = make_term_folds(positives, n_splits=5, seed=42)

    assert len(folds) == positives["term_id"].nunique()
    assert folds["term_id"].is_unique
    assert set(folds["fold"]) == set(range(5))
    assert folds.groupby("fold").size().max() - folds.groupby("fold").size().min() <= 1


def test_evaluate_rates_matches_materialized_predictions() -> None:
    scores = np.asarray([0.9, 0.8, 0.2, 0.1, 0.95, 0.7, 0.4, 0.3], dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.int8)
    term_ids = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
    rates = [0.5, 0.625]

    evaluated = evaluate_rates(scores, labels, term_ids, rates, constraint_base=3)
    by_rate = {rate: score for rate, score, _ in evaluated}

    for rate in rates:
        prediction = make_prediction(
            pd.Series(scores),
            rate,
            term_ids=pd.Series(term_ids),
            min_positive_excess_base=3,
        )
        expected = f1_score(labels, prediction, average="macro")
        assert np.isclose(by_rate[rate], expected)


def test_transformer_inner_split_is_grouped_by_term() -> None:
    rows: list[tuple[str, str, int]] = []
    for term_index in range(20):
        rows.append((f"term_{term_index:02d}", f"positive_{term_index:02d}", 1))
        rows.append((f"term_{term_index:02d}", f"negative_{term_index:02d}", 0))
    frame = pd.DataFrame(rows, columns=["term_id", "item_id", "label"])

    train, valid = split_inner_validation_by_term(frame, valid_size=0.20, seed=42)

    assert set(train["term_id"]).isdisjoint(set(valid["term_id"]))
    assert train["term_id"].nunique() == 16
    assert valid["term_id"].nunique() == 4
    assert set(train["label"]) == {0, 1}
    assert set(valid["label"]) == {0, 1}
