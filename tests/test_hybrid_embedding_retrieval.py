from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contrastive_biencoder import (  # noqa: E402
    apply_outer_term_split,
    assemble_candidate_item_ids,
    grouped_inner_split,
    known_positive_mask_numpy,
    score_embedding_pairs,
)
from embedding_experiment import make_submission, optimize_ablations  # noqa: E402
from hybrid_embedding_retrieval import (  # noqa: E402
    ConservativeNegativeFilter,
    build_slates,
    cache_is_valid,
    exact_cosine_topk,
    load_manifest,
    make_term_split_manifest,
    mine_negative_rows,
    write_embedding_cache_fixture,
    write_negative_output,
)


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)


def make_items(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [f"i{index:04d}" for index in range(count)],
            "title": [f"unique product title {index}" for index in range(count)],
            "category": [f"category {index % 7}" for index in range(count)],
            "brand": [f"brand {index % 11}" for index in range(count)],
            "gender": [""] * count,
            "age_group": [""] * count,
            "attributes": [f"attribute {index}" for index in range(count)],
        }
    )


class RetrievalAndMiningTests(unittest.TestCase):
    def test_exact_blockwise_cosine_matches_direct_scores(self) -> None:
        rng = np.random.default_rng(10)
        query = normalized(rng.normal(size=(7, 13)))
        items = normalized(rng.normal(size=(31, 13)))
        indices, scores = exact_cosine_topk(query, items, 9, query_block_size=3, item_block_size=8)
        direct = query @ items.T
        for row in range(len(query)):
            expected = np.lexsort((np.arange(len(items)), -direct[row]))[:9]
            np.testing.assert_array_equal(indices[row], expected)
            np.testing.assert_allclose(scores[row], direct[row, expected], atol=1e-6)

    def test_grouped_splits_are_disjoint_and_repeatable(self) -> None:
        positives = pd.DataFrame(
            [(f"t{term:02d}", f"i{term:02d}_{item}") for term in range(25) for item in range(1 + term % 4)],
            columns=["term_id", "item_id"],
        )
        pilot_a = make_term_split_manifest(positives, "pilot", 42)
        pilot_b = make_term_split_manifest(positives, "pilot", 42)
        pd.testing.assert_frame_equal(pilot_a, pilot_b)
        holdout = set(pilot_a.loc[pilot_a["fold"].eq(0), "term_id"])
        training = set(pilot_a.loc[pilot_a["fold"].eq(1), "term_id"])
        self.assertFalse(holdout & training)
        self.assertEqual(len(holdout), 5)

        confirm = make_term_split_manifest(positives, "confirm", 42)
        self.assertEqual(set(confirm["fold"]), set(range(5)))
        self.assertTrue(confirm["term_id"].is_unique)

        inner_train, inner_valid, _ = grouped_inner_split(positives, 0.10, 42)
        self.assertFalse(set(inner_train["term_id"]) & set(inner_valid["term_id"]))

    def test_hybrid_negative_ratios_bands_filters_and_determinism(self) -> None:
        items = make_items(650)
        items.loc[0, ["title", "brand"]] = ["one two three four five six seven eight nine ten", "same brand"]
        items.loc[1, ["title", "brand"]] = ["ONE two three four five six seven eight nine ten", "same-brand"]
        items.loc[2, "title"] = "one two three four five six seven eight nine"
        positives = pd.DataFrame({"term_id": ["t0"], "item_id": ["i0000"]})
        negative_filter = ConservativeNegativeFilter(items[["item_id", "title", "brand"]], positives)
        self.assertFalse(negative_filter.valid("t0", 0))
        self.assertFalse(negative_filter.valid("t0", 1))
        self.assertFalse(negative_filter.valid("t0", 2))

        lexical = np.arange(500, dtype=np.int32)[None, :]
        embedding = np.arange(50, 550, dtype=np.int32)[None, :]
        scores = np.linspace(1.0, 0.0, 500, dtype=np.float32)[None, :]
        pool = {
            "lexical_indices": lexical,
            "lexical_scores": scores,
            "embedding_indices": embedding,
            "embedding_scores": scores,
        }
        first = mine_negative_rows(["t0"], {"t0": 0}, pool, negative_filter, 42, "contrastive")
        second = mine_negative_rows(["t0"], {"t0": 0}, pool, negative_filter, 42, "contrastive")
        self.assertEqual(first, second)
        frame = pd.DataFrame(first)
        self.assertEqual(len(frame), 20)
        self.assertEqual(frame["item_id"].nunique(), 20)
        self.assertEqual((frame["negative_source"] == "tfidf_semi_hard").sum(), 10)
        self.assertEqual((frame["negative_source"] == "embedding_semi_hard").sum(), 10)
        self.assertTrue(frame["source_rank"].between(101, 500).all())
        self.assertFalse(frame["item_id"].isin(["i0000", "i0001", "i0002"]).any())
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "contrastive_negative_pool.csv"
            write_negative_output(first, output, positives, "contrastive", 42)
            written = pd.read_csv(output)
            self.assertEqual(written.columns.tolist(), ["term_id", "item_id", "negative_source", "source_rank", "source_score"])
            audit = json.loads(output.with_suffix(".audit.json").read_text())
            self.assertEqual(audit["positive_overlaps"], 0)
            self.assertEqual(audit["duplicates"], 0)

        reranker = pd.DataFrame(
            mine_negative_rows(["t0"], {"t0": 0}, pool, negative_filter, 42, "reranker")
        )
        self.assertEqual(len(reranker), 50)
        self.assertEqual((reranker["negative_source"] == "uniform_random").sum(), 20)
        ranked = reranker.loc[reranker["source_rank"].gt(0), "source_rank"]
        self.assertTrue(ranked.between(21, 200).all())

        shortage_pool = {
            "lexical_indices": np.zeros((1, 500), dtype=np.int32),
            "lexical_scores": scores,
            "embedding_indices": np.zeros((1, 500), dtype=np.int32),
            "embedding_scores": scores,
        }
        backfill_a = mine_negative_rows(["t0"], {"t0": 0}, shortage_pool, negative_filter, 99, "contrastive")
        backfill_b = mine_negative_rows(["t0"], {"t0": 0}, shortage_pool, negative_filter, 99, "contrastive")
        self.assertEqual(backfill_a, backfill_b)
        self.assertEqual(len(backfill_a), 20)
        self.assertTrue(all(str(row["negative_source"]).startswith("uniform_backfill") for row in backfill_a))
        self.assertTrue(all(row["item_id"] != "i0000" for row in backfill_a))

    def test_known_positive_mask_keeps_only_designated_target(self) -> None:
        terms = ["q1", "q2"]
        candidates = ["a", "b", "c", "d", "e"]
        known = {"q1": {"a", "b", "e"}, "q2": {"b", "c"}}
        mask = known_positive_mask_numpy(terms, candidates, known, np.asarray([0, 1]))
        np.testing.assert_array_equal(mask[0], [False, True, False, False, True])
        np.testing.assert_array_equal(mask[1], [False, False, True, False, False])

    def test_cached_pair_scores_match_direct_cosine(self) -> None:
        term_ids = ["t0", "t1"]
        item_ids = ["i0", "i1", "i2"]
        terms = normalized(np.asarray([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]))
        items = normalized(np.asarray([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]))
        pairs = pd.DataFrame({"term_id": ["t0", "t1", "t0"], "item_id": ["i2", "i0", "i1"]})
        actual = score_embedding_pairs(pairs, term_ids, terms, item_ids, items, chunk_size=2)
        expected = np.asarray([terms[0] @ items[2], terms[1] @ items[0], terms[0] @ items[1]])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_outer_manifest_excludes_only_heldout_terms(self) -> None:
        positives = pd.DataFrame(
            {"term_id": ["a", "a", "b", "c"], "item_id": ["1", "2", "3", "4"]}
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "split.csv"
            pd.DataFrame({"term_id": ["a", "b", "c"], "fold": [0, 1, 1]}).to_csv(path, index=False)
            training, heldout = apply_outer_term_split(positives, path, 0)
        self.assertEqual(heldout, {"a"})
        self.assertEqual(set(training["term_id"]), {"b", "c"})

    def test_batch_recipe_is_20_20_60_and_deterministic(self) -> None:
        batch = pd.DataFrame(
            {
                "term_id": [f"t{index}" for index in range(4)],
                "item_id": [f"p{index}" for index in range(4)],
                "row_index": np.arange(4),
            }
        )
        pool = {
            f"t{index}": {"lexical": [f"l{index}"], "embedding": [f"e{index}"]}
            for index in range(4)
        }
        catalog = np.asarray([*(f"p{i}" for i in range(4)), *(f"l{i}" for i in range(4)), *(f"e{i}" for i in range(4)), *(f"u{i}" for i in range(100))])
        first = assemble_candidate_item_ids(batch, pool, catalog, 42, 0)
        second = assemble_candidate_item_ids(batch, pool, catalog, 42, 0)
        self.assertEqual(first[0], second[0])
        self.assertEqual(len(first[0]), 20)
        counts = pd.Series(first[2]).value_counts().to_dict()
        self.assertEqual(counts["positive"], 4)
        self.assertEqual(counts["lexical"], 2)
        self.assertEqual(counts["embedding"], 2)
        self.assertEqual(counts["uniform"], 12)


class SyntheticEndToEndTests(unittest.TestCase):
    def test_cache_mining_slates_scoring_ablation_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_dir = root / "data"
            cache_dir = root / "cache"
            run_dir = root / "pilot"
            data_dir.mkdir()
            items = make_items(140)
            terms = pd.DataFrame({"term_id": [f"t{index}" for index in range(10)], "query": [f"query {index}" for index in range(10)]})
            positives = pd.DataFrame({"id": [f"p{index}" for index in range(10)], "term_id": terms["term_id"], "item_id": [f"i{index:04d}" for index in range(10)], "label": 1})
            items.to_csv(data_dir / "items.csv", index=False)
            terms.to_csv(data_dir / "terms.csv", index=False)
            positives.to_csv(data_dir / "training_pairs.csv", index=False)

            rng = np.random.default_rng(9)
            item_embedding = normalized(rng.normal(size=(len(items), 12)))
            term_embedding = normalized(rng.normal(size=(len(terms), 12)))
            embedding_indices, embedding_scores = exact_cosine_topk(term_embedding, item_embedding, 100)
            lexical_indices = np.vstack([np.roll(np.arange(len(items)), term)[:100] for term in range(len(terms))])
            lexical_scores = np.broadcast_to(np.linspace(1, 0, 100), lexical_indices.shape)
            write_embedding_cache_fixture(
                cache_dir,
                items["item_id"].tolist(),
                terms["term_id"].tolist(),
                item_embedding,
                term_embedding,
                lexical_indices,
                lexical_scores,
                embedding_indices,
                embedding_scores,
            )
            self.assertTrue(cache_is_valid(cache_dir))
            self.assertTrue(load_manifest(cache_dir)["complete"])

            build_slates(Namespace(data_dir=str(data_dir), cache_dir=str(cache_dir), output_dir=str(run_dir), mode="pilot", seed=42, limit_terms=0))
            slate = pd.read_csv(run_dir / "hybrid_validation_slates.csv", dtype={"slate_id": str})
            self.assertEqual(int(slate["label"].sum()), len(positives))
            self.assertTrue({"semantic_cosine", "semantic_rank_pct"}.issubset(slate.columns))
            heldout = slate.loc[slate["fold"].eq(0)].reset_index(drop=True)
            heldout["lgbm_prob"] = heldout["label"] * 0.8 + (1 - heldout["label"]) * 0.2
            heldout["lexical_score"] = heldout["retrieval_score"]
            baseline = run_dir / "baseline.csv"
            heldout.to_csv(baseline, index=False)
            semantic = run_dir / "semantic.csv"
            heldout[["slate_id", "term_id", "item_id", "semantic_cosine"]].to_csv(semantic, index=False)

            optimize_ablations(
                Namespace(
                    baseline_scores=str(baseline), frozen_scores=str(semantic), contrastive_scores=str(semantic),
                    hybrid_scores=str(semantic), retrieval_metrics=str(run_dir / "retrieval_metrics.json"),
                    output_dir=str(run_dir), weight_step=0.5, rates=[0.1, 0.2, 0.3],
                    contrastive_recall_at_100=0.5, hybrid_contrastive_recall_at_100=0.6,
                )
            )
            selected = json.loads((run_dir / "selected_blend.json").read_text())
            final_pairs = heldout[["slate_id", "term_id"]].rename(columns={"slate_id": "id"})
            final_paths: list[str] = []
            for component in selected["weights"]:
                if not selected["weights"][component]:
                    continue
                path = run_dir / f"final_{component}.csv"
                values = heldout["lgbm_prob"] if component in {"lgbm_prob", "catboost_prob", "lexical_score"} else heldout["semantic_cosine"]
                frame = final_pairs.copy()
                frame["value"] = values.to_numpy()
                frame.to_csv(path, index=False)
                final_paths.append(f"{component}={path}:value")
            submission = run_dir / "submission.csv"
            make_submission(
                Namespace(
                    selected_blend=str(run_dir / "selected_blend.json"), component=final_paths,
                    output=str(submission), positive_rate=None, expected_rows=len(heldout),
                )
            )
            output = pd.read_csv(submission, dtype={"id": str})
            self.assertEqual(len(output), len(heldout))
            self.assertTrue(output["id"].is_unique)
            self.assertTrue(set(output["prediction"]).issubset({0, 1}))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional locally")
    def test_one_synthetic_masked_infonce_training_step(self) -> None:
        import torch
        from torch.nn import functional as F

        from contrastive_biencoder import masked_infonce_loss

        query = torch.nn.Parameter(torch.randn(3, 5))
        item = torch.nn.Parameter(torch.randn(9, 5))
        optimizer = torch.optim.SGD([query, item], lr=0.05)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        mask = torch.from_numpy(
            known_positive_mask_numpy(
                ["a", "b", "c"],
                ["p0", "p1", "p2", "x", "y", "z", "u", "v", "w"],
                {"a": {"p0", "p1"}, "b": {"p1"}, "c": {"p2"}},
                targets.numpy(),
            )
        )
        before = query.detach().clone()
        loss = masked_infonce_loss(
            F.normalize(query, dim=1), F.normalize(item, dim=1), mask, targets, 0.05, F
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, query.detach()))


if __name__ == "__main__":
    unittest.main()
