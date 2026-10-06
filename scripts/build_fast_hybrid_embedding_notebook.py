"""Generate the self-contained Kaggle notebook for hybrid embedding retrieval."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "fast_hybrid_embedding_retrieval_kaggle.ipynb"
MODULES = [
    "lexical_baseline.py",
    "train_term_negatives.py",
    "pu_lgbm.py",
    "pu_catboost.py",
    "category_signal.py",
    "ensemble_scores.py",
    "grouped_oof_validation.py",
    "hybrid_embedding_retrieval.py",
    "contrastive_biencoder.py",
    "embedding_experiment.py",
]


def markdown(source: str) -> dict[str, object]:
    """Create a markdown cell from a plain string."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    """Create an unexecuted code cell from Python source."""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def module_cell(filename: str) -> dict[str, object]:
    """Embed one repository module as a Kaggle writefile cell."""

    source = (ROOT / "src" / filename).read_text(encoding="utf-8")
    return code(f"%%writefile src/{filename}\n{source}")


cells: list[dict[str, object]] = [
    markdown("""# Fast hybrid embedding retrieval and negative sampling

This standalone Kaggle notebook implements the fast embedding workflow as a
contrastive shared-encoder two-tower model. It caches frozen multilingual MiniLM
catalog embeddings and exact top-500 retrieval once, mixes representative random
negatives with conservative lexical/embedding semi-hard negatives, and keeps
checkpoint selection term-grouped and separate from the outer holdout.

The default `MODE="pilot"` uses a fixed stratified 80/20 term holdout. After
selecting a recipe by Macro-F1 (Recall@100 breaks ties), use `MODE="confirm"` to
run five grouped folds for only that pilot recipe. Use `MODE="final"` with the
confirmed blend JSON to train on all positive terms, score every submission pair
by encoding each unique query/item once, and write a validated submission.

All long stages are resumable. Save the packaged artifact archive or the output
directory as a Kaggle dataset between sessions. The original
`grouped_5fold_oof_kaggle.ipynb` is not used or modified.

The negative mix follows the representative-random plus conservative semi-hard
pattern supported by [Facebook's retrieval study](https://ar5iv.labs.arxiv.org/html/2006.11632)
and [JD's e-commerce model](https://ar5iv.labs.arxiv.org/html/2006.02282).
"""),
    code("""from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys

# --------------------------- EDIT THIS CELL ---------------------------
MODE = "pilot"  # "pilot", "confirm", or "final"
SEED = 42
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cuda"
SKIP_EXISTING = True
PACKAGE_ARTIFACTS = True

# The pilot evaluates both contrastive variants. Confirm/final read the winning
# pilot/confirmed JSON from an attached cache or an earlier run.
PILOT_SELECTED_BLEND = None  # e.g. Path("/kaggle/input/.../selected_blend.json")
CONFIRMED_SELECTED_BLEND = None  # required for MODE="final"
COMPUTE_CONTRASTIVE_RECALL = True

RUN_GBDT = True
GBDT_MODELS = ["lgbm"]  # add "catboost" for the full current baseline
CATBOOST_DEVICES = "0"
CATBOOST_ITERATIONS = 2000
LGBM_ESTIMATORS = 4000

TRAIN_EPOCHS = 2
TRAIN_BATCH_SIZE = 64
ENCODE_BATCH_SIZE = 512
WEIGHT_STEP = 0.10

# Final GBDT components normally come from the confirmed full-data baseline.
# Every entry uses: selected-component-name -> (score CSV, score column).
# The notebook adds the selected frozen/contrastive semantic component itself.
FINAL_COMPONENTS = {
    # "lgbm_prob": (Path("/kaggle/input/my-scores/pu_lgbm_scores.csv"), "prob"),
    # "catboost_prob": (Path("/kaggle/input/my-scores/catboost_scores.csv"), "prob"),
    # "lexical_score": (Path("/kaggle/input/my-scores/lexical_scores.csv"), "score"),
}
EXPECTED_SUBMISSION_ROWS = 3_359_679
# ---------------------------------------------------------------------

WORK_DIR = Path("/kaggle/working")
SRC_DIR = WORK_DIR / "src"
ROOT_OUT = WORK_DIR / "fast_hybrid_embedding"
CACHE_DIR = ROOT_OUT / "frozen_cache"
RUN_DIR = ROOT_OUT / MODE
SRC_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)

data_candidates = [
    Path("/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"),
    Path("/kaggle/input/trendyol-e-ticaret-yarismasi-2026-kaggle"),
]
DATA_DIR = next((path for path in data_candidates if (path / "training_pairs.csv").exists()), None)
if DATA_DIR is None:
    matches = list(Path("/kaggle/input").glob("**/training_pairs.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate competition data; matches={matches[:10]}"
        )
    DATA_DIR = matches[0].parent

print(f"MODE={MODE} DATA_DIR={DATA_DIR} RUN_DIR={RUN_DIR}")
"""),
    markdown("## Runtime dependencies"),
    code("""required = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "tqdm": "tqdm",
    "torch": "torch",
    "transformers": "transformers",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "rapidfuzz": "rapidfuzz",
    "text_unidecode": "text-unidecode",
}
missing = [
    package
    for module, package in required.items()
    if importlib.util.find_spec(module) is None
]
if missing:
    print("installing", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("all dependencies are available")
"""),
    markdown("## Embedded source modules"),
]

for module in MODULES:
    cells.append(module_cell(module))

cells.extend(
    [
        code("""sys.path.insert(0, str(SRC_DIR))
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(SRC_DIR)
ENV.setdefault("TOKENIZERS_PARALLELISM", "false")
ENV.setdefault("PYTHONHASHSEED", str(SEED))
ENV.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def run_command(arguments):
    command = [sys.executable, *map(str, arguments)]
    print("\\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=ENV)

run_command([SRC_DIR / "hybrid_embedding_retrieval.py", "--help"])
run_command([SRC_DIR / "contrastive_biencoder.py", "--help"])
"""),
        markdown("## 1. Frozen MiniLM cache and exact hybrid retrieval"),
        code("""cache_manifest = CACHE_DIR / "manifest.json"
run_command([
    SRC_DIR / "hybrid_embedding_retrieval.py", "build-cache",
    "--data-dir", DATA_DIR, "--cache-dir", CACHE_DIR,
    "--model-name", BASE_MODEL, "--query-max-length", 48, "--item-max-length", 128,
    "--encode-batch-size", ENCODE_BATCH_SIZE, "--retrieval-topk", 500,
    "--query-block-size", 64, "--item-block-size", 16384, "--device", DEVICE,
])

display(json.loads(cache_manifest.read_text()))
"""),
        markdown("## 2. Conservative hybrid negative files"),
        code("""contrastive_negative_pool = ROOT_OUT / "contrastive_negative_pool.csv"
reranker_negatives = ROOT_OUT / "reranker_negatives.csv"
cache_key = json.loads(cache_manifest.read_text())["cache_key"]

def negative_cache_matches(path):
    audit = path.with_suffix(".audit.json")
    return (
        path.exists()
        and audit.exists()
        and json.loads(audit.read_text()).get("cache_fingerprint") == cache_key
    )

if not SKIP_EXISTING or not negative_cache_matches(contrastive_negative_pool):
    run_command([
        SRC_DIR / "hybrid_embedding_retrieval.py", "mine-contrastive",
        "--data-dir", DATA_DIR, "--cache-dir", CACHE_DIR,
        "--output", contrastive_negative_pool, "--seed", SEED,
    ])
if not SKIP_EXISTING or not negative_cache_matches(reranker_negatives):
    run_command([
        SRC_DIR / "hybrid_embedding_retrieval.py", "mine-reranker",
        "--data-dir", DATA_DIR, "--cache-dir", CACHE_DIR,
        "--output", reranker_negatives, "--seed", SEED,
    ])

display(json.loads(contrastive_negative_pool.with_suffix(".audit.json").read_text()))
display(json.loads(reranker_negatives.with_suffix(".audit.json").read_text()))
"""),
        markdown("## 3. Fixed pilot or five-fold hybrid validation slates"),
        code("""slate_path = RUN_DIR / "hybrid_validation_slates.csv"
split_manifest = RUN_DIR / "split_manifest.csv"
retrieval_metrics = RUN_DIR / "retrieval_metrics.json"
if MODE in {"pilot", "confirm"}:
    slate_cache_matches = (
        slate_path.exists() and retrieval_metrics.exists()
        and json.loads(retrieval_metrics.read_text()).get("cache_fingerprint") == cache_key
    )
    if not SKIP_EXISTING or not slate_cache_matches:
        run_command([
            SRC_DIR / "hybrid_embedding_retrieval.py", "build-slates",
            "--data-dir", DATA_DIR, "--cache-dir", CACHE_DIR,
            "--output-dir", RUN_DIR, "--mode", MODE, "--seed", SEED,
        ])
    display(json.loads(retrieval_metrics.read_text()))
else:
    print("final mode uses submission_pairs.csv; no outer validation slate is built")
"""),
        markdown("## 4. Current GBDT baseline on the identical split"),
        code("""baseline_scores = RUN_DIR / "baseline_oof_scores.csv"
if MODE in {"pilot", "confirm"} and RUN_GBDT:
    folds = [0] if MODE == "pilot" else list(range(5))
    gbdt_root = RUN_DIR / "gbdt"
    for fold in folds:
        fold_output = gbdt_root / f"fold_{fold}" / "oof_scores.csv"
        if SKIP_EXISTING and fold_output.exists():
            print(f"resume {fold_output}")
            continue
        arguments = [
            SRC_DIR / "grouped_oof_validation.py", "run-fold",
            "--data-dir", DATA_DIR, "--output-dir", gbdt_root, "--slates", slate_path,
            "--fold", fold, "--models", *GBDT_MODELS, "--inner-valid-size", 0.10,
            "--catboost-iterations", CATBOOST_ITERATIONS, "--lgbm-estimators", LGBM_ESTIMATORS,
            "--task-type", "GPU" if "catboost" in GBDT_MODELS else "CPU",
            "--devices", CATBOOST_DEVICES, "--seed", SEED, "--no-semantic-features",
        ]
        run_command(arguments)
    if MODE == "pilot":
        shutil.copy2(gbdt_root / "fold_0" / "oof_scores.csv", baseline_scores)
    else:
        run_command([
            SRC_DIR / "grouped_oof_validation.py", "combine",
            "--output-dir", gbdt_root, "--n-splits", 5, "--output", baseline_scores,
        ])
elif MODE in {"pilot", "confirm"}:
    raise ValueError("Set RUN_GBDT=True or provide baseline_oof_scores.csv in RUN_DIR")
"""),
        markdown("## 5. Contrastive checkpoints and held-out scores"),
        code("""import pandas as pd

slates = (
    pd.read_csv(
        slate_path,
        dtype={"slate_id": str, "term_id": str, "item_id": str},
    )
    if MODE != "final"
    else None
)

def train_and_score(label, use_semi_hard, fold):
    model_dir = RUN_DIR / "models" / label / f"fold_{fold}"
    score_path = RUN_DIR / "scores" / f"{label}_fold_{fold}.csv"
    pair_path = RUN_DIR / "scores" / f"pairs_fold_{fold}.csv"
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    heldout = slates.loc[slates["fold"].eq(fold), ["slate_id", "term_id", "item_id"]]
    heldout.to_csv(pair_path, index=False)
    if not SKIP_EXISTING or not (model_dir / "training_metrics.json").exists():
        arguments = [
            SRC_DIR / "contrastive_biencoder.py", "train",
            "--data-dir", DATA_DIR, "--negative-pool", contrastive_negative_pool,
            "--model-dir", model_dir, "--model-name", BASE_MODEL,
            "--split-manifest", split_manifest, "--outer-fold", fold,
            "--cache-manifest", cache_manifest, "--projection-dim", 256,
            "--temperature", 0.05, "--query-max-length", 48, "--item-max-length", 128,
            "--batch-size", TRAIN_BATCH_SIZE, "--epochs", TRAIN_EPOCHS,
            "--inner-valid-size", 0.10, "--uniform-multiplier", 3,
            "--device", DEVICE, "--fp16", "--seed", SEED,
            "--use-semi-hard" if use_semi_hard else "--no-use-semi-hard",
        ]
        run_command(arguments)
    if not SKIP_EXISTING or not score_path.exists():
        run_command([
            SRC_DIR / "contrastive_biencoder.py", "score-pairs",
            "--data-dir", DATA_DIR, "--model-dir", model_dir,
            "--pairs", pair_path, "--id-column", "slate_id", "--output", score_path,
            "--encode-batch-size", ENCODE_BATCH_SIZE, "--device", DEVICE, "--fp16",
        ])
    return model_dir, score_path

contrastive_scores = None
hybrid_scores = None
models_for_recall = {}
if MODE == "pilot":
    plain_model, contrastive_scores = train_and_score("contrastive_random", False, 0)
    hybrid_model, hybrid_scores = train_and_score("contrastive_hybrid", True, 0)
    models_for_recall = {"contrastive": plain_model, "hybrid": hybrid_model}
elif MODE == "confirm":
    if PILOT_SELECTED_BLEND is None:
        raise ValueError("MODE='confirm' requires PILOT_SELECTED_BLEND from the pilot artifacts")
    pilot_selected = json.loads(Path(PILOT_SELECTED_BLEND).read_text())
    winning_recipe = pilot_selected["recipe"]
    print("confirming pilot winner", winning_recipe)
    fold_scores = []
    for fold in range(5):
        if winning_recipe == "contrastive_cosine_feature":
            model, score = train_and_score("contrastive_random", False, fold)
            contrastive_scores = score
        elif winning_recipe == "contrastive_plus_hybrid_negatives":
            model, score = train_and_score("contrastive_hybrid", True, fold)
            hybrid_scores = score
        else:
            break
        fold_scores.append(pd.read_csv(score, dtype={"slate_id": str}))
        models_for_recall[f"fold_{fold}"] = model
    if fold_scores:
        combined_path = RUN_DIR / "scores" / f"{winning_recipe}_oof.csv"
        pd.concat(fold_scores, ignore_index=True).sort_values("slate_id").to_csv(
            combined_path, index=False
        )
        if winning_recipe == "contrastive_cosine_feature":
            contrastive_scores = combined_path
        else:
            hybrid_scores = combined_path
"""),
        markdown("## 6. Retrieval Recall@100 for checkpoint tie-breaking"),
        code("""from hybrid_embedding_retrieval import cache_paths, exact_cosine_topk

def checkpoint_recall_at_100(label, model_dir, heldout_fold=0):
    embedding_dir = RUN_DIR / "encoded" / label
    term_file = embedding_dir / "terms_embeddings.npy"
    item_file = embedding_dir / "items_embeddings.npy"
    if not SKIP_EXISTING or not term_file.exists():
        run_command([
            SRC_DIR / "contrastive_biencoder.py", "encode",
            "--data-dir", DATA_DIR, "--model-dir", model_dir,
            "--output-dir", embedding_dir, "--entity", "terms",
            "--batch-size", ENCODE_BATCH_SIZE, "--device", DEVICE, "--fp16",
        ])
    if not SKIP_EXISTING or not item_file.exists():
        run_command([
            SRC_DIR / "contrastive_biencoder.py", "encode",
            "--data-dir", DATA_DIR, "--model-dir", model_dir,
            "--output-dir", embedding_dir, "--entity", "items",
            "--batch-size", ENCODE_BATCH_SIZE, "--device", DEVICE, "--fp16",
        ])
    term_ids = pd.read_csv(embedding_dir / "terms_ids.csv", dtype=str)["term_id"].tolist()
    item_ids = pd.read_csv(embedding_dir / "items_ids.csv", dtype=str)["item_id"].tolist()
    term_pos = {value: index for index, value in enumerate(term_ids)}
    item_pos = {value: index for index, value in enumerate(item_ids)}
    split = pd.read_csv(split_manifest, dtype={"term_id": str})
    selected_terms = set(split.loc[split["fold"].eq(heldout_fold), "term_id"])
    positives = pd.read_csv(DATA_DIR / "training_pairs.csv", dtype=str)
    positives = positives.loc[positives["term_id"].isin(selected_terms)]
    q_rows = [term_pos[value] for value in sorted(selected_terms)]
    q_ids = sorted(selected_terms)
    term_embedding = __import__("numpy").load(term_file, mmap_mode="r")[q_rows]
    item_embedding = __import__("numpy").load(item_file, mmap_mode="r")
    indices, _ = exact_cosine_topk(term_embedding, item_embedding, 100, device=DEVICE)
    retrieved = {
        term: {item_ids[position] for position in indices[row]}
        for row, term in enumerate(q_ids)
    }
    hits = sum(
        str(row.item_id) in retrieved[str(row.term_id)]
        for row in positives.itertuples(index=False)
    )
    value = hits / max(1, len(positives))
    metrics = {"recall_at_100": value, "positives": len(positives)}
    (embedding_dir / "recall_at_100.json").write_text(json.dumps(metrics, indent=2))
    return value

contrastive_recall = 0.0
hybrid_contrastive_recall = 0.0
if MODE == "pilot" and COMPUTE_CONTRASTIVE_RECALL:
    contrastive_recall = checkpoint_recall_at_100(
        "contrastive_random", models_for_recall["contrastive"]
    )
    hybrid_contrastive_recall = checkpoint_recall_at_100(
        "contrastive_hybrid", models_for_recall["hybrid"]
    )
print({"contrastive_recall_at_100": contrastive_recall,
       "hybrid_contrastive_recall_at_100": hybrid_contrastive_recall})
"""),
        markdown("## 7. Identical-split ablations and recipe selection"),
        code("""if MODE in {"pilot", "confirm"}:
    heldout_folds = {0} if MODE == "pilot" else set(range(5))
    frozen_scores = RUN_DIR / "scores" / "frozen_oof.csv"
    frozen_frame = slates.loc[
        slates["fold"].isin(heldout_folds),
        ["slate_id", "term_id", "item_id", "semantic_cosine", "semantic_rank_pct"],
    ]
    frozen_frame.to_csv(frozen_scores, index=False)
    arguments = [
        SRC_DIR / "embedding_experiment.py", "optimize-ablations",
        "--baseline-scores", baseline_scores, "--frozen-scores", frozen_scores,
        "--retrieval-metrics", retrieval_metrics, "--output-dir", RUN_DIR,
        "--weight-step", WEIGHT_STEP,
        "--contrastive-recall-at-100", contrastive_recall,
        "--hybrid-contrastive-recall-at-100", hybrid_contrastive_recall,
    ]
    if contrastive_scores is not None:
        arguments.extend(["--contrastive-scores", contrastive_scores])
    if hybrid_scores is not None:
        arguments.extend(["--hybrid-scores", hybrid_scores])
    run_command(arguments)
    display(pd.read_csv(RUN_DIR / "ablation_results.csv"))
    display(json.loads((RUN_DIR / "selected_blend.json").read_text()))
"""),
        markdown("## 8. Final full-data training, unique-pair scoring, and submission"),
        code("""if MODE == "final":
    if CONFIRMED_SELECTED_BLEND is None:
        raise ValueError("MODE='final' requires CONFIRMED_SELECTED_BLEND")
    selected = json.loads(Path(CONFIRMED_SELECTED_BLEND).read_text())
    recipe = selected["recipe"]
    final_specs = dict(FINAL_COMPONENTS)
    semantic_scores = RUN_DIR / "final_semantic_scores.csv"
    required_components = {
        name for name, weight in selected["weights"].items() if float(weight) > 0
    }

    # With the fast default (LightGBM + lexical), final mode is self-contained.
    # User-supplied FINAL_COMPONENTS still take precedence when confirmed full
    # score files are attached.
    if "lexical_score" in required_components and "lexical_score" not in final_specs:
        lexical_scores = RUN_DIR / "final_lexical_scores.csv"
        if not SKIP_EXISTING or not lexical_scores.exists():
            run_command([SRC_DIR / "lexical_baseline.py", "score", "--data-dir", DATA_DIR,
                         "--pairs", "submission_pairs.csv", "--output", lexical_scores])
        final_specs["lexical_score"] = (lexical_scores, "score")
    if "lgbm_prob" in required_components and "lgbm_prob" not in final_specs:
        lgbm_model = RUN_DIR / "final_pu_lgbm.joblib"
        lgbm_scores = RUN_DIR / "final_pu_lgbm_scores.csv"
        if not SKIP_EXISTING or not lgbm_model.exists():
            run_command([SRC_DIR / "pu_lgbm.py", "train", "--data-dir", DATA_DIR,
                         "--negatives", reranker_negatives, "--model", lgbm_model,
                         "--n-estimators", LGBM_ESTIMATORS, "--seed", SEED])
        if not SKIP_EXISTING or not lgbm_scores.exists():
            run_command([SRC_DIR / "pu_lgbm.py", "predict", "--data-dir", DATA_DIR,
                         "--model", lgbm_model, "--output", lgbm_scores])
        final_specs["lgbm_prob"] = (lgbm_scores, "prob")

    if recipe in {"contrastive_cosine_feature", "contrastive_plus_hybrid_negatives"}:
        final_model = RUN_DIR / "models" / "final_contrastive"
        use_hybrid = recipe == "contrastive_plus_hybrid_negatives"
        if not SKIP_EXISTING or not (final_model / "training_metrics.json").exists():
            run_command([
                SRC_DIR / "contrastive_biencoder.py", "train", "--data-dir", DATA_DIR,
                "--negative-pool", contrastive_negative_pool, "--model-dir", final_model,
                "--model-name", BASE_MODEL, "--cache-manifest", cache_manifest,
                "--projection-dim", 256, "--temperature", 0.05, "--query-max-length", 48,
                "--item-max-length", 128, "--batch-size", TRAIN_BATCH_SIZE,
                "--epochs", TRAIN_EPOCHS, "--inner-valid-size", 0.10,
                "--uniform-multiplier", 3, "--device", DEVICE, "--fp16", "--seed", SEED,
                "--use-semi-hard" if use_hybrid else "--no-use-semi-hard",
            ])
        if not SKIP_EXISTING or not semantic_scores.exists():
            run_command([
                SRC_DIR / "contrastive_biencoder.py", "score-pairs", "--data-dir", DATA_DIR,
                "--model-dir", final_model, "--pairs", DATA_DIR / "submission_pairs.csv",
                "--output", semantic_scores, "--encode-batch-size", ENCODE_BATCH_SIZE,
                "--device", DEVICE, "--fp16",
            ])
        component_name = "hybrid_semantic_cosine" if use_hybrid else "contrastive_semantic_cosine"
        final_specs[component_name] = (semantic_scores, "semantic_cosine")
    elif recipe == "frozen_cosine_feature":
        if not SKIP_EXISTING or not semantic_scores.exists():
            import numpy as np
            from contrastive_biencoder import score_embedding_pairs
            pairs = pd.read_csv(DATA_DIR / "submission_pairs.csv", dtype=str)
            item_ids = pd.read_csv(CACHE_DIR / "item_ids.csv", dtype=str)["item_id"].tolist()
            term_ids = pd.read_csv(CACHE_DIR / "term_ids.csv", dtype=str)["term_id"].tolist()
            scores = score_embedding_pairs(
                pairs, term_ids, np.load(CACHE_DIR / "term_embeddings.npy", mmap_mode="r"),
                item_ids, np.load(CACHE_DIR / "item_embeddings.npy", mmap_mode="r"),
            )
            frozen = pairs[["id", "term_id", "item_id"]].copy()
            frozen["semantic_cosine"] = scores
            frozen["semantic_rank_pct"] = frozen.groupby("term_id")[
                "semantic_cosine"
            ].rank(method="average", pct=True)
            frozen.to_csv(semantic_scores, index=False)
        final_specs["frozen_semantic_cosine"] = (semantic_scores, "semantic_cosine")

    missing = required_components - set(final_specs)
    if missing:
        raise ValueError(f"Add full-data score paths to FINAL_COMPONENTS for: {sorted(missing)}")
    arguments = [
        SRC_DIR / "embedding_experiment.py", "make-submission",
        "--selected-blend", CONFIRMED_SELECTED_BLEND,
        "--output", RUN_DIR / "submission_fast_hybrid_embedding.csv",
        "--expected-rows", EXPECTED_SUBMISSION_ROWS,
    ]
    for name, (path, column) in final_specs.items():
        if name in required_components:
            arguments.extend(["--component", f"{name}={path}:{column}"])
    run_command(arguments)
    display(json.loads((RUN_DIR / "submission_fast_hybrid_embedding.report.json").read_text()))
"""),
        markdown("## 9. Package resumable artifacts"),
        code("""if PACKAGE_ARTIFACTS:
    archive = shutil.make_archive(
        str(WORK_DIR / f"fast_hybrid_embedding_{MODE}_artifacts"),
        "gztar",
        ROOT_OUT,
    )
    print(f"packaged {archive}")

print("run complete", {"mode": MODE, "run_dir": str(RUN_DIR), "cache": str(CACHE_DIR)})
"""),
    ]
)

for index, cell in enumerate(cells):
    cell["id"] = f"fast-hybrid-{index:03d}"


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUTPUT} with {len(cells)} cells")
