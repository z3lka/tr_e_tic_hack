from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "grouped_5fold_oof_kaggle.ipynb"
MODULES = [
    "lexical_baseline.py",
    "train_term_negatives.py",
    "pu_catboost.py",
    "category_signal.py",
    "ensemble_scores.py",
    "grouped_oof_validation.py",
    "transformer_biencoder.py",
]


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def module_cell(filename: str) -> dict[str, object]:
    module_source = (ROOT / "src" / filename).read_text(encoding="utf-8")
    source = f"%%writefile src/{filename}\n{module_source}"
    return code(source)


cells: list[dict[str, object]] = [
    markdown(
        """# Trendyol term-grouped 5-fold OOF validation

This notebook is self-contained: it writes all required repository modules into `/kaggle/working/src`. It builds candidate slates, trains leakage-safe CatBoost/LightGBM folds, optionally trains transformer folds, combines OOF scores, and optimizes Macro-F1 blend/rate/count constraints.

Recommended Kaggle workflow:

1. For a one-session attempt, use `STAGE = \"all\"` and all five folds.
2. If the session is too short, first run `STAGE = \"build\"`, save `/kaggle/working/grouped_oof` as a Kaggle dataset, attach it next session, set `CACHE_ROOT`, and run one or two folds at a time with `STAGE = \"folds\"`.
3. After all fold artifacts are cached, use `STAGE = \"optimize\"`.

For leakage-safe transformer OOF training, `BASE_MODEL` must be the original pretrained encoder, not your previous full-data fine-tuned `bi-encoder` checkpoint.
"""
    ),
    code(
        '''from pathlib import Path
import os
import shutil
import subprocess
import sys

# --------------------------- EDIT THIS CELL ---------------------------
STAGE = "all"  # "build", "folds", "optimize", or "all"
FOLDS_TO_RUN = [0, 1, 2, 3, 4]

RUN_CATBOOST = True
RUN_LGBM = True
RUN_TRANSFORMER = True

# Set this to an attached Kaggle dataset directory containing a previous
# grouped_oof output folder, for example:
# CACHE_ROOT = Path("/kaggle/input/my-trendyol-oof-cache/grouped_oof")
CACHE_ROOT = None

SKIP_EXISTING = True
N_SPLITS = 5
SEED = 42
BASE_CANDIDATES = 100

# Use the original pretrained model. With Internet disabled, attach an offline
# Kaggle dataset containing this Hugging Face model and set BASE_MODEL to it.
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSFORMER_EPOCHS = 2
TRANSFORMER_BATCH_SIZE = 64
TRANSFORMER_GRAD_ACCUM = 1

CATBOOST_ITERATIONS = 2000
CATBOOST_DEVICES = "0:1"  # CatBoost multi-GPU syntax uses a colon.
LGBM_ESTIMATORS = 4000
INNER_VALID_SIZE = 0.10
WEIGHT_STEP = 0.10

# "cuda" makes transformer_biencoder.py discover and use all visible GPUs.
# To select explicitly, use "cuda:0,1".
TRANSFORMER_DEVICE = "cuda"

PACKAGE_CACHE = True
# ---------------------------------------------------------------------

WORK_DIR = Path("/kaggle/working")
SRC_DIR = WORK_DIR / "src"
OUT_DIR = WORK_DIR / "grouped_oof"
SRC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)

data_candidates = [
    Path("/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"),
    Path("/kaggle/input/trendyol-e-ticaret-yarismasi-2026-kaggle"),
]
DATA_DIR = next((path for path in data_candidates if (path / "training_pairs.csv").exists()), None)
if DATA_DIR is None:
    matches = list(Path("/kaggle/input").glob("**/training_pairs.csv"))
    if len(matches) == 1:
        DATA_DIR = matches[0].parent
    else:
        raise FileNotFoundError(f"Could not uniquely locate competition data; matches={matches[:10]}")

if CACHE_ROOT is not None:
    CACHE_ROOT = Path(CACHE_ROOT)
    if not CACHE_ROOT.exists():
        raise FileNotFoundError(CACHE_ROOT)
    shutil.copytree(CACHE_ROOT, OUT_DIR, dirs_exist_ok=True)
    print(f"restored cache from {CACHE_ROOT}")

print(f"STAGE={STAGE} FOLDS_TO_RUN={FOLDS_TO_RUN}")
print(f"DATA_DIR={DATA_DIR}")
print(f"OUT_DIR={OUT_DIR}")
'''
    ),
    markdown("## Install/check runtime dependencies"),
    code(
        '''import importlib.util

required_packages = {
    "rapidfuzz": "rapidfuzz",
    "text_unidecode": "text-unidecode",
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "sklearn": "scikit-learn",
}
if RUN_TRANSFORMER:
    required_packages.update({"torch": "torch", "transformers": "transformers"})

missing_packages = [pip_name for module, pip_name in required_packages.items() if importlib.util.find_spec(module) is None]
if missing_packages:
    print("installing", missing_packages)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing_packages], check=True)
else:
    print("all dependencies are already available")
'''
    ),
    markdown("## Write embedded source modules"),
]

for module in MODULES:
    cells.append(module_cell(module))

cells.extend(
    [
        code(
            '''sys.path.insert(0, str(SRC_DIR))
env = os.environ.copy()
env["PYTHONPATH"] = str(SRC_DIR)
env.setdefault("TOKENIZERS_PARALLELISM", "false")

def run_command(arguments):
    command = [sys.executable, *map(str, arguments)]
    print("\\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)

run_command([SRC_DIR / "grouped_oof_validation.py", "--help"])
print("embedded source import check passed")
'''
        ),
        markdown("## Stage 1 — build five-fold candidate slates"),
        code(
            '''slate_path = OUT_DIR / "validation_slates.csv"
should_build = STAGE in {"build", "all"}
if should_build and (not SKIP_EXISTING or not slate_path.exists()):
    run_command([
        SRC_DIR / "grouped_oof_validation.py",
        "build-slates",
        "--data-dir", DATA_DIR,
        "--output-dir", OUT_DIR,
        "--n-splits", N_SPLITS,
        "--base-candidates", BASE_CANDIDATES,
        "--max-features", 250000,
        "--min-df", 2,
        "--word-ngram-max", 2,
        "--chunk-size", 128,
        "--seed", SEED,
    ])
elif should_build:
    print(f"skip existing {slate_path}")
else:
    print("build stage disabled")

if STAGE in {"folds", "optimize"} and not slate_path.exists():
    raise FileNotFoundError(
        f"{slate_path} is required. Run STAGE='build' first or set CACHE_ROOT to a saved cache."
    )
'''
        ),
        code(
            '''import json
import pandas as pd

if (OUT_DIR / "slate_summary.json").exists():
    print(json.dumps(json.loads((OUT_DIR / "slate_summary.json").read_text()), indent=2))
if slate_path.exists():
    slate_preview = pd.read_csv(slate_path, nrows=5)
    display(slate_preview)
'''
        ),
        markdown("## Stage 2 — fold-specific CatBoost and LightGBM"),
        code(
            '''should_run_folds = STAGE in {"folds", "all"}
models = []
if RUN_CATBOOST:
    models.append("catboost")
if RUN_LGBM:
    models.append("lgbm")

if should_run_folds and models:
    for fold in FOLDS_TO_RUN:
        fold_score_path = OUT_DIR / f"fold_{fold}" / "oof_scores.csv"
        if SKIP_EXISTING and fold_score_path.exists():
            print(f"skip existing {fold_score_path}")
            continue
        arguments = [
            SRC_DIR / "grouped_oof_validation.py",
            "run-fold",
            "--data-dir", DATA_DIR,
            "--output-dir", OUT_DIR,
            "--slates", slate_path,
            "--fold", fold,
            "--models", *models,
            "--task-type", "GPU" if RUN_CATBOOST else "CPU",
            "--devices", CATBOOST_DEVICES,
            "--threads", -1,
            "--inner-valid-size", INNER_VALID_SIZE,
            "--early-stopping-rounds", 100,
            "--catboost-iterations", CATBOOST_ITERATIONS,
            "--lgbm-estimators", LGBM_ESTIMATORS,
            "--seed", SEED,
        ]
        run_command(arguments)
elif not should_run_folds:
    print("GBDT fold stage disabled")
else:
    print("No GBDT models selected")
'''
        ),
        markdown("## Stage 3 — optional transformer folds"),
        code(
            '''if should_run_folds and RUN_TRANSFORMER:
    for fold in FOLDS_TO_RUN:
        fold_dir = OUT_DIR / f"fold_{fold}"
        transformer_score_path = fold_dir / "transformer_oof_scores.csv"
        transformer_model_dir = fold_dir / "transformer"
        if SKIP_EXISTING and transformer_score_path.exists():
            print(f"skip existing {transformer_score_path}")
            continue
        run_command([
            SRC_DIR / "transformer_biencoder.py",
            "train",
            "--data-dir", DATA_DIR,
            "--slates", slate_path,
            "--fold", fold,
            "--model-dir", transformer_model_dir,
            "--valid-output", transformer_score_path,
            "--model-name", BASE_MODEL,
            "--epochs", TRANSFORMER_EPOCHS,
            "--batch-size", TRANSFORMER_BATCH_SIZE,
            "--grad-accum-steps", TRANSFORMER_GRAD_ACCUM,
            "--inner-valid-size", INNER_VALID_SIZE,
            "--device", TRANSFORMER_DEVICE,
            "--fp16",
            "--seed", SEED,
        ])
elif not should_run_folds:
    print("transformer fold stage disabled")
else:
    print("RUN_TRANSFORMER=False; skipping transformer folds")
'''
        ),
        markdown("## Combine completed OOF folds"),
        code(
            '''gbdt_fold_paths = [OUT_DIR / f"fold_{fold}" / "oof_scores.csv" for fold in range(N_SPLITS)]
all_gbdt_folds_ready = all(path.exists() for path in gbdt_fold_paths)
if all_gbdt_folds_ready:
    run_command([
        SRC_DIR / "grouped_oof_validation.py",
        "combine",
        "--output-dir", OUT_DIR,
        "--n-splits", N_SPLITS,
    ])
else:
    missing = [str(path) for path in gbdt_fold_paths if not path.exists()]
    print("OOF combine waits for all five GBDT folds. Missing:")
    print("\\n".join(missing))
'''
        ),
        markdown("## Stage 4 — optimize Macro-F1 blend, rate, and constraint"),
        code(
            '''combined_oof_path = OUT_DIR / "oof_scores.csv"
should_optimize = STAGE in {"optimize", "all"}
if should_optimize and combined_oof_path.exists():
    transformer_fold_paths = [
        OUT_DIR / f"fold_{fold}" / "transformer_oof_scores.csv"
        for fold in range(N_SPLITS)
    ]
    components = []
    if RUN_CATBOOST:
        components.append("catboost_prob")
    if RUN_LGBM:
        components.append("lgbm_prob")
    if all(path.exists() for path in transformer_fold_paths):
        components.append("transformer_prob")
    else:
        print("not every transformer fold is ready; optimizing available GBDT components only")

    run_command([
        SRC_DIR / "grouped_oof_validation.py",
        "optimize",
        "--oof-scores", combined_oof_path,
        "--output-dir", OUT_DIR,
        "--components", *components,
        "--weight-step", WEIGHT_STEP,
        "--constraint-bases", 0, BASE_CANDIDATES,
        "--show-top", 30,
    ])
elif should_optimize:
    print(f"cannot optimize until {combined_oof_path} exists")
else:
    print("optimization stage disabled")
'''
        ),
        code(
            '''if (OUT_DIR / "oof_best.json").exists():
    print((OUT_DIR / "oof_best.json").read_text())
if (OUT_DIR / "oof_optimization.csv").exists():
    display(pd.read_csv(OUT_DIR / "oof_optimization.csv").head(30))
'''
        ),
        markdown("## Package outputs for the next Kaggle session"),
        code(
            '''if PACKAGE_CACHE:
    archive_base = WORK_DIR / "grouped_oof_cache"
    archive_path = shutil.make_archive(
        str(archive_base),
        "gztar",
        root_dir=OUT_DIR.parent,
        base_dir=OUT_DIR.name,
    )
    print(f"wrote {archive_path} ({Path(archive_path).stat().st_size / 1024**3:.2f} GiB)")

print("\\nAvailable artifacts:")
for path in sorted(OUT_DIR.glob("**/*")):
    if path.is_file():
        print(f"{path.relative_to(OUT_DIR)}  {path.stat().st_size / 1024**2:.1f} MiB")
'''
        ),
    ]
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {len(cells)} cells)")
