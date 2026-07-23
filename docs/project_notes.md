# Project Notes

Last updated: 2026-07-10

## What We Are Doing

We are working on the Trendyol / TEKNOFEST e-commerce product-term relevance competition. The Kaggle stage is a binary classification/ranking task: for each search term and product candidate, predict whether the product is relevant (`1`) or not (`0`).

Important problem details:

- Metric: Macro-F1.
- Training data is positive-only: `training_pairs.csv` contains relevant product-term pairs.
- Test/submission candidates contain both relevant and irrelevant pairs.
- A useful solution needs good negative sampling, because true negative labels are not provided in training.
- The practical target is to rank each `term_id` / `item_id` pair and write an `id,prediction` submission file.

## Competition Links And Runtime Paths

- Kaggle competition URL: https://www.kaggle.com/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle
- Kaggle competition input path: `/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle`
- Kaggle working output path: `/kaggle/working`
- Local data path: `data/`
- Local outputs path: `outputs/`
- Competition PDF: `docs/competition-specification.pdf`
- Local Kaggle run logs: `kaggle_outputs/` (not tracked)

## Data

Local competition data:

| file | rows excluding header | size | columns |
| --- | ---: | ---: | --- |
| `data/training_pairs.csv` | 250,000 | 12M | `id,term_id,item_id,label` |
| `data/items.csv` | 966,444 | 383M | `item_id,title,category,brand,gender,age_group,attributes` |
| `data/submission_pairs.csv` | 3,359,679 | 163M | `id,term_id,item_id` |
| `data/terms.csv` | 50,153 | 1.6M | `term_id,query` |
| `data/sample_submission.csv` | 3,359,679 | 67M | `id,prediction` |

## Code Layout

- `src/lexical_baseline.py`: Turkish-ish normalization, token/stem heuristics, lexical pair features, scoring, threshold submissions.
- `src/pu_lgbm.py`: PU-style LightGBM classifier using lexical features and sampled negatives.
- `src/train_term_negatives.py`: mines negatives from training terms and item catalog using TF-IDF similarity bands.
- `src/pu_catboost.py`: category-aware CatBoost reranker using lexical features plus category-prior features.
- `src/sample_negative_audit.py`: samples mined negatives by band/score bucket for manual or LLM audit without using LLM labels for training.
- `src/transformer_biencoder.py`: Hugging Face transformer bi-encoder expert for term/item relevance, with cached term/item embeddings and multi-GPU Kaggle support.
- `src/ensemble_scores.py`: two-level rank blend utility for CatBoost/LGBM GBDT scores plus transformer scores, and rate-based submission creation.
- `src/category_signal.py`: reusable query-to-category TF-IDF/SGD prior model extracted from the category-aware notebook.
- `src/grouped_oof_validation.py`: leakage-safe grouped OOF pipeline that builds top-100-plus-positive slates, trains fold-specific category priors and GBDTs, combines optional transformer scores, and optimizes Macro-F1.
- `docs/grouped_oof_validation.md`: full and debug commands for the grouped five-fold workflow.
- `notebooks/category_aware_pu_lgbm.ipynb`: main local experiment notebook with recorded public scores and full category-aware candidates.
- `notebooks/catboost_gpu12000_2026-06-30.ipynb`: Kaggle-ready CatBoost GPU full-run notebook.
- `notebooks/catboost_train_term_negatives_kaggle.ipynb`: Kaggle-ready CatBoost notebook that avoids submission-pair negative leakage by using train-term negatives.
- `notebooks/vector_space_negatives_catboost_kaggle.ipynb`: self-contained Kaggle notebook using vector-space negative mining and CatBoost.
- `notebooks/grouped_5fold_oof_kaggle.ipynb`: standalone Kaggle GPU notebook with embedded source modules for the full grouped OOF workflow and resumable fold/cache controls.
- `scripts/build_grouped_oof_kaggle_notebook.py`: regenerates the standalone grouped OOF notebook from the current `src/` modules.
- `notebooks/from_zero_sub.ipynb`, `notebooks/playground.ipynb`: exploratory notebooks.

## What Has Been Done

1. Built a lexical baseline.
   - Generated full lexical scores: `outputs/lexical_scores.csv`.
   - Generated threshold and top-k submission files.

2. Submitted/tested lexical variants.
   - `outputs/submission_lexical_t575.csv`: public score `0.74`, positive rate `0.17482`.
   - `outputs/submission_lexical_t700.csv`: public score `0.61`, positive rate `0.11028`.
   - `outputs/submission_lexical_t450.csv`: public score `0.76`, positive rate `0.20307`.
   - `outputs/submission_lexical_top20.csv`: public score `0.54`, positive rate `0.19160`.

3. Built PU LightGBM v1.
   - Model: `outputs/pu_lgbm_v1.joblib`.
   - Full scores: `outputs/pu_lgbm_v1_scores.csv`.
   - Submission: `outputs/submission_pu_lgbm_v1_r20.csv`.
   - Public score: `0.78`.
   - Positive rate: `0.20000`.

4. Built category-aware features.
   - Term-to-category priors: `outputs/category_aware_notebook/term_category_topk.csv`.
   - Full scores: `outputs/category_aware_notebook/category_aware_scores.csv`.
   - Rank-normalized blend scores with old PU scores: `outputs/category_aware_notebook/category_aware_blend_scores.csv`.
   - Candidate submissions at 18%, 20%, 22%, and 24% positive rates.
   - No public Kaggle score is recorded in the repo for these category-aware candidate files.

5. Built CatBoost category-aware candidates.
   - Model: `outputs/catboost_category_aware_2026-06-30.cbm`.
   - Scores: `outputs/catboost_category_aware_2026-06-30_scores.csv`.
   - Submission: `outputs/catboost_category_aware_2026-06-30.csv`.
   - Also has `outputs/catboost_2026-06-30.csv`.
   - Both are 20% positive-rate files.
   - No public Kaggle score is recorded in the repo for these files.

6. Built cleaner train-term negative mining.
   - Output: `outputs/train_term_negatives.csv`.
   - Rows: 872,183 excluding header.
   - Columns: `term_id,item_id,negative_band,tfidf_score`.
   - Purpose: train on negatives mined from training terms and catalog items instead of sampling negatives from the test/submission candidate pool.

7. Built and submitted the MoE-style GBDT + transformer rank blend on Kaggle.
   - Negative samples stayed deterministic from the vector miner; LLM use is audit-only.
   - Transformer bi-encoder checkpoint was saved as `transformer_biencoder.tar.gz` for session restarts.
   - Transformer score file was saved/reused as `transformer_biencoder_scores.csv`.
   - Final blend score file was saved/reused as `moe_gbdt_rankblend_scores.csv`.
   - Best submitted result so far: `submission_moe_gbdt_rankblend_r26.csv`, public score `0.870`.

8. Implemented term-grouped five-fold candidate-slate validation.
   - Folds are assigned by `term_id`, matching the disjoint training/test query structure.
   - Each validation slate is an unsupervised top-100 catalog retrieval unioned with all known positives.
   - Fold-specific category priors and models never use outer held-out labels.
   - A separate inner term split handles early stopping; the outer fold is scoring-only.
   - CatBoost, LightGBM, optional transformer OOF scores, blend weights, positive rates, and the base-100 constraint share one optimizer.

## What Worked

- A simple lexical score was already competitive when the threshold produced about 20% positives.
  - Best lexical recorded result: `submission_lexical_t450.csv`, public score `0.76`.
- PU LightGBM improved over lexical.
  - Best recorded result in the repo: `submission_pu_lgbm_v1_r20.csv`, public score `0.78`.
- Positive rate around 24%-26% is currently strongest for the MoE blend.
  - Earlier models worked around 18%-22%, but the transformer/GBDT blend improved as rate increased.
- Category priors and score blending were implemented successfully and produced full 3,359,679-row submission files.
- The two-level rank blend gave the largest recorded jump so far, moving from the old `0.78`-`0.79` range to `0.870`.

## What Did Not Work Or Is Risky

- Too strict lexical thresholding hurt recall.
  - `submission_lexical_t700.csv` had only 11.0% positives and scored `0.61`.
- Fixed top-20 per query was weak.
  - `submission_lexical_top20.csv` scored `0.54`.
- Synthetic validation is not reliable enough by itself.
  - Negatives are sampled/noisy, so local Macro-F1 can mislead.
  - Kaggle leaderboard feedback is still needed.
- Older PU negative sampling used submission-pair candidates as unlabeled negatives.
  - This works pragmatically, but it is a leakage/robustness risk.
  - Newer scripts prefer `outputs/train_term_negatives.csv`; using submission negatives now requires explicit `--allow-submission-negatives`.
- Kaggle session restarts are expensive if cached artifacts are not saved.
  - Save `moe_gbdt_rankblend_scores.csv`, `transformer_biencoder_scores.csv`, `transformer_biencoder.tar.gz`, and `train_term_negatives.csv` as reusable Kaggle dataset/cache artifacts.
- Public leaderboard feedback is now strong, but it is still public-LB feedback.
  - Keep changes controlled: rate sweeps, blend-weight sweeps, or seed averaging before major retraining changes.

## Key Submission Artifacts

Known scored files:

| path | rows | positive rate | public score | note |
| --- | ---: | ---: | ---: | --- |
| `outputs/submission_lexical_t575.csv` | 3,359,679 | 0.17482 | 0.74 | lexical threshold 5.75 |
| `outputs/submission_lexical_t700.csv` | 3,359,679 | 0.11028 | 0.61 | too strict |
| `outputs/submission_lexical_t450.csv` | 3,359,679 | 0.20307 | 0.76 | best lexical |
| `outputs/submission_lexical_top20.csv` | 3,359,679 | 0.19160 | 0.54 | fixed per-query top-k weak |
| `outputs/submission_pu_lgbm_v1_r20.csv` | 3,359,679 | 0.20000 | 0.78 | best recorded result |
| `submission_moe_gbdt_rankblend_r22.csv` | 3,359,679 | 0.22000 | 0.854 | Kaggle MoE rank blend |
| `submission_moe_gbdt_rankblend_r24.csv` | 3,359,679 | 0.24000 | 0.867 | Kaggle MoE rank blend |
| `submission_moe_gbdt_rankblend_r26.csv` | 3,359,679 | 0.26000 | 0.870 | best recorded result |

Unscored candidate files:

| path | rows | positive rate | note |
| --- | ---: | ---: | --- |
| `outputs/category_aware_notebook/submission_category_aware_r18.csv` | 3,359,679 | 0.18000 | category-aware LGBM |
| `outputs/category_aware_notebook/submission_category_aware_r20.csv` | 3,359,679 | 0.20000 | category-aware LGBM |
| `outputs/category_aware_notebook/submission_category_aware_r22.csv` | 3,359,679 | 0.22000 | category-aware LGBM |
| `outputs/category_aware_notebook/submission_category_aware_blend_r20.csv` | 3,359,679 | 0.20000 | blend with PU v1 scores |
| `outputs/category_aware_notebook/submission_category_aware_blend_r22.csv` | 3,359,679 | 0.22000 | blend with PU v1 scores |
| `outputs/catboost_category_aware_2026-06-30.csv` | 3,359,679 | 0.20000 | CatBoost category-aware |
| `outputs/catboost_2026-06-30.csv` | 3,359,679 | 0.20000 | CatBoost candidate |
| `outputs/pu_category_rankblend_2026-06-30.csv` | 3,359,679 | 0.20000 | rank blend candidate |

## Suggested Next Step

Reuse `moe_gbdt_rankblend_scores.csv` first before retraining. Generate rate sweeps around the current best:

1. `r25`, `r26`, `r27`, and `r28` from the saved MoE score file.
2. If quota allows, sweep transformer-heavy blend weights and test the best-looking rates around `0.26`.
3. Only retrain the transformer after cheap score/rate/blend sweeps are exhausted.

Keep `submission_moe_gbdt_rankblend_r26.csv` as the known-good baseline.
