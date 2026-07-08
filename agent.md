# Agent Notes

Last updated: 2026-07-08

## What We Are Doing

We are working on the Trendyol / TEKNOFEST e-commerce product-term relevance competition. The Kaggle stage is a binary classification/ranking task: for each search term and product candidate, predict whether the product is relevant (`1`) or not (`0`).

Important problem details:

- Metric: Macro-F1.
- Training data is positive-only: `training_pairs.csv` contains relevant product-term pairs.
- Test/submission candidates contain both relevant and irrelevant pairs.
- A useful solution needs good negative sampling, because true negative labels are not provided in training.
- The practical target is to rank each `term_id` / `item_id` pair and write an `id,prediction` submission file.

## Competition Links And Paths

- Kaggle competition URL: https://www.kaggle.com/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle
- Kaggle competition input path: `/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle`
- Kaggle working output path: `/kaggle/working`
- Local repo path: `/Users/z3lka/trendyol_e_ticaret_hack`
- Local data path: `/Users/z3lka/trendyol_e_ticaret_hack/data`
- Local outputs path: `/Users/z3lka/trendyol_e_ticaret_hack/outputs`
- Local competition PDF: `/Users/z3lka/trendyol_e_ticaret_hack/E-TİCARET_T_NFSoo.pdf`
- Kaggle run log with detected input path: `/Users/z3lka/trendyol_e_ticaret_hack/kaggle_outputs/negative-sampling.log`

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
- `notebooks/category_aware_pu_lgbm.ipynb`: main local experiment notebook with recorded public scores and full category-aware candidates.
- `notebooks/catboost_gpu12000_2026-06-30.ipynb`: Kaggle-ready CatBoost GPU full-run notebook.
- `notebooks/catboost_train_term_negatives_kaggle.ipynb`: Kaggle-ready CatBoost notebook that avoids submission-pair negative leakage by using train-term negatives.
- `notebooks/vector_space_negatives_catboost_kaggle.ipynb`: self-contained Kaggle notebook using vector-space negative mining and CatBoost.
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

## What Worked

- A simple lexical score was already competitive when the threshold produced about 20% positives.
  - Best lexical recorded result: `submission_lexical_t450.csv`, public score `0.76`.
- PU LightGBM improved over lexical.
  - Best recorded result in the repo: `submission_pu_lgbm_v1_r20.csv`, public score `0.78`.
- Positive rate around 18%-22% appears to be the useful range.
  - The best recorded file uses exactly 20%.
- Category priors and score blending were implemented successfully and produced full 3,359,679-row submission files.

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
- Category-aware, CatBoost, and blend candidates exist, but the repo does not record public scores for them.
  - Do not assume they beat `0.78` until submitted.

## Key Submission Artifacts

Known scored files:

| path | rows | positive rate | public score | note |
| --- | ---: | ---: | ---: | --- |
| `outputs/submission_lexical_t575.csv` | 3,359,679 | 0.17482 | 0.74 | lexical threshold 5.75 |
| `outputs/submission_lexical_t700.csv` | 3,359,679 | 0.11028 | 0.61 | too strict |
| `outputs/submission_lexical_t450.csv` | 3,359,679 | 0.20307 | 0.76 | best lexical |
| `outputs/submission_lexical_top20.csv` | 3,359,679 | 0.19160 | 0.54 | fixed per-query top-k weak |
| `outputs/submission_pu_lgbm_v1_r20.csv` | 3,359,679 | 0.20000 | 0.78 | best recorded result |

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

When the Kaggle submission quota is available, submit one or two unscored candidates against the known `0.78` baseline. Start with:

1. `outputs/category_aware_notebook/submission_category_aware_blend_r20.csv`
2. `outputs/category_aware_notebook/submission_category_aware_blend_r22.csv`

Keep `outputs/submission_pu_lgbm_v1_r20.csv` as the known-good baseline.
