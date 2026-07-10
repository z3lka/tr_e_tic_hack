# Term-grouped 5-fold validation

This pipeline validates on unseen search terms and on candidate slates shaped like the competition test set. It replaces the old random pair split for model selection; it does not change the final full-data training command automatically.

Artifacts are resumable under `outputs/grouped_oof/`:

- `term_folds.csv`: deterministic fold assignment, one row per training term.
- `validation_slates.csv`: top-100 TF-IDF retrieval candidates unioned with every known positive.
- `retrieval_vectorizer.joblib`: fitted unsupervised slate retriever vocabulary.
- `fold_N/term_category_topk.csv`: category prior fit without fold `N` labels.
- `fold_N/oof_scores.csv`: held-out CatBoost/LightGBM predictions.
- `fold_N/transformer_oof_scores.csv`: optional held-out transformer predictions.
- `oof_scores.csv`: combined predictions from all folds.
- `oof_optimization.csv` and `oof_best.json`: blend/rate/constraint search.

## 1. Build candidate slates

Run this once. It fits an unsupervised retrieval vocabulary, retrieves 100 catalog products per training term, unions the retrieved products with all known positives, and labels every other retrieved product as negative.

```bash
PYTHONPATH=src python src/grouped_oof_validation.py build-slates \
  --data-dir data \
  --output-dir outputs/grouped_oof \
  --n-splits 5 \
  --base-candidates 100 \
  --max-features 250000 \
  --chunk-size 128
```

The summary includes `retrieved_positive_recall`. Candidate-count distributions should also be compared with `data/submission_pairs.csv`; a large mismatch means the validation retriever should be improved before trusting absolute OOF scores.

## 2. Train CatBoost and LightGBM folds

On a GPU Kaggle session, the following trains all five folds sequentially. LightGBM remains CPU-based; `--task-type GPU` applies to CatBoost.

```bash
PYTHONPATH=src python src/grouped_oof_validation.py run-all \
  --data-dir data \
  --output-dir outputs/grouped_oof \
  --n-splits 5 \
  --models catboost lgbm \
  --task-type GPU \
  --devices 0 \
  --catboost-iterations 2000 \
  --lgbm-estimators 4000
```

For restartable runs, execute one fold at a time by replacing `run-all` with `run-fold --fold 0`, then repeat for folds 1 through 4. After all folds finish:

```bash
PYTHONPATH=src python src/grouped_oof_validation.py combine \
  --output-dir outputs/grouped_oof \
  --n-splits 5
```

## 3. Optional transformer folds

The transformer trainer accepts the same slates. Train one checkpoint per fold:

```bash
PYTHONPATH=src python src/transformer_biencoder.py train \
  --data-dir data \
  --slates outputs/grouped_oof/validation_slates.csv \
  --fold 0 \
  --model-dir outputs/grouped_oof/fold_0/transformer \
  --valid-output outputs/grouped_oof/fold_0/transformer_oof_scores.csv \
  --device cuda \
  --fp16
```

Repeat for folds 1 through 4, changing all three fold numbers. Then rerun the `combine` command. It automatically merges each `fold_N/transformer_oof_scores.csv` into the combined OOF file.

## 4. Optimize Macro-F1

The optimizer rank-normalizes every component within its held-out fold, searches simplex blend weights, positive rates, and both unconstrained and base-100 constrained decisions.

```bash
PYTHONPATH=src python src/grouped_oof_validation.py optimize \
  --oof-scores outputs/grouped_oof/oof_scores.csv \
  --output-dir outputs/grouped_oof \
  --components catboost_prob lgbm_prob transformer_prob \
  --weight-step 0.10 \
  --constraint-bases 0 100
```

If transformer folds have not been trained, omit `transformer_prob`. Start with the default `--weight-step 0.25` for a quick search, then rerun near the best configuration with `0.10`.

## Debug smoke run

The limits below are only for verifying the pipeline, not for measuring model quality:

```bash
PYTHONPATH=src python src/grouped_oof_validation.py build-slates \
  --output-dir /tmp/trendyol_oof_smoke \
  --n-splits 2 \
  --base-candidates 20 \
  --limit-terms 30 \
  --limit-items 500 \
  --max-features 5000 \
  --min-df 1
```

## Leakage boundary

- A term appears in exactly one fold.
- Fold-specific category priors use positive pairs from the other folds only.
- Models train only on the other folds' candidate rows.
- An inner term-group split from the four training folds is used for early stopping.
- The outer held-out fold is never used for fitting or checkpoint selection; it is scored only after the model is fixed.
- Retrieval may use held-out query text and catalog text because both are available at inference; it never uses held-out labels to rank the base 100.
- Known held-out positives are unioned only to construct and label the validation slate, mirroring the inferred test candidate construction.
