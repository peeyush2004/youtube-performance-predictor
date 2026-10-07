# Baseline Model Training and Evaluation

## 1. Scope

This stage builds and evaluates a leakage-safe baseline for estimating observed view counts in YouTube trending snapshots. It uses the approved modeling dataset, targets, features, and chronological split. This project estimates views recorded in a snapshot; it does not forecast future views.

## 2. Split design

The split is built in `src/split_dataset.py`.

- Primary temporal variable: `trending_date`
- Split rule: chronological date windows with whole-video assignment
- Leakage constraint: no `video_id` overlap across train, validation, and test
- Cross-boundary policy: exclude any video whose timeline spans multiple windows

The script creates:

- `data/processed/splits/train.csv`
- `data/processed/splits/validation.csv`
- `data/processed/splits/test.csv`
- `data/processed/splits/split_metadata.json`

The split metadata records:

- `train_end_inclusive = 2018-04-02`
- `validation_end_inclusive = 2018-05-12`
- excluded cross-boundary videos: 1,472
- excluded cross-boundary rows: 20,771
- zero `video_id` overlap across splits

## 3. Targets and features

The dataset uses two approved targets:

- `observed_views`: direct raw view count from the snapshot
- `log_observed_views`: `log1p(observed_views)`

The baseline feature set is Group A only:

- `region`
- `category_name`
- `title_length`
- `title_word_count`
- `description_length`
- `description_word_count`
- `description_missing`
- `tag_count`
- `publish_year`
- `publish_month`
- `publish_day_of_week`
- `publish_hour`

Audit-only features and traceability columns are retained in the processed file but excluded from baseline predictors.

## 4. Preprocessing

The pipeline in `src/train_baseline.py` uses:

- categorical: constant imputation with `Unknown`, followed by `OneHotEncoder(handle_unknown="ignore")`
- numeric: median imputation followed by `StandardScaler`

Preprocessing is fit on the training split and reused for validation and test.

## 5. Models evaluated

Baseline regressors evaluated on the chronological split:

- `DummyRegressor`
- `LinearRegression`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`

Each model was trained once per target:

- `raw_views` using `observed_views`
- `log_views` using `log_observed_views`

Validation data is used for selection; the held-out test split is reserved for final reporting.

## 6. Selection rule

The selected baseline is chosen by the lowest validation MAE on the original view-count scale, with RMSE used as a tie-breaker.

## 7. Verified baseline results

Baseline artifacts:

- `data/processed/baseline_experiment_results.csv`
- `models/baseline_regression_pipeline.joblib`
- `models/baseline_regression_metadata.json`

### Selected model

- Model: `RandomForestRegressor`
- Target: `log_views` (`log_observed_views` column)
- Feature group: `Group_A`
- Validation MAE (views): 500,163.33
- Validation RMSE (views): 1,818,047.28

### Held-out test metrics for the selected model

- Test MAE (views): 1,013,177.49
- Test RMSE (views): 4,183,634.21
- Test R²: 0.02047
- Test median absolute error (views): 128,854.46
- Test MAE (log scale): 1.1686
- Test RMSE (log scale): 1.5365

### Best among evaluated Group A baselines

`RandomForestRegressor` achieved the strongest validation MAE for the raw-target baseline and also performed best among the tested log-target pipelines. The selected metadata records the log-target model.

## 8. Execution status

The baseline workflow was executed against the leakage-safe split. `src/split_dataset.py` created the split outputs, and `src/train_baseline.py` produced evaluation results and the selected model artifact.

# Baseline Diagnostic Analysis

## 1. Split verification

The split is leakage-safe.

- Train range: 2017-11-14 to 2018-04-02; rows: 237,400; unique videos: 123,672
- Validation range: 2018-04-03 to 2018-05-12; rows: 50,981; unique videos: 32,257
- Test range: 2018-05-13 to 2018-06-14; rows: 51,003; unique videos: 26,780

Video overlap checks are zero across all split pairs:

- train vs validation: 0
- train vs test: 0
- validation vs test: 0

Date ordering also holds:

- max(train_date) = 2018-04-02
- min(validation_date) = 2018-04-03
- max(validation_date) = 2018-05-12
- min(test_date) = 2018-05-13

Therefore, `max(train_date) < min(validation_date)` and `max(validation_date) < min(test_date)` are both satisfied.

## 2. Metric breakdown on the saved baseline

The selected baseline is the saved `RandomForestRegressor` trained on the log target and Group A features.

### Train performance

Original view scale:

- MAE: 268,934.72
- RMSE: 2,304,340.83
- R²: 0.82699
- Median AE: 23,516.52

Log scale:

- MAE: 0.2969
- RMSE: 0.4128
- R²: 0.9532
- Median AE: 0.2131

### Validation performance

Original view scale:

- MAE: 500,163.33
- RMSE: 1,818,047.28
- R²: 0.09525
- Median AE: 73,600.01

Log scale:

- MAE: 1.0976
- RMSE: 1.4187
- R²: 0.4182
- Median AE: 0.8845

### Test performance

Original view scale:

- MAE: 1,013,177.49
- RMSE: 4,183,634.21
- R²: 0.02047
- Median AE: 128,854.46

Log scale:

- MAE: 1.1686
- RMSE: 1.5365
- R²: 0.1620
- Median AE: 0.9001

Validation and test performance are materially worse than training performance. The gap is especially large on the original view scale because the target is heavy-tailed and predictions are compressed toward the center.

## 3. DummyRegressor comparison

From the saved baseline experiment results:

- Validation MAE, `DummyRegressor`: 569,442.00
- Validation MAE, `RandomForestRegressor`: 500,163.33
- Improvement: 69,278.67 views lower MAE (12.2%)
- Test MAE, `DummyRegressor`: 1,108,571.99
- Test MAE, `RandomForestRegressor`: 1,013,177.49
- Improvement: 95,394.50 views lower MAE (8.6%)

The RandomForest beats the dummy baseline on both validation and test, but absolute errors remain large. This does not support a claim of accurate prediction across the full distribution.

## 4. Error analysis by actual view bucket

| Bucket | Count | MAE | Median AE | RMSE | Mean actual views | Mean predicted views |
|---|---:|---:|---:|---:|---:|---:|
| <10K | 529 | 38,541.15 | 22,756.60 | 70,287.30 | 5,654.79 | 44,178.35 |
| 10K-100K | 15,761 | 44,685.48 | 25,079.39 | 97,783.76 | 50,149.52 | 73,598.17 |
| 100K-1M | 24,005 | 264,745.20 | 164,730.40 | 511,724.20 | 368,783.20 | 232,415.40 |
| 1M-10M | 9,626 | 2,375,006.75 | 1,626,017.17 | 3,237,447.32 | 2,831,994.00 | 737,602.20 |
| >10M | 1,082 | 20,086,310.00 | 14,224,669.00 | 26,941,520.00 | 21,918,290.00 | 2,349,445.00 |

Low-view videos are overpredicted on average, while very large videos are substantially underpredicted.

## 5. Region and category performance differences

### Test performance by region

| Region | Observations | MAE | Median AE | RMSE | Mean actual views | Mean predicted views |
|---|---:|---:|---:|---:|---:|---:|
| JP | 3,112 | 240,128.90 | 45,391.82 | 1,338,998.00 | 301,082.80 | 82,584.69 |
| RU | 6,256 | 247,386.00 | 44,992.42 | 1,249,089.00 | 306,167.50 | 83,716.42 |
| KR | 4,600 | 340,709.60 | 67,837.99 | 2,635,730.00 | 418,200.00 | 128,945.50 |
| MX | 5,992 | 345,582.90 | 60,535.66 | 1,530,420.00 | 397,957.50 | 75,325.42 |
| FR | 6,404 | 489,113.90 | 98,836.11 | 1,986,393.00 | 556,188.10 | 93,386.89 |
| DE | 6,456 | 656,248.30 | 129,711.40 | 2,451,821.00 | 763,073.80 | 150,183.50 |
| IN | 3,951 | 870,338.10 | 181,844.30 | 2,577,537.00 | 1,102,493.00 | 390,600.90 |
| CA | 6,445 | 983,272.60 | 217,809.20 | 3,309,878.00 | 1,229,512.00 | 374,296.30 |
| US | 4,722 | 3,475,482.00 | 1,069,206.00 | 8,776,929.00 | 4,128,928.00 | 794,714.50 |
| GB | 3,065 | 3,975,862.00 | 1,126,965.00 | 10,002,810.00 | 4,305,573.00 | 1,730,385.00 |

These are descriptive differences; region-level patterns do not imply that region causes model performance. Category estimates also depend on sample size. High-activity categories such as Music show high absolute error, consistent with difficulty capturing the large-view tail.

## 6. Time-period analysis

| Split | Median views | Mean views | P95 views | P99 views | Max views |
|---|---:|---:|---:|---:|---:|
| Train | 147,209 | 968,865 | 3,077,741 | 14,589,405 | 328,860,380 |
| Validation | 130,944 | 612,061 | 2,408,958 | 8,775,507 | 49,603,959 |
| Test | 229,235 | 1,188,605 | 4,800,135 | 16,281,629 | 123,010,920 |

The later test period has a heavier upper tail and higher mean views than the earlier splits, contributing to higher test error.

## 7. Systematic prediction bias

For the test set, by view bucket:

- <10K: mean actual = 5,654.79; mean predicted = 44,178.35; mean error = +38,523.56
- 10K-100K: mean actual = 50,149.52; mean predicted = 73,598.17; mean error = +23,448.65
- 100K-1M: mean actual = 368,783.20; mean predicted = 232,415.40; mean error = -136,367.80
- 1M-10M: mean actual = 2,831,994.00; mean predicted = 737,602.20; mean error = -2,094,391.80
- >10M: mean actual = 21,918,290.00; mean predicted = 2,349,445.00; mean error = -19,568,845.00

The model overpredicts low-view videos, underpredicts large-view videos, and compresses predictions toward the center of the target distribution.

## 8. Feature importance

RandomForest impurity feature importance is descriptive, not causal. The top 20 transformed features are in `reports/baseline_feature_importance.csv`, with a plot at `reports/figures/baseline_feature_importance.png`.

The top 20 transformed features are description_length, description_word_count, tag_count, title_length, publish_hour, region_GB, title_word_count, region_CA, publish_day_of_week, region_IN, region_US, publish_month, category_name_Music, publish_year, category_name_Entertainment, region_KR, category_name_Comedy, category_name_News & Politics, category_name_Sports, and region_DE. These importances do not establish cause and effect.

## 9. Actual vs. predicted observations

Plots under `reports/figures/` show actual vs. predicted observations on original and log scales, log residual distributions, MAE by actual-view bucket, and predicted vs. actual views over the chronological test period. Together they show prediction compression and underestimation of extremely high-view observations.

## 10. Main limitations

- The test period has a heavier upper tail than the earlier periods.
- High-view videos dominate error, and the model systematically underpredicts them.
- The target is a trending-snapshot view count, not a future forecast.
- Region and category distributions differ across splits.
- The current results support a baseline diagnostic, not a claim of accurate prediction over the full distribution.

## 11. Most informative next experiment

A targeted diagnostic on the high-view tail is the most informative next step: evaluate the same model on test videos above a threshold such as 10M views and report their metrics separately. Keep the diagnostic within the existing baseline scope before considering other model iterations.

## 12. Diagnostic conclusion

The split is leakage-safe and chronologically ordered. The RandomForest baseline beats the dummy baseline but remains weak on the test split, especially for large-view videos and the later, higher-variance period. Its central limitation is compression of the view-count distribution: it overpredicts low-view observations and underpredicts the very high-view tail.

# Group A vs Group B Feature Experiment

## Purpose and setup

This controlled experiment tests whether the snapshot-context field `days_to_trending` improves estimation of **observed views in a trending snapshot**. It is not a future-view forecast. Group A reproduces the selected baseline feature set exactly: `region`, `category_name`, title and description length/word counts, `description_missing`, `tag_count`, and the four publish-time calendar fields. These are metadata-like snapshot features. Group B contains all Group A fields plus only `days_to_trending`.

Both groups use the selected `RandomForestRegressor` (200 trees, `random_state=42`, `n_jobs=-1`), the same training-only median imputation and scaling for numeric fields, the same constant `Unknown` imputation and one-hot encoding for categorical fields, and `log_observed_views` with the baseline's clipped `expm1` inverse transform. The existing chronological train/validation/test CSV assignments were reused unchanged. Group A validation and test MAE/RMSE reproduce the saved baseline. No baseline artifact was overwritten.

`days_to_trending` is the UTC calendar-day difference from publication to the dated trending snapshot. It is snapshot-context information: it would not be available before publication, so Group B must not be presented as a pre-publication predictor. Views, engagement outcomes, identifiers, timestamps, thumbnail URL, and channel title are not model predictors.

## Data quality and split checks

The full modeling dataset has 360,155 rows and no missing `days_to_trending` values. `reports/days_to_trending_summary.csv` records counts, missing values, extrema, mean, median, sample standard deviation, requested percentiles, and threshold counts for the full dataset and each split.

| Dataset | Count | Missing | Min | Max | Mean | Median | Std dev | P01 | P05 | P25 | P75 | P95 | P99 | Negative | Zero | >30 days | >365 days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 360,155 | 0 | 0 | 4,215 | 7.751 | 1 | 99.455 | 0 | 0 | 1 | 3 | 14 | 30 | 0 | 18,114 | 3,448 | 917 |
| Train | 237,400 | 0 | 0 | 4,215 | 9.631 | 1 | 122.368 | 0 | 1 | 1 | 3 | 11 | 28 | 0 | 11,832 | 1,950 | 917 |
| Validation | 50,981 | 0 | 0 | 37 | 2.533 | 1 | 4.218 | 0 | 0 | 1 | 2 | 11 | 24 | 0 | 3,118 | 187 | 0 |
| Test | 51,003 | 0 | 0 | 37 | 2.810 | 1 | 4.418 | 0 | 0 | 1 | 2 | 13 | 24 | 0 | 2,984 | 74 | 0 |

There are no negative values to investigate; zero is possible when upload and trending dates fall on the same UTC calendar date. Because `trending_date` has day precision and `publish_time` has time precision, day-level computation is appropriate. The maximum, 4,215 days, belongs to video `MJO3FmmFuh4` (published 2006-07-23, trending 2018-02-05). No suspicious values were removed or transformed. The unchanged split satisfies both strict date-ordering checks and has zero video ID overlap across split pairs.

## Validation and test comparison

Values below are Group A / Group B. Percentage change is `(Group B - Group A) / Group A`; negative values mean lower error for Group B. Full train/validation/test metrics on both scales are in `reports/group_a_vs_group_b_results.csv`.

| Scale and metric | Validation A | Validation B | Change | Test A | Test B | Change |
|---|---:|---:|---:|---:|---:|---:|
| Views MAE | 500,163.33 | 466,058.40 | -6.82% | 1,013,177.49 | 973,393.95 | -3.93% |
| Views RMSE | 1,818,047.28 | 1,788,497.47 | -1.63% | 4,183,634.21 | 4,152,636.25 | -0.74% |
| Views R² | 0.09525 | 0.12442 | — | 0.02047 | 0.03493 | — |
| Views median AE | 73,600.01 | 70,771.78 | — | 128,854.46 | 125,523.20 | — |
| Log MAE | 1.09757 | 1.00012 | — | 1.16859 | 1.09668 | — |
| Log RMSE | 1.41865 | 1.31802 | — | 1.53648 | 1.45096 | — |
| Log R² | 0.41824 | 0.49785 | — | 0.16203 | 0.25272 | — |
| Log median AE | 0.88448 | 0.77852 | — | 0.90010 | 0.83911 | — |

The test MAE reduction is 39,783.54 views and the RMSE reduction is 30,997.96 views. These are modest aggregate changes on one fixed split.

## Test view buckets and regions

| Actual-view bucket | Rows | Mean actual | Mean predicted A | Mean predicted B | MAE A | MAE B | RMSE A | RMSE B | Mean signed error A | Mean signed error B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <10K | 529 | 5,655 | 44,178 | 37,717 | 38,541 | 32,184 | 70,287 | 58,764 | 38,524 | 32,062 |
| 10K-100K | 15,761 | 50,150 | 73,598 | 76,952 | 44,685 | 48,542 | 97,784 | 106,200 | 23,449 | 26,802 |
| 100K-1M | 24,005 | 368,783 | 232,415 | 242,803 | 264,745 | 234,077 | 511,724 | 360,407 | -136,368 | -125,980 |
| 1M-10M | 9,626 | 2,831,994 | 737,602 | 719,262 | 2,375,007 | 2,240,728 | 3,237,447 | 2,969,888 | -2,094,391 | -2,112,732 |
| >10M | 1,082 | 21,918,293 | 2,349,445 | 1,942,775 | 20,086,313 | 20,032,943 | 26,941,524 | 27,043,373 | -19,568,848 | -19,975,518 |

Bucket results are in `reports/group_a_vs_group_b_bucket_metrics.csv`; the comparison figure is `reports/figures/group_a_vs_group_b_bucket_mae.png`. In the >10M bucket, MAE decreases by only about 53,370 views (0.27%), RMSE worsens by about 101,849 views (0.38%), and mean underprediction increases. Group B does not meaningfully fix the high-view tail. Test MAE by region is in `reports/group_a_vs_group_b_region_comparison.csv`; regional differences are descriptive and do not establish causation.

## Relationship and feature importance

Across the full modeling dataset, Pearson correlation between `days_to_trending` and `log_observed_views` is -0.0020; Spearman correlation is 0.5320. The scatter plot is `reports/figures/days_to_trending_vs_log_views.png`; the distribution is `reports/figures/days_to_trending_distribution.png`. These associations do not imply causality.

Using the same RandomForest impurity-based method as the baseline diagnostic, `days_to_trending` ranks first in Group B at 0.2676. The table and plot are `reports/group_b_feature_importance.csv` and `reports/figures/group_b_feature_importance.png`. Impurity importance is descriptive and can be biased; rank does not imply causality or prove generalization.

## Interpretation and limitations

Adding snapshot age modestly improves aggregate validation and test errors on this fixed split, but the test gains are small and the >10M tail remains severely underpredicted. The feature is snapshot context and is unavailable before publication. These results concern observed views at trending snapshots, not future views. Temporal distribution shift, heavy tails, and the single held-out period limit conclusions.

# Environment Recovery and Runtime Verification

## Current Environment Status

- Current environment: .venv (.venv\Scripts\python.exe), Python 3.11.9.
- The current .venv is based on .python311, as recorded in .venv/pyvenv.cfg.
- .venv_broken and .venv_broken_2 were removed during final storage cleanup.
- .python311 remains because the current .venv depends on it.

## Historical Recovery Record (2026-10-03)

At the time of the recovery, the previously unusable `.venv` was temporarily preserved as `.venv_broken`; a fresh `.venv` was created on 2026-10-03 using native 64-bit Windows CPython 3.11.9. The native interpreter used was `C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\python3.11.exe`; the new venv executable is `.venv\Scripts\python.exe`.

Verified package versions: pandas 3.0.6, NumPy 2.4.6, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0, and Streamlit 1.65.0; pip is 24.0. `pip check` reports no broken requirements. NumPy's compiled extension imports and `array([1, 2, 3]) * 2` returns `[2, 4, 6]`. Required imports, including `RandomForestRegressor`, `ColumnTransformer`, and `Pipeline`, succeed. `src/inspect_dataset.py` completed successfully.

The existing Group A and Group B experiment evaluator was rerun in memory on the existing split, without writing model or result artifacts. All 24 metrics per group matched the saved CSV within floating-point tolerance. Group A test MAE/log MAE reproduced as 1,013,177.493884 / 1.168591089; Group B reproduced as 973,393.949578 / 1.096679302.

Both saved model artifacts loaded successfully with `joblib.load(..., mmap_mode="r")`. Each is a scikit-learn `Pipeline` with preprocessing and a 200-tree `RandomForestRegressor` (`random_state=42`, `n_jobs=-1`). The Group A feature columns and Group B's additional `days_to_trending` column match their metadata. Group A metadata uses `log_views`, the baseline script's alias for the actual `log_observed_views` column. Existing artifacts were not overwritten. The experiment methodology and conclusions are unchanged.

# Random Forest vs HistGradientBoosting Experiment

## Purpose and controlled setup

This experiment checks whether changing the model family reduces the Group B RandomForest's regression-to-the-middle behavior. This experiment changes the model family while keeping the feature set, target, split, and preprocessing methodology controlled as closely as practical. It estimates observed views in trending-video snapshots and does not forecast future views.

The existing modeling dataset and chronological split were reused without changes: train 2017-11-14–2018-04-02 (237,400 rows), validation 2018-04-03–2018-05-12 (50,981 rows), and test 2018-05-13–2018-06-14 (51,003 rows). The experiment verified strict date ordering and zero `video_id` overlap. Both models use `log_observed_views`, the exact 13 Group B features (Group A plus only `days_to_trending`), training-only median imputation and scaling for numeric fields, and training-only constant `Unknown` imputation plus one-hot encoding for categorical fields. The inverse transform matches the Group B RandomForest experiment. No excluded outcome or post-outcome fields were introduced.

The reference RandomForest artifact was loaded read-only and its saved Group B test metrics were checked before comparison; RandomForest was not retrained. HistGradientBoosting used `learning_rate=0.05`, `max_iter=300`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, and `random_state=42`. Internal early stopping was disabled to avoid introducing a random internal validation split. The existing one-hot preprocessing produces dense output for compatibility with HistGradientBoosting; the feature meanings and training-only fit boundary remain unchanged.

## Results

The complete comparison table (train, validation, test; original-view and log scales; MAE, RMSE, R², and median absolute error) is saved in `reports/random_forest_vs_histgradientboosting_results.csv`.

| Model | Validation MAE (views) | Test MAE (views) | Test RMSE (views) | Test R² | Test log MAE | Test log R² |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest Group B | 466,058.40 | 973,393.95 | 4,152,636.25 | 0.034929 | 1.096679 | 0.252718 |
| HistGradientBoosting Group B | 465,575.38 | 974,215.22 | 4,121,922.12 | 0.049153 | 1.106291 | 0.286436 |

Relative to RandomForest, HistGradientBoosting test MAE is 821.27 views higher (0.0844% worse), test RMSE is 30,714.13 lower (0.7393% better), test R² is higher by 0.014223, and test log MAE is 0.009612 higher (0.8763% worse). Validation original-scale MAE is 483.03 views lower, while validation log MAE is higher (1.074397 vs 1.000123). The metrics are mixed and do not establish a general improvement.

## Tail, compression, and region diagnostics

Test actual-view buckets and model comparisons are in `reports/random_forest_vs_histgradientboosting_bucket_metrics.csv`; predicted/actual summaries and ratios are in `reports/random_forest_vs_histgradientboosting_compression.csv`. In the >10M bucket (1,082 rows), RandomForest MAE is 20,032,942.90 and HistGradientBoosting MAE is 20,032,251.21, a decrease of only 691.69 views (about 0.0035%). HistGradientBoosting RMSE is modestly lower there, but its mean prediction is lower (1,888,482 vs 1,942,775) and its mean signed error indicates greater average underprediction (-20,029,810 vs -19,975,518). The severe high-view-tail underprediction therefore remains.

Across all test rows, mean predicted/actual views is 0.2594 for HistGradientBoosting and 0.2654 for RandomForest; median predicted views are 121,658 and 125,042 respectively, against a shared actual median of 229,235. Overall prediction compression did not improve. The >10M bucket mean predicted/actual ratio is 0.0862 for HistGradientBoosting and 0.0886 for RandomForest. Test MAE by region is saved in `reports/random_forest_vs_histgradientboosting_region_mae.csv`; HistGradientBoosting has lower MAE only in GB among the listed regions, and these descriptive differences are not causal evidence.

Validation-to-test original-scale MAE ratios are approximately 2.092 for HistGradientBoosting and 2.089 for RandomForest, so this comparison does not show better validation-to-test MAE generalization. HistGradientBoosting should not replace the existing Group B RandomForest as the next-stage candidate based on this experiment: it has slightly worse test MAE and log MAE, only a negligible >10M MAE change, and more compressed overall predictions, despite lower test RMSE and higher R². No model is described as accurate on the strength of these mixed metrics.

## Artifact and limitations

The completed HistGradientBoosting pipeline and its configuration/split/version metadata are saved at `models/group_b_histgradientboosting_pipeline.joblib` and `models/group_b_histgradientboosting_metadata.json`. Six requested diagnostic plots are in `reports/figures/` with `histgb_` and `rf_vs_histgb_` prefixes; existing RandomForest figures were not overwritten. Bucket and region analyses are descriptive. This is one chronological held-out period with a heavy upper tail and does not establish performance on future YouTube views or other periods.

# Distribution-Aware Error Analysis

## Method

Overall MAE averages error over the test rows, but the target distribution is strongly right-skewed: view counts and individual errors vary by orders of magnitude. MAE alone therefore hides where error occurs. This analysis loads the saved Group B RandomForest and HistGradientBoosting artifacts and predicts on the existing test split; it does not fit or overwrite a model. Reproduced overall test MAE matched the saved results. The chronological split and Group B feature set were retained.

Disjoint ranges are `<=100K`, `>100K to <=1M`, `>1M to <=10M`, and `>10M`. The cumulative `<=1M`, `<=10M`, and `>10M` subsets are also reported. MAPE is calculated only for positive actual views and is supplementary: it can become unstable for very small actual values, so it is not the primary metric. Log MAE and median absolute log error use the difference between `log1p(predicted views)` and `log1p(actual views)`.

## RandomForest test results

The full set of requested metrics is in `reports/distribution_aware_error_analysis.csv`; both models and all subsets are in `reports/distribution_aware_model_comparison.csv`.

| Actual-view subset | Rows (% of test) | MAE (views) | MedianAE (views) | Log MAE | Mean signed error (views) | Mean predicted / actual | Share of total absolute error |
|---|---:|---:|---:|---:|---:|---:|---:|
| All test observations | 51,003 (100.00%) | 973,394 | 125,523 | 1.0967 | -873,192 | 0.2654 | 100.00% |
| `<=100K` | 16,290 (31.94%) | 48,011 | 26,697 | 0.7620 | +26,973 | 1.5538 | 1.58% |
| `>100K to <=1M` | 24,005 (47.07%) | 234,077 | 154,414 | 0.9734 | -125,980 | 0.6584 | 11.32% |
| `>1M to <=10M` | 9,626 (18.87%) | 2,240,728 | 1,550,594 | 1.7616 | -2,112,732 | 0.2540 | 43.45% |
| `>10M` | 1,082 (2.12%) | 20,032,943 | 14,050,328 | 2.9543 | -19,975,518 | 0.0886 | 43.66% |
| Cumulative `<=1M` | 40,295 (79.01%) | 158,856 | 74,976 | 0.8879 | -64,146 | 0.7320 | 12.89% |
| Cumulative `<=10M` | 49,921 (97.88%) | 560,293 | 119,215 | 1.0564 | -459,164 | 0.3789 | 56.34% |

The `>10M` group is 2.12% of test observations but accounts for 43.66% of the RandomForest's total absolute error. Together, observations above 1M are about 20.99% of test rows and account for about 87.11% of total absolute error. The model overpredicts on average in the `<=100K` group and underpredicts on average in each larger disjoint range. In the `>10M` group, mean prediction is only 8.86% of mean actual views.

For the cumulative `<=1M` subset, MAE is 158,856 and log MAE is 0.8879, versus overall MAE 973,394 and log MAE 1.0967. Errors are substantially lower for these ordinary-scale observations than for the extreme tail, but this does not by itself establish that predictions are sufficiently useful for a particular decision. MAPE and median absolute percentage error are secondary descriptions; percentage errors at small actual counts can be volatile.

## Model comparison and interpretation

HistGradientBoosting and RandomForest behave similarly in most ranges. HistGradientBoosting has slightly lower MAE in `>1M to <=10M` and `>10M`, but the >10M difference is negligible (20,032,251 vs 20,032,943). Its >10M mean predicted/actual ratio is 0.0862, also indicating severe underprediction. HistGradientBoosting is slightly worse by MAE in the lower three disjoint ranges. These results do not identify a materially different distributional behavior that resolves the tail issue.

For this dataset, feature set, observed-view target, and chronological test period, the current models estimate ordinary-scale observations with much lower errors than the extreme tail, while a small fraction of tail observations account for a large share of total absolute error. This demonstrates a limitation of the evaluated setup on this test period; it does not prove that viral-video performance is impossible in general. The figures `reports/figures/error_by_view_range_rf.png`, `error_contribution_by_view_range_rf.png`, `rf_vs_histgb_error_by_view_range.png`, and `actual_vs_predicted_rf_log_scale.png` visualize the analysis. No model was trained in this analysis.

# Random Forest vs HistGradientBoosting vs XGBoost

## Purpose and controlled setup

This single controlled XGBoost baseline asks whether a third model family changes Group B observed-view estimation, especially the high-view underprediction. This is **not** a hyperparameter optimization study. It uses the same modeling data, saved chronological train/validation/test assignments, `log_observed_views` target, Group B features, leakage exclusions, and training-only preprocessing fit boundary. The split chronology and zero video-ID overlap were reverified. The saved RandomForest and HistGradientBoosting artifacts were loaded read-only, and their validation/test metrics reproduced from predictions; neither was retrained or overwritten.

XGBoost 3.2.0 was installed into `.venv` with `--no-deps`; existing package versions were not upgraded and `pip check` reported no broken requirements. The fixed configuration was `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `reg_lambda=1`, `objective=reg:squarederror`, `eval_metric=rmse`, `random_state=42`, `n_jobs=-1`, `tree_method=hist`, and `verbosity=1`. No tuning or early stopping was used.

The categorical fields were retained with training-only constant `Unknown` imputation and one-hot encoding; numeric fields use training-only median imputation and scaling. XGBoost received a sparse CSR matrix rather than a dense matrix. The transformed training data had 237,400 rows and 40 columns, occupying about 38.0 MB as CSR; the hypothetical float64 dense representation was estimated at 75,968,000 bytes (72.4 MiB). This changes the matrix representation for compatibility and memory handling, not feature semantics.

## Overall evaluation

The complete machine-readable train/validation/test comparison on original and log scales is in `reports/rf_histgb_xgb_results.csv`.

| Model | Validation MAE | Validation log MAE | Test MAE | Test RMSE | Test R² | Test log MAE | Test MedianAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| RandomForest Group B | 466,058.40 | 1.000123 | 973,393.95 | 4,152,636.25 | 0.034929 | 1.096679 | 125,523.20 |
| HistGradientBoosting Group B | 465,575.38 | 1.074397 | 974,215.22 | 4,121,922.12 | 0.049153 | 1.106291 | 132,513.89 |
| XGBoost Group B | 453,751.32 | 1.048393 | 964,651.57 | 4,093,751.92 | 0.062105 | 1.100355 | 130,516.75 |

Compared with RandomForest, XGBoost test MAE is lower by 8,742.38 views (0.90%) and RMSE is lower by 58,884.33 (1.42%); test R² is higher by 0.027176. Test log MAE is slightly higher (1.100355 vs 1.096679), and test MedianAE is higher (130,516.75 vs 125,523.20). Compared with HistGradientBoosting, XGBoost improves these aggregate test MAE, RMSE, and log MAE values, while the overall differences remain modest. It is not uniformly better across metrics.

## Tail errors and error contribution

All three models' disjoint test-range metrics are in `reports/rf_histgb_xgb_bucket_comparison.csv`; XGBoost-only rows are in `reports/xgb_bucket_metrics.csv`.

| Model | >10M rows | >10M MAE | >10M mean signed error | >10M mean predicted / actual | >1M share of total absolute error | >10M share of total absolute error |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest Group B | 1,082 | 20,032,942.90 | -19,975,517.98 | 0.08864 | 87.11% | 43.66% |
| HistGradientBoosting Group B | 1,082 | 20,032,251.20 | -20,029,810.47 | 0.08616 | 86.81% | 43.62% |
| XGBoost Group B | 1,082 | 19,789,526.77 | -19,774,746.47 | 0.09780 | 86.64% | 43.52% |

XGBoost reduces >10M MAE by about 243,416 views (1.22%) compared with RandomForest, and its mean signed error and predicted/actual ratio are modestly less negative/higher. Nonetheless, it predicts only about 9.78% of mean actual views in this bucket and its mean signed error remains about -19.77 million. The tail underprediction remains severe. XGBoost reduces the share of total absolute error from rows above 1M by about 0.47 percentage points and the >10M share by only about 0.14 percentage points compared with RandomForest; the distribution of error changes little.

## Prediction compression and validation-to-test behavior

Mean predicted/mean actual ratios by actual-view range are:

| Model | `<=100K` | `>100K to <=1M` | `>1M to <=10M` | `>10M` |
|---|---:|---:|---:|---:|
| RandomForest Group B | 1.5538 | 0.6584 | 0.2540 | 0.0886 |
| HistGradientBoosting Group B | 1.5522 | 0.6090 | 0.2582 | 0.0862 |
| XGBoost Group B | 1.4724 | 0.6197 | 0.2881 | 0.0978 |

XGBoost ratios are closer to 1 in the larger two ranges and in the >10M tail, but remain far below actual mean views for high-view videos; low-range predictions are also less overestimated than RandomForest. This is a modest shift in compression, not a solution to the high tail.

| Model | Validation MAE | Test MAE | Test / validation MAE |
|---|---:|---:|---:|
| RandomForest Group B | 466,058.40 | 973,393.95 | 2.0886 |
| HistGradientBoosting Group B | 465,575.38 | 974,215.22 | 2.0925 |
| XGBoost Group B | 453,751.32 | 964,651.57 | 2.1259 |

XGBoost has the lowest validation and test MAE in this one comparison, but its test/validation ratio is highest, so the results do not indicate better validation-to-test degradation. The test gains are modest and the test log MAE is worse than RandomForest. No model is selected using a single metric, and this experiment does not claim accurate estimates or prediction of future views: the target remains observed views in dated trending snapshots.

The saved pipeline and metadata are `models/group_b_xgboost_pipeline.joblib` and `models/group_b_xgboost_metadata.json`. The six new figures are `reports/figures/xgb_actual_vs_predicted_test.png`, `xgb_actual_vs_predicted_log_test.png`, `xgb_residual_distribution_log_test.png`, `xgb_mae_by_view_range.png`, `rf_vs_histgb_vs_xgb_bucket_mae.png`, and `rf_vs_histgb_vs_xgb_metrics.png`. This is one fixed baseline on one chronological evaluation period; it does not establish performance on other periods or future views.

# XGBoost Log Target vs Raw Target Experiment

## Purpose and controlled setup

This experiment isolates target representation: the existing Group B XGBoost trained on `log_observed_views` is compared with one XGBoost trained directly on `observed_views`. The saved log-target pipeline was used as the reference and was not retrained. Both use the same existing modeling data and chronological split, exact Group B feature list, XGBoost 3.2.0, configuration, random state, and preprocessing. Date order and zero video-ID overlap were verified. The raw-target output is direct model output, with no clipping or log transform; all raw-scale metrics use that unmodified prediction.

Both use the same fixed parameters: `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `reg_lambda=1`, `objective=reg:squarederror`, `eval_metric=rmse`, `random_state=42`, `n_jobs=-1`, `tree_method=hist`, and `verbosity=1`. The preprocessing is fit on training features only: constant `Unknown` imputation and one-hot encoding for categorical fields, median imputation and scaling for numeric fields. Both use the same sparse representation. The only intended experiment change is the training target representation.

## Overall results

All original-scale metrics across train, validation, and test, plus the log-target model's log-scale scores, are in `reports/xgb_log_vs_raw_results.csv`.

| Target representation | Validation MAE | Test MAE | Test RMSE | Test R² | Test MedianAE | Test log MAE where available |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost log target | 453,751.32 | 964,651.57 | 4,093,751.92 | 0.062105 | 130,516.76 | 1.100355 |
| XGBoost raw target | 612,549.62 | 1,060,950.31 | 4,036,236.56 | 0.088274 | 239,901.72 | Not computed |

Raw-target training raises test MAE by 96,298.75 views (9.98%) and test MedianAE by 109,384.96 (83.81%), while lowering RMSE by 57,515.36 (1.41%) and increasing R² by 0.026169. Log MAE is not reported for the raw-target model: a log score would require additional handling for its negative predictions, and raw-scale metrics are primary for this comparison. Thus the raw model trades larger typical absolute errors for a lower squared error and better R².

## Ordinary-vs-tail tradeoff

The disjoint test buckets are in `reports/xgb_log_vs_raw_bucket_metrics.csv`; cumulative `<=1M`, `>1M`, and `>10M` results are in `reports/xgb_log_vs_raw_tradeoff.csv`.

| Test subset | Log-target MAE | Raw-target MAE | Log-target mean signed error | Raw-target mean signed error | Log-target predicted / actual | Raw-target predicted / actual |
|---|---:|---:|---:|---:|---:|---:|
| `<=1M` cumulative | 163,164 | 351,734 | -74,257 | +157,986 | 0.6898 | 1.6600 |
| `>1M` cumulative | 3,980,710 | 3,729,784 | -3,810,423 | -2,411,267 | 0.1996 | 0.4935 |
| `>10M` | 19,789,527 | 17,213,439 | -19,774,746 | -16,110,995 | 0.0978 | 0.2650 |

Raw-target training improves the >10M MAE by about 2.58 million views (13.02%) and the mean predicted/actual ratio rises from 0.0978 to 0.2650. Underprediction is reduced but remains substantial. The improvement comes with materially worse errors for the 79.01% of test observations at or below 1M: their cumulative MAE more than doubles, and the mean prediction shifts from underprediction to overprediction. In the disjoint `<=100K` range, raw-target MAE is about 222,515 versus 45,323 for the log target; in `>100K to <=1M`, it is about 439,423 versus 243,132. Whether that ordinary-range degradation is acceptable depends on the project's error priorities; it is not a free tail improvement.

## Prediction distribution and limitations

Predicted test distribution summaries are in `reports/xgb_log_vs_raw_prediction_distribution.csv`.

| Series | Mean | Median | 95th percentile | 99th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Actual test views | 1,188,605 | 229,235 | 4,800,134 | 16,281,629 | 123,010,920 |
| Log-target predictions | 329,945 | 112,637 | 1,204,702 | 3,486,237 | 16,112,656 |
| Raw-target predictions | 807,180 | 280,690 | 3,170,815 | 10,233,241 | 91,228,544 |

Raw-target predictions have a wider range and are closer to the actual upper quantiles, but the raw model's minimum prediction is -8,828,256 views. Predictions were not clipped because that would add an output transformation and violate the target-only comparison; the negative value is included in raw-scale metrics. In the log-axis scatter and log1p distribution plot, nonpositive predictions are floored at zero (or shown at 1 on log axes) only for visualization. The log-target model retains the established nonnegative inverse transform.

The test MAE / validation MAE ratio is 2.126 for the log target and 1.732 for the raw target. Although the raw model has a smaller ratio, its absolute validation and test MAEs are both higher, so this ratio alone does not make it preferable. On this fixed dataset, feature set, and evaluation period, log-target training remains more appropriate when ordinary observations and overall MAE matter: it has lower test MAE and MedianAE and much lower error through 1M views. Raw-target training is a meaningful alternative only if the project explicitly gives more weight to upper-tail raw-scale errors and accepts the ordinary-range degradation and occasional negative outputs. Neither target representation solves >10M underprediction, and this experiment does not establish future-view prediction performance.

The separate raw-target artifact and metadata are `models/group_b_xgboost_raw_target_pipeline.joblib` and `models/group_b_xgboost_raw_target_metadata.json`; metadata records `target=observed_views` and `target_transform=none`. Four new figures are `reports/figures/xgb_log_vs_raw_actual_predicted.png`, `xgb_log_vs_raw_bucket_mae.png`, `xgb_log_vs_raw_prediction_distribution.png`, and `xgb_log_vs_raw_bias.png`. This is one controlled target-representation experiment, not hyperparameter tuning.

# Final XGBoost Group A vs Group B Ablation

## Purpose and controlled setup

This final feature ablation asks whether adding `days_to_trending` provides incremental value to the selected XGBoost log-target setup. Model A uses exactly Group A's metadata-style fields. Model B uses those same fields plus only `days_to_trending`. Both target `log_observed_views`; the saved Group B XGBoost model is the reference and was not retrained. Group A was trained once. Both used XGBoost 3.2.0, the same fixed XGBoost configuration, train-only imputation/scaling and sparse one-hot preprocessing, and the unchanged chronological train/validation/test assignment. The date ordering and zero `video_id` overlap were rechecked. No forbidden outcome, engagement, identifier, or timestamp features were added.

The complete original-scale and log-scale metrics, with Group B minus Group A absolute and percentage changes, are in `reports/xgb_group_a_vs_group_b_results.csv`. Negative changes in error metrics represent improvements.

| Test metric | Group A | Group B | Group B minus Group A | Percent change |
|---|---:|---:|---:|---:|
| Views MAE | 1,036,434.86 | 964,651.57 | -71,783.29 | -6.93% |
| Views RMSE | 4,231,182.95 | 4,093,751.92 | -137,431.03 | -3.25% |
| Views R² | -0.001924 | 0.062105 | +0.064029 | Relative percentage is unstable because Group A is near zero |
| Views MedianAE | 136,782.76 | 130,516.76 | -6,266.00 | -4.58% |
| Log MAE | 1.284790 | 1.100355 | -0.184435 | -14.36% |
| Log RMSE | 1.640041 | 1.414122 | -0.225918 | -13.78% |
| Log R² | 0.045261 | 0.290178 | +0.244917 | Relative percentage is not emphasized |
| Log MedianAE | 1.044769 | 0.879372 | -0.165398 | -15.83% |

Group B also reduces validation views MAE from 506,136.43 to 453,751.32 (10.35%) and test views MAE by 6.93%. Its MAE advantage appears on both held-out splits, while R² and the log-scale metrics improve as well. The R² percentage change for views is not substantively interpretable because Group A's test R² is close to zero; the absolute difference is reported instead.

## Test view ranges

| Actual-view range | Rows | Group A MAE | Group B MAE | Group A log MAE | Group B log MAE | Group A mean signed error | Group B mean signed error | Group A predicted / actual | Group B predicted / actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `<=100K` | 16,290 | 38,922 | 45,323 | 0.6933 | 0.7466 | +15,603 | +23,006 | 1.3204 | 1.4724 |
| `>100K to <=1M` | 24,005 | 255,224 | 243,132 | 1.2013 | 1.0415 | -207,426 | -140,261 | 0.4375 | 0.6197 |
| `>1M to <=10M` | 9,626 | 2,435,007 | 2,203,737 | 2.2510 | 1.6779 | -2,420,871 | -2,015,971 | 0.1452 | 0.2881 |
| `>10M` | 1,082 | 20,943,825 | 19,789,527 | 3.4466 | 2.5930 | -20,943,825 | -19,774,746 | 0.0445 | 0.0978 |

Bucket RMSE, MedianAE, signed-error medians, summed absolute errors, and total-error contribution shares are in `reports/xgb_group_a_vs_group_b_bucket_metrics.csv`. The added feature worsens MAE in the `<=100K` group by about 16.45%, but improves it in the three higher ranges: approximately 4.74% for `>100K to <=1M`, 9.49% for `>1M to <=10M`, and 5.51% for `>10M`. The tail ratio more than doubles from 0.0445 to 0.0978, but Group B still substantially underpredicts the >10M range. The feature helps across most ranges, not all of them, and does not solve the extreme tail.

## Group B feature importance

The Group B XGBoost model's gain-based `feature_importances_` for one-hot categorical columns were summed back to their source feature; the resulting importances were normalized to sum to one. `days_to_trending` has source-level importance **0.141495** and ranks third, behind `region` (0.419374) and `category_name` (0.328646). The complete table is `reports/xgb_group_b_feature_importance.csv`. This is descriptive model importance, not causal evidence or proof that the feature will generalize to other periods.

## Interpretation and scope

For estimates of observed views in dated trending snapshots, Group B provides a consistent aggregate improvement on this split and helps the test ranges above 100K, so it remains the final feature set for that snapshot-estimation use when the field is available. Group A remains the metadata-only set. `days_to_trending` is snapshot-context information and is not available before publication; Group B must not be presented as a pre-publication forecasting model. The project estimates observed views at the dated trending snapshot, not future views. This is one fixed ablation on one chronological evaluation period; the bucket-specific degradation at `<=100K` and persistent >10M underprediction limit the conclusion.

The fitted Group A model and metadata are `models/group_a_xgboost_pipeline.joblib` and `models/group_a_xgboost_metadata.json`. The saved Group B model was left unchanged. New figures are `reports/figures/xgb_group_a_vs_b_mae.png`, `xgb_group_a_vs_b_bucket_mae.png`, and `xgb_group_a_vs_b_metrics.png`.
