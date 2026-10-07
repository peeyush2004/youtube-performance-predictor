# Dataset and ML Problem Design

> **Historical proposal, not the implemented model contract.** This document records an earlier pre-publication and classification design. The implemented project instead estimates `observed_views` in dated trending snapshots; see `docs/modeling_dataset.md`, `docs/model_training.md`, and `docs/final_model_contract.md`.

## Project Objective

Predict a YouTube video's performance from information available immediately before its scheduled publication. The first release is a tabular-data baseline, not a real-time analytics system. One dataset row represents one video and one prediction-time snapshot.

The prediction cutoff, denoted by $t_0$, is immediately before scheduled publication. Every input value must have been known at or before $t_0$. This gives the eventual application a clear contract: it receives pre-publication metadata and predicts an outcome that occurs later.

## Prediction Targets

The primary outcome is `views`: the cumulative view count at a fixed horizon of 30 days after publication (`views_30d`). Thirty days is a proposed operational definition, not an assumption that every source provides an exact 30-day snapshot. Before data collection, confirm that the chosen source provides comparable counts at that horizon. If it does not, choose a consistently observed horizon and update this specification; do not mix lifetime counts or observations of different ages as though they were equivalent.

The second target, `performance_category`, is derived from the view target and is used only for classification. Neither target is an input feature.

## Mathematical Problem Definition

For video $i$, let $t_{0,i}$ be its prediction cutoff and let $x_i = \phi(z_i(t_{0,i}))$ be the encoded feature vector built only from raw information $z_i$ observable by that cutoff. Let $v_i$ be its non-negative view count at the fixed 30-day horizon. Regression learns a function $f$ that estimates $\log(1 + v_i)$ from $x_i$; a view-count estimate is obtained with $\hat{v}_i = \exp(f(x_i)) - 1$, subject to the retransformation caveat below.

For classification, calculate training-only cut points $q_{1/3}$ and $q_{2/3}$ from the training fold's $v_i$ values. Define $c_i$ as Low when $v_i < q_{1/3}$, Medium when $q_{1/3} \le v_i < q_{2/3}$, and High when $v_i \ge q_{2/3}$. A classifier learns $g(x_i)$ to predict this derived category. The cut points are frozen and applied to validation/test targets; neither $v_i$ nor $c_i$ is part of $x_i$.

## Data Leakage Rules

- Build every feature as of $t_0$. Store or otherwise retain timestamps that establish when a value was observed.
- Never use the current video's post-publication views, likes, comments, watch time, or engagement as predictors.
- Treat the current video's `views` as a target only. It is necessarily measured after publication; using it as an input would leak the answer.
- Historical channel features are allowed only when computed from earlier videos whose required outcome window had finished and whose data was available by the current video's $t_0$.
- Fit imputers, encoders, scalers, feature selectors, and classification thresholds on training data only. Apply the fitted transformations unchanged to validation and test data.
- Do not use video IDs, channel IDs, URLs, or other identifiers as ordinary numeric or categorical predictors in Version 1. Retain IDs for joins, grouping, deduplication, and splitting.
- Do not calculate aggregates over the full dataset before splitting. This can transfer future outcomes into past rows.

## Proposed Dataset Schema

Types are conceptual; implementation can map them to Pandas and a persisted tabular format later. All timestamps should be timezone-aware and normalized to UTC. A row is eligible only if its prediction-time inputs and fixed-horizon target can be defined.

### A. Safe Input Features

These are usable as predictors only when the stated value is finalized or observable by $t_0$.

| Column | Type | Description | Role | Available by $t_0$? | Leakage risk | V1? |
|---|---|---|---|---|---|---|
| `category_id` | categorical/string | Video category selected before publication | Input feature | Yes, if category is set by $t_0$ | Low | Yes |
| `duration_seconds` | integer/float | Duration of the final video asset | Input feature | Yes, if the upload-ready asset is known | Low | Yes |
| `title_length_chars` | integer | Character count of the finalized title | Input feature | Yes | Low | Yes |
| `description_length_chars` | integer | Character count of the finalized description | Input feature | Yes | Low | Yes |
| `tag_count` | integer | Number of tags selected before publication | Input feature | Yes | Low | Yes |
| `is_short` | boolean | Short-form designation from an explicit platform field or a documented duration rule | Input feature | Yes, if format/asset is known | Low; rule may change over time | Yes, with rule version documented |
| `scheduled_publish_weekday` | categorical | Weekday derived from the scheduled publication time in a documented channel/audience timezone | Input feature | Yes, if schedule and timezone are known | Low | Yes, when reliable |
| `scheduled_publish_hour` | integer/categorical | Hour derived from the same scheduled time and timezone | Input feature | Yes, if schedule and timezone are known | Low | Yes, when reliable |
| `channel_age_days_at_t0` | integer | Days between channel creation and prediction cutoff | Input feature | Yes, if channel creation date is known | Low | Yes, when available |

### B. Potentially Useful; Requires Careful Handling

These may be safe under a strict timestamp/cutoff contract, but are not unconditional Version 1 features.

| Column | Type | Description | Role | Available by $t_0$? | Leakage risk | V1? |
|---|---|---|---|---|---|---|
| `channel_subscriber_count_at_t0` | integer | Subscriber count from a snapshot taken no later than $t_0$ | Input feature | Only with a dated snapshot | High if collected after publication or replaced by a current count | Conditional: include only with verified snapshot time |
| `channel_id` | string | Channel identifier | Metadata/group key | Yes | High memorization and channel overlap risk if modeled directly | No; retain for grouping and history joins |
| `video_id` | string | Video identifier | Metadata/deduplication key | Usually | High memorization risk if modeled | No |
| `scheduled_publish_at_utc` | datetime | Planned publication timestamp, retained as provenance for derived schedule features | Metadata/derivation source | Yes, if schedule is known | Low as metadata; actual time may differ from plan | Keep for audit; derive weekday/hour |
| `actual_publish_at_utc` | datetime | Time the video actually went live | Metadata | Known only at/after publication | It may not be available at a pre-publication prediction cutoff; can encode a schedule change | No; prefer the scheduled time |
| `title_text` | string/text | Finalized title content | Input feature candidate | Yes, if finalized by $t_0$ | Medium: may contain post-publication edits, URLs, or outcome-derived annotations | Version 2 |
| `description_text` | string/text | Finalized description content | Input feature candidate | Yes, if finalized by $t_0$ | Medium: may be edited later or contain outcome-derived links/text | Version 2 |
| `tag_values` | list/text | Tag content selected before publication | Input feature candidate | Yes, if finalized by $t_0$ | Medium: verify tags are a pre-publication snapshot | Version 2 |
| `channel_upload_count_90d` | integer | Number of earlier channel uploads in the 90 days before $t_0$ | Historical input feature | Yes, if computed using only earlier publication times | High if calculated with future uploads or a full-dataset group aggregate | Version 2 |
| `channel_median_views_30d_prev10` | float | Median 30-day views for up to ten eligible earlier videos from this channel | Historical input feature | Only after those earlier videos' 30-day windows mature and become observable | High if current/future videos or immature outcomes are included | Version 2 |
| `channel_median_engagement_rate_prev10` | float | Median engagement rate on up to ten eligible earlier videos, using a consistent snapshot definition | Historical input feature | Only if prior metrics were observed by $t_0$ | High if current-video likes/comments/views or future observations enter the aggregate | Version 2 |

For historical aggregates, define the lookback, eligible rows, metric formula, and observation timestamps before computing them. “Previous” means information available by the current row's $t_0$, not merely a lower row number or earlier position in a shuffled file. Use a no-history indicator when a channel has too few eligible videos; do not silently replace missing history with a value computed from future data.

### C. Post-Publication / Leakage Features

These are not predictors for the pre-publication task. Some could define a different early-performance task if the prediction cutoff were deliberately moved later, but that would require a separately specified task and evaluation.

| Column | Type | Description | Role | Available by $t_0$? | Leakage risk | V1? |
|---|---|---|---|---|---|---|
| `views_24h` | integer | Current video's views during/at 24 hours after publication | Post-publication outcome | No | Direct leakage for pre-publication views prediction | No |
| `views_7d` | integer | Current video's views during/at seven days after publication | Post-publication outcome | No | Direct leakage for pre-publication views prediction | No |
| `likes` | integer | Current video's post-publication likes | Post-publication outcome | No | Direct leakage | No |
| `comments` | integer | Current video's post-publication comments | Post-publication outcome | No | Direct leakage | No |
| `watch_time_minutes` | numeric | Current video's accumulated watch time | Post-publication outcome | No | Direct leakage | No |
| `current_video_engagement_rate` | numeric | Engagement calculated from the current video's post-publication metrics | Post-publication outcome | No | Direct leakage; may also use the target in its denominator | No |
| `final_views_lifetime` | integer | Lifetime/final view count observed at an unspecified or variable age | Post-publication outcome | No | Target leakage if used as an input; inconsistent-age label if used as the target | No; use a fixed-horizon target instead |

### D. Target Variables and Target Metadata

| Column | Type | Description | Role | Available by $t_0$? | Leakage risk | V1? |
|---|---|---|---|---|---|---|
| `views` (semantics: `views_30d`) | non-negative integer | Cumulative views at the fixed 30-day horizon | Regression target ($y$), never an input | No; observed after publication | It leaks the answer if placed in $X$; safe as the supervised target | Yes |
| `performance_category` | ordered categorical: Low/Medium/High | Category derived from `views` using thresholds learned from training data | Classification target ($c$), never an input | No; derived after the target is observed | It directly encodes the target if placed in $X$ | Yes, derived for classification experiments |
| `views_observed_at_utc` | datetime | Timestamp when the view count used for `views` was observed | Target provenance/quality metadata | No | Not a predictor; needed to verify the target horizon | Yes, as metadata only |

Keep the field name `views` if desired for the primary project target, but document and enforce its 30-day meaning. Do not mix it with lifetime views or counts collected at different video ages. The target horizon should be stored as dataset-level metadata or otherwise made explicit.

## Version 1 Feature Set

Start with a small, interpretable pre-publication feature set:

- `category_id`
- `duration_seconds` and `is_short`
- `title_length_chars`, `description_length_chars`, and `tag_count`
- `scheduled_publish_weekday` and `scheduled_publish_hour`, if schedule and timezone are known before $t_0$
- `channel_age_days_at_t0`, when available
- `channel_subscriber_count_at_t0` only when a timestamped snapshot at or before $t_0$ can be established

If a trustworthy subscriber snapshot is unavailable, omit it rather than substituting a present-day count. Do not include raw IDs or historical channel performance in the first baseline. This gives a testable model without allowing the outcome to influence feature values.

## Version 2 Feature Set

After Version 1 has a reliable data contract and temporal evaluation:

- Text representations of finalized titles, descriptions, and tags.
- Strictly historical channel statistics such as `channel_upload_count_90d`, prior-video median 30-day views, and prior engagement rates.
- Carefully derived audience-local scheduling features when the channel timezone is reliable.
- Additional pre-publication metadata such as language or thumbnail properties, only if captured before $t_0$ and justified by data availability.

Add features incrementally and compare them against the Version 1 baseline using the same temporal splits. Keep `channel_id` as a grouping key by default; direct channel-ID encoding can memorize channels and obscure generalization.

## Classification Strategy

Three plausible ways to form Low/Medium/High labels are:

| Method | Advantages | Disadvantages |
|---|---|---|
| Dataset-wide quantiles/percentiles | Data-derived; tertiles aim for reasonably balanced classes; straightforward to reproduce | Categories are relative to the particular training population, not universal view-count meanings; thresholds can shift between dataset versions or time periods |
| Channel-relative performance | Answers whether a video outperformed its own channel's baseline; reduces the advantage of large channels | Requires enough prior videos and mature historical outcomes; definitions vary with baseline/lookback and are difficult for new channels; historical-feature leakage is easy to introduce |
| Fixed view-count thresholds | Easy to explain and stable across datasets when thresholds have external meaning | Requires a defensible business/platform/domain standard; arbitrary cutoffs create misleading labels; class imbalance and time inflation can be severe |

**Recommendation for Version 1:** define Low/Medium/High using the 33rd and 67th percentiles of `views` in the training portion only. Fit these two cut points independently inside each training fold, freeze them, and apply them unchanged to that fold's validation/test rows. Use deterministic boundary rules, for example Low below the first cut, Medium from the first cut up to (but not including) the second, and High at or above the second. Since `log1p` is strictly increasing, the percentile ordering is the same on raw views and log views; ties can still collapse or imbalance bins. Report cut points and class counts. If ties make three meaningful bins impossible, document the issue and reconsider the label definition without inspecting final-test labels to tune it.

These labels mean relative popularity in the training population, not a universal definition of success. Version 2 can investigate channel-relative categories, for example by comparing 30-day views with a prior-channel baseline, but must compute that baseline strictly as of $t_0$ and define cold-start behavior. Fixed thresholds should be used only if a defensible external standard is agreed upon.

## Regression Target Strategy

View counts are non-negative and typically highly right-skewed: a small number of videos may have far more views than most. A raw-view loss can be dominated by those large counts. For the first regression comparison, use `log1p(views)` as the model target and apply `expm1` to predictions for view-count reporting. Compare with a raw-target baseline and evaluate on the original view-count scale as well as a log-sensitive scale.

An inverse-transformed log prediction is not automatically an unbiased estimate of the arithmetic mean view count; it is closer to a typical/central prediction on the log scale. If “expected views” specifically means the conditional arithmetic mean, assess a training-only retransformation correction (such as a smearing estimate) and calibrate it on validation data. State clearly whether reported predictions are typical views or estimated mean views. Keep zero-view labels valid; `log1p` handles zero.

## Feature Encoding Strategy

- One-hot encode low-cardinality fields such as `category_id` and weekday, with an explicit unknown-category behavior for categories not seen during training.
- Encode hour cyclically using sine/cosine terms, or one-hot encode it if the data volume is adequate. Do not treat hour as a simple linear quantity where 23 is far from 0.
- Keep booleans as boolean or 0/1 values, and scale numeric features for scale-sensitive models such as linear/logistic regression.
- Use a single fitted preprocessing pipeline per training fold. Do not fit encoders or scalers using validation/test rows.
- Avoid target encoding in Version 1. If considered later, compute it out-of-fold and within time boundaries.
- Text fields, if added in Version 2, need a separately validated text representation that uses only finalized pre-publication content.

## Missing-Value Strategy

- First record why values are missing and whether missingness itself has operational meaning.
- Impute numeric predictors with a training-fold median and, where useful, add a missingness indicator.
- Impute categorical predictors with an explicit `Unknown`/`Missing` category or a training-fold most-frequent value.
- Do not impute missing targets. Exclude rows without a valid, consistently defined target and report the exclusion count/reason.
- For historical features, preserve a no-history indicator and use only history available by $t_0$. Avoid treating “new channel” as ordinary random missingness.
- Fit all imputers on the training portion only. Do not fill missing subscriber counts from a later snapshot.

## Train/Test Split Strategy

Use a **time-based split as the primary evaluation**, not a random split. Sort by scheduled/publication time and use older videos for training, a subsequent period for validation, and the most recent period for the final test. Exact date boundaries and proportions should be chosen after the dataset size and coverage are known; the final test period must remain untouched until model selection is complete.

The 30-day target creates an observation/maturity constraint. At a simulated training cutoff, include only training labels whose 30-day observation was available by that cutoff. Purge rows whose target-observation window crosses a split boundary when that would give the model information unavailable at the simulated fit time. Do not let future labels feed feature aggregates.

Videos from the same channel may legitimately occur in both earlier training and later test periods if the intended deployment predicts future uploads from known channels. For each later row, historical features may use only earlier channel videos whose outcomes had matured by that row's $t_0$. Keep channel IDs out of the ordinary feature matrix and report performance by channel or use channel-clustered uncertainty estimates where feasible. If deployment must work for never-before-seen channels, add a separate group-held-out-channel evaluation; it answers a different, harder generalization question.

## Evaluation Metrics

### Regression

- **MAE in views** as the primary, directly interpretable error measure.
- **RMSLE** or RMSE on `log1p(views)` as a complementary measure that is less dominated by the largest channels/videos and reflects multiplicative error.
- **Median absolute error** as a robust summary; raw-scale RMSE and $R^2$ may be reported as secondary diagnostics, with the skew caveat.
- Compare against simple training-only dummy baselines. Avoid MAPE as a primary metric because zero and near-zero targets make it unstable.

### Classification

- **Macro-F1** and **balanced accuracy** to give each class weight despite possible imbalance.
- Per-class precision, recall, and F1, plus a confusion matrix.
- Optionally report one-vs-rest ROC-AUC when class probabilities and enough test examples per class are available; it should not replace threshold-based metrics.
- Because Low/Medium/High are ordered, also inspect the severity of errors (for example, Low predicted High) rather than treating every wrong class as equally close.

## Candidate ML Algorithms (Later)

No algorithms are implemented or trained in this design phase. When data is approved and available, compare:

- Regression baselines: `DummyRegressor`, then Linear/Ridge Regression on `log1p(views)`.
- Nonlinear tabular regression: `RandomForestRegressor` and `HistGradientBoostingRegressor`.
- Classification baseline: `DummyClassifier`, then `LogisticRegression`.
- Nonlinear tabular classification: `RandomForestClassifier` and `HistGradientBoostingClassifier`.
- Consider XGBoost only later if dataset size/quality and validation results justify the extra dependency and tuning cost.

Use the same temporal folds, feature contract, and appropriate preprocessing for fair comparisons. Establish simple baselines before tuning more complex models.

## Final Recommended Approach

1. Treat prediction as a pre-publication task with an explicit $t_0$ cutoff.
2. Define the primary regression target as cumulative `views` at a single fixed horizon, initially proposed as 30 days (`views_30d`).
3. Start with the Version 1 metadata features listed above; include subscriber count only from a verified pre-cutoff snapshot. Keep current-video engagement and raw IDs out of the feature matrix.
4. Model `log1p(views)` initially, but evaluate and explain errors in original view units; apply a training-only correction if the intended output is the arithmetic expected value.
5. For an initial Low/Medium/High target, derive tertile cut points from training targets only and freeze them for validation/test. Treat labels as population-relative, not universal.
6. Use chronological train/validation/test partitions with 30-day label-maturity controls. Compute all historical features row-by-row as of each prediction cutoff.
7. Compare simple dummy/linear baselines with random forest and histogram gradient boosting before considering XGBoost or text/history features.

This is a design specification only. It does not collect data, define a data source, create synthetic rows, implement preprocessing/model code, train a model, or build an application.