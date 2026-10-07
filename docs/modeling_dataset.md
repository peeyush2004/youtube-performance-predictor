# Modeling Dataset

## 1. Objective and Target

The approved task is snapshot-level estimation for DataSnaek's historical trending-video records. It is not a future view prediction, 30-day forecast, or pre-publication model.

The raw target field is `views`. The processed dataset names the target `observed_views`:

> `observed_views = views`, the count recorded in that regional trending snapshot.

`log_observed_views` is `log1p(observed_views)`. Both targets are retained; outliers are not removed.

## 2. Raw Data and Observation Unit

The input is the unmodified DataSnaek Kaggle Version 115 copy in `data/raw/`: ten regional CSVs and ten category JSON maps. The CSV row unit is one video-region-trending-date source record. Videos legitimately recur on different dates and in different regions; these are retained. Exact duplicate rows and conflicting records for the same video/date/region are handled separately below.

The CSV schema has 16 common fields: `video_id`, `trending_date`, `title`, `channel_title`, `category_id`, `publish_time`, `tags`, `views`, `likes`, `dislikes`, `comment_count`, `thumbnail_link`, `comments_disabled`, `ratings_disabled`, `video_error_or_removed`, and `description`. There is no `channelId` or duration field. The target observation date is a date token (`YY.DD.MM`), not an exact request timestamp.

## 3. Regional Merge

`src/build_modeling_dataset.py` reads the ten `*videos.csv` files in filename order and derives `region` strictly from the two-letter filename prefix (for example, `USvideos.csv` becomes `US`). It does not infer region from video metadata. Regional files are concatenated only after per-file exact-duplicate analysis; regional observations remain distinct in the snapshot key.

## 4. Duplicate and Conflict Handling

The deterministic rule is applied within each source region:

1. Remove exact full-row duplicates, keeping the first source-order occurrence. This removed **12,570** rows.
2. Count repeated `(video_id, trending_date)` keys before exact-deduplication. There were **13,355** repeated key groups across the regions. The region is implicit at this stage because each group is processed within one filename.
3. After exact duplicates are removed, any key with more than one remaining full source row is a conflict. The comparison uses all original CSV columns, not just the views value.
4. Exclude every remaining row belonging to a conflicting `(video_id, trending_date, region)` key. Do not aggregate view counts or select a “best” row. This excluded **1,269 keys** and **3,217 rows after exact-duplicate removal**.
5. Repeated keys whose contents were identical are resolved by step 1 and retain one deterministic first occurrence. Videos on different dates and the same video in different regions remain separate.

The result has one row per `(video_id, region, trending_date)` key. This is a dataset-quality resolution rule, not a claim that videos themselves are unique observations.

## 5. Leakage Exclusions

The following raw fields are not model predictors and are omitted from the processed feature columns:

| Field | Reason |
|---|---|
| `views` | Used only to form the `observed_views` target; direct target leakage if used as an input |
| `likes`, `dislikes`, `comment_count` | Same-snapshot post-publication engagement metrics that accumulated alongside views; no earlier prediction timestamp is available |
| `thumbnail_link` | Mutable snapshot URL/content reference; not included in Version 1 |
| `title`, `description`, `tags` | Raw text is not included; only simple length/word/tag-count summaries are derived. These are snapshot-time values and may have been edited after publication, so they are not pre-publication features |
| `video_id`, `trending_date`, `publish_time` | Retained for traceability and time/group evaluation, not direct model inputs |
| `channel_title` | No stable channel ID exists; high-cardinality display name is excluded as a predictor |

`comments_disabled`, `ratings_disabled`, and `video_error_or_removed` are retained in the output for auditing, but are **audit-only** and not part of Group A or Group B predictors. They describe snapshot-time status and may be post-publication or selection-related.

## 6. Category Mapping

Each regional JSON file maps its `items[].id` category ID to `items[].snippet.title`. The mapping is applied within the matching filename region. `category_id` is preserved in the processed file; the derived `category_name` is used as the categorical feature. An unmapped ID produces a missing `category_name`, not an invented label. The final file has **357,470 mapped names out of 360,155 rows**; **2,685** category names remain missing.

## 7. Feature Engineering

| Feature | Exact rule |
|---|---|
| `title_length` | Number of decoded Unicode code points in `title` (`len`); missing title remains missing |
| `title_word_count` | Whitespace split count for `title`; this is a simple count, not language-aware NLP |
| `description_length` | Number of decoded Unicode code points; missing description remains missing, while an empty string has length zero |
| `description_word_count` | Whitespace split count; missing description remains missing, while an empty string has count zero |
| `description_missing` | Boolean flag captured before length/count derivation, distinguishing a missing description from an empty one |
| `tag_count` | Split `tags` on the source pipe delimiter (`|`), trim whitespace, and count non-empty tokens. Empty tags and the source `[none]` sentinel count as zero; missing tags remain missing |
| `publish_year`, `publish_month`, `publish_day_of_week`, `publish_hour` | Derived from actual `publish_time`, parsed as UTC. Weekday uses Monday=0 through Sunday=6. This is actual publication time, not scheduled time |
| `days_to_trending` | Integer UTC calendar-day difference: normalized `trending_date` minus normalized `publish_time`. Because `trending_date` has date-only precision, this is not exact elapsed hours or a publication-time-aligned duration |
| `region` | Two-letter source filename prefix |
| `category_name` | Regional JSON lookup from `category_id` |
| `observed_views` | Copy of raw `views` under the approved modeling target name |
| `log_observed_views` | `numpy.log1p(observed_views)`; `observed_views` remains unchanged |

`video_duration_seconds` is **excluded**: none of the ten approved CSVs contains a duration field. No duration is inferred or fabricated. `is_short` is consequently unavailable too.

The four regional files JP, KR, MX, and RU contain a small number of malformed UTF-8 byte sequences. The reader uses UTF-8 replacement decoding for those files (10, 66, 3, and 49 replacement characters, respectively). The raw bytes are unchanged, but replacement characters can affect text lengths and word counts; this is a known feature-quality limitation.

## 8. Feature Groups

These groups define candidate predictors for the **snapshot-estimation** task only. They do not represent pre-publication availability.

**Group A — Metadata-only snapshot features:** `region`, `category_name`, `title_length`, `title_word_count`, `description_length`, `description_word_count`, `description_missing`, `tag_count`, `publish_year`, `publish_month`, `publish_day_of_week`, and `publish_hour`.

**Group B — Snapshot-context feature:** `days_to_trending`. This is known only at/after the trending observation and must not be described as a pre-publication predictor.

**Retained traceability fields:** `video_id`, `trending_date`, `publish_time`, and `category_id`. These are not direct model inputs. `category_id` is retained as the original mapping key; `category_name` is the candidate categorical feature.

**Audit-only fields:** `comments_disabled`, `ratings_disabled`, and `video_error_or_removed`. They remain in the CSV to support review but are not default predictors because their timing/selection semantics may create leakage.

## 9. Missing-Value Handling

No train/test-dependent imputation is performed. Missing `description` values remain missing in `description_length` and `description_word_count`, and `description_missing=True` distinguishes them from empty strings. In the processed CSV, missing numeric text features serialize as empty cells; the boolean flag preserves the distinction. Missing/unmapped category names remain missing while `category_id` is retained. Missing targets cause the builder to stop with an error rather than silently fill or discard them; none were found in this source. No zero-fill, scaling, categorical encoding, or outlier removal is performed.

## 10. Target Distribution and Outliers

After exact-duplicate and conflict handling, the target summary is:

| Statistic | `observed_views` |
|---|---:|
| Minimum | 117 |
| Median | 176,046 |
| Mean | 1,341,316.385 |
| 95th percentile | 4,509,672.400 |
| 99th percentile | 20,731,956.560 |
| Maximum | 424,538,912 |

The high tail is retained. `log_observed_views` is an additional target representation, not a replacement for the raw count.

## 11. Final Modeling File

`data/processed/modeling_dataset.csv` contains **360,155 rows and 22 columns**:

```text
video_id
region
trending_date
publish_time
category_id
category_name
title_length
title_word_count
description_length
description_word_count
description_missing
tag_count
publish_year
publish_month
publish_day_of_week
publish_hour
comments_disabled
ratings_disabled
video_error_or_removed
days_to_trending
observed_views
log_observed_views
```

Raw `views`, `likes`, `dislikes`, `comment_count`, and `thumbnail_link` are not in the processed file. The target alias is not a predictor. This dataset-construction step does not train a model or fit an encoding pipeline, imputer, or scaler; those modeling steps are documented separately in `docs/model_training.md`.

## 12. Validation and Reproduction

From the project root, run:

```powershell
python src/build_modeling_dataset.py
```

The script uses paths relative to itself, reads raw files without writing them, applies the versioned regional mappings and deterministic conflict policy, regenerates the processed CSV, and prints a concise summary. It validates the output file/schema, non-missing non-negative target, finite log target, unique snapshot keys, valid region codes, date parsing, feature ranges, category coverage, and target distribution. Validation found zero missing targets, zero negative targets, zero non-finite log targets, zero duplicate final snapshot keys, and zero invalid/missing parsed publication/trending dates.

## 13. Known Limitations

- The source is selected on trending-list membership and is not representative of all uploads.
- The observed count is already present in the same snapshot; the dataset supports academic estimation, not future or pre-publication forecasting.
- `trending_date` has day-level precision, and the original view count was observed at an unrecorded time during that day.
- Repeated/conflicting rows exist; the documented rule removes unresolved conflicts without aggregating their counts.
- No channel ID, duration, scheduled publish time, or historical subscriber count is available. `channel_title` is only a proxy label.
- Category mapping is incomplete for 2,685 final rows.
- Several region-specific strings have malformed UTF-8 bytes; replacement decoding may affect text-derived feature values.
- Model generalization is conditional on the regional trending snapshot population and the chosen grouped/time split. The completed experiments and selected artifact are documented in `docs/model_training.md`; their results do not make the dataset representative of all uploads.

## 14. Modeling Status

The modeling dataset has been created and used in the completed experiments. Group A and Group B feature experiments were performed with Random Forest; model-family experiments also evaluated Random Forest, HistGradientBoosting, and XGBoost. The final candidate is the Group B XGBoost pipeline. The original target is `observed_views`; the training target is `log_observed_views = log1p(observed_views)`. Group B adds `days_to_trending` to Group A and uses the existing chronological, whole-video split. Experiment setup, results, and limitations are recorded in `docs/model_training.md`; the application schema is in `docs/final_model_contract.md`.
