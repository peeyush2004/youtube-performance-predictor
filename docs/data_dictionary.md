# Data Dictionary

This source-field dictionary records the initial leakage review of DataSnaek Kaggle Version 115, downloaded on 2026-10-03. Examples in angle brackets are format placeholders, not fabricated dataset rows. Raw files remain unchanged. The current task is snapshot estimation: the target is the count recorded on a trending snapshot. Use `docs/modeling_dataset.md` and `docs/final_model_contract.md` for the completed processed schema and final model inputs; the initial review notes below do not supersede those later records.

## Regional CSV Fields and Leakage Audit

`SAFE` means usable only for the approved snapshot-time estimation task. It does not mean the value was available before publication. `POTENTIAL LEAKAGE` fields are excluded from an initial model unless a later feature audit justifies them. `DEFINITE LEAKAGE` values cannot be predictors for the stated target. `IDENTIFIER` fields are retained for audit/splitting but excluded from ordinary model inputs. `TARGET` is the supervised label only.

| Column name | Data type | Description | Example value | Role | Leakage status | Modeling-task use |
|---|---|---|---|---|---|---|
| `video_id` | string | YouTube video identifier | `<video ID>` | Row/video key | IDENTIFIER | Deduplication and grouped splitting only; not a model feature |
| `trending_date` | string (`YY.DD.MM`) | Regional trending snapshot collection day | `17.14.11` | Observation-time provenance | SAFE (provenance only) | Parse for date range and temporal holdout; do not call it an exact API request timestamp |
| `title` | string | Title returned in the snapshot API response | `<title text>` | Candidate metadata | POTENTIAL LEAKAGE | Exclude initially; value may have been edited after publication and is not pre-publication ground truth |
| `channel_title` | string | Channel display name returned in the snapshot | `<channel display name>` | Channel label | IDENTIFIER | Use only for descriptive checks; not a stable channel ID or model feature |
| `category_id` | int64 | YouTube category identifier | `10` | Candidate metadata | POTENTIAL LEAKAGE | Consider only for snapshot-time estimation after verifying category mapping/vintage; not a pre-publication guarantee |
| `publish_time` | string (ISO-like timestamp) | Actual publication timestamp | `<YYYY-MM-DDTHH:MM:SSZ>` | Temporal metadata | SAFE (snapshot task only) | Use as publication provenance; it is actual time, not scheduled time, and cannot support a pre-publication claim |
| `tags` | string | Tags returned in the snapshot, serialized as pipe-separated text | `[none]` or `<tag text>` | Candidate metadata | POTENTIAL LEAKAGE | Exclude initially; may be edited after publication and some regional files need encoding validation |
| `views` | int64 | View count returned in the same trending snapshot | `117` | **TARGET** (`observed_views`) | TARGET; DEFINITE LEAKAGE if used as an input | Set the modeling target alias `observed_views = views` outside raw files; never use as a predictor |
| `likes` | int64 | Like count in the post-publication snapshot | `0` | Post-publication metric | POTENTIAL LEAKAGE | Exclude from inputs; no earlier prediction timestamp is provided |
| `dislikes` | int64 | Dislike count in the snapshot | `0` | Post-publication metric | POTENTIAL LEAKAGE | Exclude from inputs; it is a post-publication engagement metric |
| `comment_count` | int64 | Comment count in the post-publication snapshot | `0` | Post-publication metric | POTENTIAL LEAKAGE | Exclude from inputs; raw name is not `comments` |
| `thumbnail_link` | string | Thumbnail URL returned in the snapshot | `<thumbnail URL>` | Candidate metadata | POTENTIAL LEAKAGE | Exclude initially; thumbnail may change and no pre-publication version is archived |
| `comments_disabled` | bool | Whether comments were disabled in the observed record | `False` | Snapshot status | POTENTIAL LEAKAGE | Data-quality/status audit only; do not use as an input |
| `ratings_disabled` | bool | Whether ratings were disabled in the observed record | `False` | Snapshot status | POTENTIAL LEAKAGE | Data-quality/status audit only; do not use as an input |
| `video_error_or_removed` | bool | Source flag for a video error or removal | `False` | Data-quality/status flag | POTENTIAL LEAKAGE | Audit availability/selection effects only; do not use as an input |
| `description` | string | Description returned in the snapshot; may be missing | `<description text>` or blank | Candidate metadata | POTENTIAL LEAKAGE | Exclude initially; may have been edited after publication; 19,478 missing descriptions across CSVs |

### Requested Aliases Not Present

| Requested name | Present? | Actual source field / note |
|---|---|---|
| `view_count` | No | The view target is named `views` in the CSV |
| `comments` | No | The count field is `comment_count` |
| `publishedAt` | No | The timestamp field is `publish_time` |
| `channelTitle` | No | The display-name field is `channel_title`; there is no `channelId` |
| `categoryId` | No | The field is `category_id` |
| `duration` | No | No video duration field is present |

## Category JSON Mapping Fields

The ten `*_category_id.json` files are regional category lookup resources, not video observations. Kaggle JSON roots contain `kind`, `etag`, and `items`; the following describes the normalized `items` entries reported by the inspector.

| Normalized column | Data type | Description | Example value | Role | Leakage status | Modeling-task use |
|---|---|---|---|---|---|---|
| `kind` | string | YouTube category resource type | `<resource kind>` | Mapping metadata | IDENTIFIER | Provenance only |
| `etag` | string | API resource version token | `<etag>` | Mapping metadata | IDENTIFIER | Provenance only |
| `id` | string | Category identifier used to join with CSV `category_id` | `10` | Lookup key | IDENTIFIER | Optional category-name mapping after approval; not an independent model feature |
| `snippet.channelId` | string | Channel identifier associated with the category resource | `<channel ID>` | Mapping metadata | IDENTIFIER | Not a video channel ID; do not use as a model feature |
| `snippet.title` | string | Category display title | `<category title>` | Lookup label | SAFE (lookup only) | Optional category display/interpretation |
| `snippet.assignable` | bool | Whether the category can be assigned | `True` | Lookup metadata | SAFE (lookup only) | Optional validation of category IDs |

## Snapshot-Time Interpretation

The `SAFE` fields above are safe only relative to the new prediction point at the dataset snapshot. None of this dataset should be represented as pre-publication feature history. `observed_views` is the snapshot's `views`; same-snapshot likes/dislikes/comments are not included as predictors. `video_id` and `channel_title` are not model features. Category, title, tags, thumbnail, and description are not approved initial predictors until their snapshot-vintage and encoding limitations are reviewed. Feature extraction and duplicate handling for the modeling dataset are documented in `docs/modeling_dataset.md`. This initial raw-source review predates the fitted preprocessing and model experiments documented in `docs/model_training.md`.