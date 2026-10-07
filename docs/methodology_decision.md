# Methodology Decision

> **Superseded historical decision record.** This document preserves the alternatives and approval conditions considered on 2026-10-03. The project has since completed the DataSnaek snapshot-estimation implementation and model experiments. Approval gates and instructions below record the decision-time status; they do not describe the current project. The implemented methodology and final model are documented in `docs/modeling_dataset.md`, `docs/model_training.md`, and `docs/final_model_contract.md`.

Decision prepared 2026-10-03, using the then-current design in `docs/dataset_design.md`, the candidate review in `docs/dataset_research.md`, and official YouTube API documentation. At the time this decision was written, no dataset had been downloaded and no API had been called.

## 1. Problem Statement

The original project asks for a meaningful pre-publication prediction: use only values known immediately before scheduled publication to estimate cumulative views exactly 30 days later. The prior dataset research found no reviewed public dataset that supports that target for a defensible cohort.

This decision compares two paths without presenting an observed snapshot count as a future 30-day outcome. Option A preserves the original forecasting question but requires prospective collection. Option B changes the research question to supervised estimation of the view count recorded in a particular historical snapshot. If Option B is approved, its results apply to the dataset's trending-video sample and must not be described as pre-publication forecasts.

## 2. Option A: Prospective 30-Day Collection

### Feasibility and Data Plan

Option A is technically feasible only with a source of upcoming videos and cooperation from the channels that own them. Public API responses generally describe public resources. A researcher cannot use a public API key to see arbitrary channels' private drafts or scheduled uploads. A channel owner can provide the pre-publication snapshot directly or authorize an OAuth 2.0 client to access channel-specific resources. For multiple channels, each owner would need to participate and authorize access where required.

For each enrolled video, the collector would:

1. Obtain creator consent and a video record before the scheduled publication cutoff. Save the finalized category, duration, title/description lengths, tags, channel creation date, a timestamped subscriber count if permitted, planned publish time, timezone, channel/video identifiers, and exact feature-snapshot time. A creator-provided record is an alternative to reading a private scheduled video through OAuth.
2. Freeze the pre-publication feature row. Do not replace it later with updated metadata or subscriber counts.
3. Record actual publication time separately from planned time.
4. At the 30-day horizon, request the video's current statistics using the YouTube Data API `videos.list` endpoint and save the response time, view count, and offset from the intended horizon. No later count may be substituted for this observation.
5. Audit missing/late observations and retain source timestamps and consent/provenance records.

The Data API requires a Google account, Google Cloud API project, enabled YouTube Data API, and API credentials. A key can make eligible public-data requests; OAuth 2.0 is required for user-authorized/private channel data and for YouTube Analytics API requests. `videos.list` and `channels.list` each cost one quota unit per request. The current documentation lists a default 10,000-unit daily allocation for most Data API methods and a separate 100-calls/day limit for `search.list`; quota and limits can change. A known video-ID list and `videos.list` batches are preferable to repeated search calls. Never commit API keys or OAuth tokens.

### Time, Sample Size, and Academic Fit

Every label matures only after its video's 30-day outcome point. The earliest usable test result is therefore at least 30 days after the last included video's publication, in addition to recruitment and the period needed to obtain enough uploads. The total collection period depends on creator participation and upload frequency. Several dozen videos may support a technical demonstration but not robust generalization. Plan in the hundreds or more of unique videos across multiple channels, then justify the final sample with a pilot, class/channel coverage, and learning curves rather than claiming an arbitrary universal minimum. Repeated videos from one channel are not independent evidence of cross-channel performance.

This is the more scientifically aligned and deployment-relevant option if channel access, sufficient uploads, consent, and policy clearance are secured. It is also substantially more complex and slower than a ready-made dataset. Reproducibility is possible with a public collection protocol, a frozen feature schema, exact request timestamps, and a consented channel cohort, but the data remains dependent on API behavior and participating creators.

### API and Measurement Risks

- The API returns a view count at request time; it does not provide an arbitrary historical point-in-time count for a public video. A request can be scheduled at the 30-day instant, but the returned count and its processing/update time are not a guarantee of the exact underlying counter at that instant. Preserve the response timestamp and offset. Any allowed timing tolerance must be decided before collection and explicitly approved; do not silently widen the approved target window.
- YouTube documents view-count definition changes, including a change effective 2026-08-24. Counts may not be comparable across changes in measurement definition.
- Current Developer Policies specify storage/refresh requirements for API data and impose restrictions around aggregation and derived analytics for relevant API clients. Authorized and non-authorized data have different requirements. A cross-channel ML study must receive policy/institutional review before collecting, retaining, or training on API-derived data. An API key is not a license to retain or repurpose data indefinitely.
- Missing/deleted/private videos, hidden statistics, creator withdrawal, late uploads, and API quota/policy changes can reduce the usable sample.

Official references: [Data API setup and quota overview](https://developers.google.com/youtube/v3/getting-started), [per-method quota costs](https://developers.google.com/youtube/v3/determine_quota_cost), [video resource](https://developers.google.com/youtube/v3/docs/videos), [channel resource](https://developers.google.com/youtube/v3/docs/channels), [Analytics channel reports](https://developers.google.com/youtube/analytics/channel_reports), [API Terms](https://developers.google.com/youtube/terms/api-services-terms-of-service), and [Developer Policies](https://developers.google.com/youtube/terms/developer-policies). Consult the current documents again before any API work.

## 3. Option B: Public Dataset / Observed View Count

Option B is not the original forecasting problem. Its target is the view count contained in a particular dataset observation. For the strongest previously reviewed candidate, define one record as a `(video_id, region, trending_date)` snapshot and define `observed_views` as that row's `view_count`. `trending_date` is the day on which the linked scraper queried the YouTube API; the count is the API response from that collection run. This is a dated, day-resolution snapshot, not a fixed time since publication.

### Candidate Review and Selection

| Candidate | Snapshot/target support | Main reason it is weaker or stronger for Option B |
|---|---|---|
| **Trending YouTube Video Statistics (DataSnaek, Kaggle Version 115)** | `view_count` and `trending_date`; linked scraper shows the API collection process and field names | **Strongest candidate reviewed.** Dated observed counts, public scraper source, a versioned Kaggle release, documented CC0 card, and a useful set of metadata fields. Exact row count is not published; data is limited to videos while trending and repeats videos across dates/regions. |
| Rishav Sharma YouTube Trending Video Dataset (Kaggle Version 1346) | Daily `views`/`view_count` with `trending_date` | Large and updated, but release is mutable, its card does not state exact rows/full headers, and its trending-only selection remains. |
| YouTube Trending Videos Stats 2026 (Kaggle Version 3) | `views` at collection time and day-level `trending_date` | Card claims 178,399+ records, but its overview and file-summary schema counts conflict; it also documents estimated dislike backfills. |
| Top Trending Videos YouTube 2021 | Daily trend rows, publish time, view statistics | Short coverage window; exact row count/full schema not shown on card. |
| August 2020 YouTube Trending Statistics | `view_count`, `trending_date`, publication and metadata fields | Dated snapshots and compact release, but only a trending subset; no exact row count. |
| Google YouTube-8M | No view-count target or publication timestamp | Not compatible with observed-view-count regression. |

The recommendation is **DataSnaek Version 115**, not because it is the largest, but because its linked scraper documents that the view statistic and `trending_date` are collected together, its schema is available from the scraper, and its archived Kaggle version supports reproducibility better than the mutable alternatives. The Kaggle card reports CC0 and the archive is about 539 MB with ten country CSVs; the exact row count is not stated. Pin and verify the precise version, file inventory, license text, and upstream-use terms before approval to download. CC0 on the Kaggle card does not itself settle YouTube API terms for downstream use.

### Appropriate Inputs and Leakage Controls

For this snapshot-estimation task, the prediction time is the dataset observation time, not pre-publication time. Candidate features from the row include category, title length, description length, tag count, actual publish weekday/hour, and optionally video age at the snapshot (`trending_date` minus `publishedAt`). These values describe the observed row and do not establish that they were known before publication. The dataset does not provide duration, a scheduled publication time, `is_short`, channel age, or historical subscriber count. Channel age might be added through an external lookup, but it is not in the archived row and should not be assumed in Version 1.

Use `view_count` only as target (`observed_views`). Exclude likes, dislikes, comment count, watch time, and all engagement/ranking statistics from inputs. Exclude `video_id` and `channelId` from the model matrix but retain them to deduplicate and group splits. Do not claim that the resulting feature set is pre-publication-safe. Titles/tags/descriptions/category are values returned at snapshot time and may have changed since publication.

The trending condition is itself a post-publication selection mechanism. Evaluation must be described as applying to records in this historical trending dataset, not all YouTube uploads. A video can occur in multiple dates and regions; keep all rows for a video in one evaluation partition, and report a chronological holdout as an additional check if the date coverage permits it. Otherwise repeated snapshots can leak near-identical records across train and test.

## 4. Comparison

| Criterion | Option A: prospective 30-day collection | Option B: public observed-count dataset |
|---|---|---|
| Target validity | High for the intended research question if the pre-publication record and day-30 observation are correctly captured; exact API observation timing still needs a prespecified rule | High only for the redefined target `observed_views` at the dataset snapshot; does not validate a 30-day or future-view target |
| Leakage control | Strong in principle: freeze inputs before publication and prevent post-publication fields entering features | Manageable for snapshot estimation if post-publication engagement fields are excluded; cannot support pre-publication claims |
| Data availability | Not yet available; requires participating channels and prospective uploads | Public archive is available after approval, but exact row count must be verified |
| Time to obtain data | Recruitment/collection period plus at least 30 days for the final labels to mature | Short once approved; no label-maturity wait |
| API dependency | High; Data API key/project, possibly owner OAuth, API quota and policy review | No API calls needed to consume the Kaggle archive; underlying data rights/API terms still require review |
| Reproducibility | Good with a public protocol and timestamps, but participant access, API changes, and policy constraints affect replication | Good if Kaggle Version 115 and files are pinned; scraper source documents field/collection provenance |
| Implementation complexity | High: consent, scheduled metadata capture, secure credentials, polling, timestamping, missing-label handling, and policy compliance | Moderate: acquire approved snapshot, audit schema, construct snapshot rows, prevent duplicate-video split leakage |
| Dataset size potential | Uncertain; depends on channel recruitment/upload rates; hundreds or more unique videos may take months to accrue | Potentially many dated rows (up to 200 trending entries/region/day across countries), but repeated/censored records mean fewer independent videos; exact count is undisclosed |
| ML suitability | Strong for a genuine forward prediction if the cohort is large/diverse enough | Suitable for tabular regression/classification of observed counts; not a forecasting or pre-publication model |
| Deployment suitability | High alignment with a pre-publication tool if future users can supply the same frozen features | Low: the observed count already exists in the snapshot/API; useful mainly as an academic estimation/analysis task |
| MCA project suitability | Academically strong if access, time, sample, and policy approval are secured; otherwise high execution risk | Academically suitable with a narrower claim, source audit, careful splits, and limitations stated prominently |
| Main limitation | Recruitment, label maturity, exact measurement, and current API data-use/storage restrictions | Trending-only selection and co-observed metadata; target is already observed, not future views |

## 5. Leakage Considerations

**Option A:** Capture every input before the cutoff and preserve its timestamp. Post-publication views, likes, comments, watch time, engagement, and current subscriber counts are forbidden predictors. Historical channel features must use only outcomes that had matured before the current video's cutoff. Store the 30-day count only as the target, with its API response timestamp and horizon offset.

**Option B:** `observed_views` is the target and must not enter the input matrix. Exclude other post-publication counts (likes/comments/dislikes/watch time) and any trend/popularity rank as inputs. Dataset text/category values are observed at the snapshot, not guaranteed at publication. Video IDs and channel IDs are grouping keys, not predictors. Group all snapshots of a video in the same split. Trending-list inclusion limits the population regardless of feature filtering.

## 6. Recommendation

**Recommend Option B, subject to approval of the narrower research question and confirmation of dataset-use terms.** Option A is the only path that preserves the original 30-day pre-publication forecasting objective, but at present it has no enrolled cohort, depends on future channel cooperation and mature labels, and raises nontrivial API authorization, measurement, retention, and derived-data policy questions. Without those prerequisites, it cannot yet support a reliable MCA-scale experiment.

Option B has a valid, observable supervised target when stated precisely: a historical snapshot's view count. The DataSnaek release is the strongest researched candidate because its scraper explains the per-day count capture, its schema and collection code are public, and its Kaggle version is fixed. Its use is academically defensible only if the thesis title, abstract, evaluation, and conclusions consistently limit claims to observed counts in the historical trending sample. It is not a substitute for the original forecasting system and has little direct deployment value because the target count is already available at that snapshot.

If pre-publication prediction is non-negotiable, do not proceed with Option B under the current project title. Revisit Option A only after channel access, a sufficient cohort, a precise observation-timing rule, and API/data-use clearance are secured.

## 7. Revised Prediction Target

If Option B is approved, replace `views_30d` for this study with:

> `observed_views`: the `view_count` recorded for a video in a DataSnaek Version 115 row on its `trending_date` collection day.

The row unit is one video-region-snapshot, not necessarily one unique video. Do not call this target “views after 30 days,” “final views,” or “expected future views.” If the existing Low/Medium/High classification is retained, derive its 33rd/67th percentile thresholds from training targets only and describe the classes as relative to this dataset's training population.

At the time this decision was written, this was a proposed scope change. The project has since adopted and completed the snapshot-estimation methodology; its implemented target and dataset are documented in `docs/modeling_dataset.md`, and the final application contract is in `docs/final_model_contract.md`.

## 8. Revised Project Title Options

1. **YouTube Trending Video View Count Estimation Using Machine Learning** (recommended; names both the population and task)
2. **Metadata-Based Estimation of Observed View Counts in YouTube Trending Videos**
3. **A Machine Learning Study of View Counts in Historical YouTube Trending Snapshots**

Avoid “30-Day Performance Prediction,” “pre-publication prediction,” and broad claims about all YouTube videos if Option B is selected.

## 9. Data Collection Plan

### Option B recommendation recorded at the time (conditional then)

1. Confirm that the current Kaggle Version 115 is accessible and that its CC0 statement and upstream YouTube/API terms are acceptable for the academic use; ask the supervisor/institution for review if uncertain.
2. After explicit approval, download that exact version to `data/raw/`; record the access date, version, filenames, hashes, and source URL.
3. Inspect actual headers, exact row counts, date parsing, countries, duplicate video-region-date records, missingness, and target semantics. Do not infer unpublished row counts from the card's maximum daily trend listings.
4. Create one analysis record per unique `(video_id, region, trending_date)` after resolving only exact duplicate rows by a documented rule. Preserve raw files unchanged.
5. Set `observed_views` from that row's `view_count`; retain `trending_date` as target-observation provenance. Never replace it with a later count or infer a day-30 target.
6. Use metadata-derived features only; exclude engagement counters and IDs from model inputs. Split by video ID so repeated snapshots cannot cross partitions. Consider a later-date holdout and a channel-held-out sensitivity check if enough distinct groups remain.
7. Reassess effective sample size after counting unique videos and channels. If too few independent videos remain, stop and revisit scope rather than treating repeated snapshots as independent videos.

### Option A if the project later returns to 30-Day Forecasting

Recruit consenting creators/channels before implementation; obtain each feature snapshot before publication by a creator-supplied form/export or properly authorized OAuth. Freeze the feature row, planned and actual publication timestamps, and timezone. Poll `videos.list` at the prespecified 30-day instant, retain the API response timestamp/offset, and document missing/deleted videos. Obtain policy/institutional review for API credentials, authorized data, aggregation across channel owners, retention, and model-derived use. Define any acceptable observation tolerance before collection; if exact API timing cannot meet the approved target, ask the user to approve a methodology change before collecting.

No API key, OAuth credential, YouTube API request, dataset download, or dataset copy was created as part of this decision.

## 10. Limitations

- The recommended dataset samples videos because they appeared in regional trending lists. Results do not generalize to all uploaded videos, non-trending videos, or future unseen uploads without separate evidence.
- `trending_date` is day-level, not a standardized post-publication horizon. Rows may represent videos of very different ages and repeated observations of the same video.
- Title, description, tags, and category are snapshot-time API values and may have been edited after publication. Duration, `is_short`, scheduled time, and historical subscriber count are not in the candidate schema.
- The target view count is already observed in the dataset/API at the snapshot. A deployed system would not need ML to retrieve that count; the project's value is educational analysis, not live count forecasting.
- The Kaggle card reports CC0, but this does not automatically settle rights or current API terms for upstream YouTube-derived data. Verify before download/use.
- Option A remains the better scientific design for genuine pre-publication forecasting if cohort access and policy approval are later obtained; this document does not claim it is impossible.
- The 2026 API view-count definition change may complicate comparing historical/current view metrics.

## 11. Final Decision (historical; superseded)

**Decision at the time:** Option B, DataSnaek Kaggle Version 115, was recommended with a narrower observed_views snapshot-estimation target, subject to approval of the scope and dataset-use terms. The project has since completed this implementation. Its current scope is historical estimation of observed_views at the row's trending_date, not prediction of a 30-day outcome or pre-publication performance.

The approval and implementation gates recorded in this historical decision no longer describe the project status. The implemented dataset, experiments, and final application contract are documented in docs/modeling_dataset.md, docs/model_training.md, and docs/final_model_contract.md.