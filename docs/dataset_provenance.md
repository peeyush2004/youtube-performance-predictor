# Dataset Provenance

## Source and Acquisition

- **Dataset:** Trending YouTube Video Statistics, DataSnaek
- **Source:** [Kaggle dataset page](https://www.kaggle.com/datasets/datasnaek/youtube-new)
- **Pinned version:** Kaggle Version 115
- **Access/download date:** 2026-10-03
- **Reproducible download handle:** `datasnaek/youtube-new/versions/115`
- **Acquisition tool:** KaggleHub 1.0.2 in the existing project virtual environment; the version-pinned dataset was extracted by KaggleHub, then the 20 source CSV/JSON files were copied without modification into `data/raw/`.
- **Original download archive:** 201 MB compressed. KaggleHub cache is outside the project. The raw directory contains the extracted source files, not a merged or rewritten dataset.
- **License shown on Kaggle:** CC0: Public Domain. The linked scraper code has its own BSD-2-Clause license; that code license is separate from the dataset license.

The Kaggle description says the source is a daily record of top-trending YouTube videos collected using the YouTube API, covering ten countries with up to 200 trending videos per region/day over roughly six months. This is a **trending-video sample**, not a representative sample of all YouTube uploads. Results must not be generalized to all videos.

The linked [DataSnaek scraper](https://github.com/mitchelljy/Trending-YouTube-Scraper) requests video snippet and statistics together, writes a `trending_date` generated on the collection day, and stores the returned view count in the `views` column. The date has day precision; it is not a timestamp of the exact API response time. The dataset does not provide a fixed post-publication horizon.

## Observation Unit and Target

The intended source observation unit is **video + region + trending date**: a video row recorded in a regional CSV on a daily trending snapshot. `video_id` repeats across dates and can also occur in several regional files. The data contains repeated `(video_id, trending_date)` keys within some region files, including exact duplicate rows and rows with differing contents. Therefore, an individual CSV row is a source record, and the key is not guaranteed unique. No duplicates were removed.

For future modeling, `observed_views` is a logical target alias for the raw CSV field `views`:

> `observed_views = views`, the view count recorded in that trending snapshot.

This target is not `views_30d`, not a final/lifetime count, and not a future view prediction. No `observed_views` column was added to the raw files.

## Original Files

The following files are present in `data/raw/`:

```text
CAvideos.csv
CA_category_id.json
DEvideos.csv
DE_category_id.json
FRvideos.csv
FR_category_id.json
GBvideos.csv
GB_category_id.json
INvideos.csv
IN_category_id.json
JPvideos.csv
JP_category_id.json
KRvideos.csv
KR_category_id.json
MXvideos.csv
MX_category_id.json
RUvideos.csv
RU_category_id.json
USvideos.csv
US_category_id.json
```

## Programmatic Inspection Summary

Inspection was performed by `src/inspect_dataset.py` using pandas in read-only mode. The script does not clean, deduplicate, transform, or write raw data. All ten video CSVs have 16 columns with the same names and dtypes. JSON category files were inspected as mappings, separately from video observations.

| Regional file | Rows | Columns | Unique video IDs | Unique `channel_title` strings (proxy) | Exact duplicate rows | Repeated snapshot keys | Extra rows for repeated keys | Repeated keys with differing contents | Missing descriptions | Text decoding used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `CAvideos.csv` | 40,881 | 16 | 24,427 | 5,076 | 0 | 0 | 0 | 0 | 1,296 | UTF-8 |
| `DEvideos.csv` | 40,840 | 16 | 29,627 | 6,085 | 0 | 0 | 0 | 0 | 1,552 | UTF-8 |
| `FRvideos.csv` | 40,724 | 16 | 30,581 | 6,680 | 0 | 0 | 0 | 0 | 2,912 | UTF-8 |
| `GBvideos.csv` | 38,916 | 16 | 3,272 | 1,625 | 171 | 174 | 174 | 3 | 612 | UTF-8 |
| `INvideos.csv` | 37,352 | 16 | 16,307 | 1,426 | 4,263 | 4,614 | 4,894 | 506 | 561 | UTF-8 |
| `JPvideos.csv` | 20,523 | 16 | 12,912 | 4,649 | 5,677 | 5,625 | 5,980 | 252 | 2,124 | UTF-8 replacement fallback (10 replacements) |
| `KRvideos.csv` | 34,567 | 16 | 15,876 | 4,043 | 2,316 | 2,486 | 2,625 | 195 | 3,163 | UTF-8 replacement fallback (66 replacements) |
| `MXvideos.csv` | 40,451 | 16 | 33,513 | 6,977 | 49 | 195 | 378 | 146 | 4,224 | UTF-8 replacement fallback (3 replacements) |
| `RUvideos.csv` | 40,739 | 16 | 34,282 | 6,931 | 46 | 211 | 417 | 165 | 2,464 | UTF-8 replacement fallback (49 replacements) |
| `USvideos.csv` | 40,949 | 16 | 6,351 | 2,207 | 48 | 50 | 50 | 2 | 570 | UTF-8 |
| **Total / union** | **375,942** | **16 per CSV** | **184,287** | **38,155 proxy labels** | **12,570** | **13,355 key groups** | **14,518** | **1,269** | **19,478** | **Mixed** |

| Regional file | `trending_date` min–max | `publish_time` min | `publish_time` max |
|---|---|---|---|
| `CAvideos.csv` | 2017-11-14 to 2018-06-14 | 2008-01-13 01:32:16 UTC | 2018-06-14 02:25:38 UTC |
| `DEvideos.csv` | 2017-11-14 to 2018-06-14 | 2012-04-18 10:58:15 UTC | 2018-06-14 00:33:42 UTC |
| `FRvideos.csv` | 2017-11-14 to 2018-06-14 | 2009-03-22 19:58:05 UTC | 2018-06-14 01:12:21 UTC |
| `GBvideos.csv` | 2017-11-14 to 2018-06-14 | 2007-03-05 16:23:55 UTC | 2018-06-13 13:11:56 UTC |
| `INvideos.csv` | 2017-11-14 to 2018-06-14 | 2017-05-27 20:27:30 UTC | 2018-06-13 18:05:15 UTC |
| `JPvideos.csv` | 2018-02-07 to 2018-06-14 | 2015-11-08 01:08:40 UTC | 2018-06-13 22:57:32 UTC |
| `KRvideos.csv` | 2017-11-14 to 2018-06-14 | 2011-09-29 13:08:39 UTC | 2018-06-13 19:29:43 UTC |
| `MXvideos.csv` | 2017-11-14 to 2018-06-14 | 2009-10-07 09:25:39 UTC | 2018-06-14 03:40:05 UTC |
| `RUvideos.csv` | 2017-11-14 to 2018-06-14 | 2011-06-20 11:10:09 UTC | 2018-06-14 00:00:03 UTC |
| `USvideos.csv` | 2017-11-14 to 2018-06-14 | 2006-07-23 08:24:11 UTC | 2018-06-14 01:31:53 UTC |

`trending_date` is a date token in `YY.DD.MM` form; displayed UTC midnight in the inspector is a parsing convention, not an observed time-of-day. The combined `publish_time` range is **2006-07-23 08:24:11 UTC through 2018-06-14 03:40:05 UTC**.

Across the ten CSVs, the mean is **2.040 observations per unique video ID**, the maximum is **2,312 rows for one video ID**, and **13,769 video IDs appear in more than one region**. The total includes repeated snapshots and exact duplicate rows. There is no `channelId` column. The 38,155 distinct `channel_title` strings are only a display-name proxy, not a verified count of unique channels.

The nine non-US category JSON files contain 31 category entries each; `US_category_id.json` contains 32. Each maps category metadata and has no video observation dates, video IDs, or channel IDs.

### Missingness, Repeats, and Encoding Notes

- In the CSVs, every inspected field has zero missing values except `description`, with 19,478 missing cells total. The category JSON mappings have zero missing values in the inspected normalized fields.
- Exact repeated rows total 12,570. Some repeated `(video_id, trending_date)` keys also have differing row contents (1,269 such keys); do not collapse either kind until a documented rule is approved.
- Six CSVs decode as UTF-8 without substitutions. UTF-8 decoding encountered malformed byte sequences in JP, KR, MX, and RU; the inspector and builder retry with UTF-8 replacement decoding (10, 66, 3, and 49 replacement characters respectively). Raw files are unchanged. These replacements can affect text lengths/word counts; review text-derived features with that limitation in mind.
- The schema has `views`, not `view_count`; `comment_count`, not `comments`; `publish_time`, not `publishedAt`; `channel_title`, not `channelTitle`; and `category_id`, not `categoryId`. It contains no duration field and no channel ID.

## Upstream and Use Considerations

Kaggle displays a CC0 license for this dataset. The rows were collected from YouTube API data; the Kaggle license statement should not be treated as overriding YouTube API Services Terms or Developer Policies. Those terms/policies can govern API-derived data storage, refreshing, aggregation, and derived analytics. Review current requirements with the project supervisor/institution before using this dataset for training or redistributing outputs.

The sample is selected by trending-list inclusion, metrics are post-publication observations, snapshots have only day-level dates, and repeated records are present. `views` must only be used as the observed snapshot target. Do not interpret it as a standardized 30-day view count or a future outcome. `likes`, `dislikes`, and `comment_count` are post-publication metrics and are excluded from model inputs.

No preprocessing or deduplication has been applied to `data/raw/`; it remains the untouched source copy. The separate derived dataset and its conflict policy are documented in `docs/modeling_dataset.md`.