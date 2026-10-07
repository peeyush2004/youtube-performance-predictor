"""Streamlit UI for the frozen Group B XGBoost inference pipeline."""

from __future__ import annotations

import logging

import streamlit as st

from app.prediction import (
    InputValidationError,
    TRAINING_YEAR_RANGE,
    load_model_bundle,
    predict_record,
)


LOGGER = logging.getLogger(__name__)
WEEKDAY_LABELS = {
    0: "Monday (0)",
    1: "Tuesday (1)",
    2: "Wednesday (2)",
    3: "Thursday (3)",
    4: "Friday (4)",
    5: "Saturday (5)",
    6: "Sunday (6)",
}

st.set_page_config(
    page_title="YouTube Trending Video View Count Estimation",
    page_icon="\U0001F4CA",
    layout="wide",
)


GLOBAL_STYLES = """
<style>
:root {
    --yt-ink: #edf3fb;
    --yt-muted: #a6b4c8;
    --yt-border: rgba(148, 172, 204, 0.19);
    --yt-panel: rgba(16, 28, 45, 0.78);
    --yt-accent: #53d6c5;
    --yt-blue: #6da8ff;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 82% 2%, rgba(57, 126, 180, 0.16), transparent 32rem),
        radial-gradient(ellipse at 12% 18%, rgba(53, 184, 160, 0.09), transparent 30rem),
        #0b1220;
    color: var(--yt-ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--yt-border);
    border-radius: 18px;
    background: var(--yt-panel);
    box-shadow: 0 14px 38px rgba(0, 0, 0, 0.16);
    transition: border-color 180ms ease, transform 180ms ease, box-shadow 180ms ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(109, 168, 255, 0.34);
    box-shadow: 0 17px 40px rgba(0, 0, 0, 0.22);
}
.hero-card {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    padding: 2.25rem 2.4rem 2rem;
    border: 1px solid rgba(109, 168, 255, 0.27);
    border-radius: 24px;
    background: linear-gradient(118deg, rgba(20, 38, 61, 0.97), rgba(17, 33, 50, 0.91) 58%, rgba(17, 54, 62, 0.84));
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.25);
    animation: rise-in 520ms cubic-bezier(.2,.75,.25,1) both;
}
.hero-card::before {
    content: "";
    position: absolute;
    z-index: -1;
    inset: 0;
    opacity: 0.28;
    background-image: linear-gradient(rgba(137, 176, 215, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(137, 176, 215, 0.08) 1px, transparent 1px);
    background-size: 34px 34px;
    mask-image: linear-gradient(90deg, transparent, black 58%, transparent);
}
.hero-card::after {
    content: "";
    position: absolute;
    left: 2.4rem;
    bottom: 0;
    height: 3px;
    width: min(38%, 300px);
    border-radius: 99px;
    background: linear-gradient(90deg, var(--yt-accent), var(--yt-blue), transparent);
    transform-origin: left;
    animation: accent-in 850ms 120ms ease-out both;
}
.hero-eyebrow, .snapshot-label, .section-number, .result-kicker {
    color: var(--yt-accent);
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.hero-title {
    max-width: 850px;
    margin: 0.7rem 0 0.65rem;
    color: var(--yt-ink);
    font-size: clamp(2rem, 4vw, 3.15rem);
    font-weight: 730;
    line-height: 1.08;
    letter-spacing: -0.035em;
}
.hero-subtitle {
    margin: 0;
    color: #c1cee0;
    font-size: 1.08rem;
    line-height: 1.55;
}
.method-card {
    margin: 1rem 0 1.35rem;
    padding: 1.05rem 1.25rem;
    border: 1px solid rgba(83, 214, 197, 0.28);
    border-left: 4px solid var(--yt-accent);
    border-radius: 14px;
    background: linear-gradient(100deg, rgba(23, 49, 58, 0.88), rgba(16, 30, 45, 0.83));
    color: #dce8f4;
    font-size: 0.94rem;
    line-height: 1.6;
}
.method-label { display: block; margin-bottom: 0.28rem; color: var(--yt-accent); font-size: 0.72rem; font-weight: 750; letter-spacing: 0.12em; text-transform: uppercase; }
.snapshot-card {
    min-height: 104px;
    padding: 1.05rem 1.1rem;
    border: 1px solid var(--yt-border);
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(20, 35, 54, 0.95), rgba(14, 24, 39, 0.91));
    transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
}
.snapshot-card:hover { transform: translateY(-3px); border-color: rgba(83, 214, 197, 0.42); background: #14243a; }
.snapshot-value { margin-top: 0.42rem; color: var(--yt-ink); font-size: 1.08rem; font-weight: 680; }
.section-heading { display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.25rem; }
.section-number {
    display: grid;
    width: 2.35rem;
    height: 2.35rem;
    flex: 0 0 2.35rem;
    place-items: center;
    border: 1px solid rgba(83, 214, 197, 0.31);
    border-radius: 11px;
    background: rgba(83, 214, 197, 0.08);
    color: var(--yt-accent);
    font-size: 0.78rem;
}
.section-title { margin: 0; color: var(--yt-ink); font-size: 1.14rem; font-weight: 700; letter-spacing: 0.005em; }
.section-copy { margin: 0.18rem 0 0; color: var(--yt-muted); font-size: 0.86rem; }
div[data-testid="stFormSubmitButton"] button {
    min-height: 3.15rem;
    border: 1px solid rgba(83, 214, 197, 0.55);
    border-radius: 13px;
    background: linear-gradient(105deg, #36bfae, #478fdc);
    color: #07131f;
    font-size: 1rem;
    font-weight: 750;
    box-shadow: 0 8px 22px rgba(43, 156, 174, 0.19);
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
}
div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-2px); filter: brightness(1.08); box-shadow: 0 12px 27px rgba(43, 156, 174, 0.29); }
div[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid rgba(109, 168, 255, 0.76); outline-offset: 3px; }
div[data-testid="stMetric"] {
    padding: 1.1rem 1.25rem;
    border: 1px solid rgba(109, 168, 255, 0.26);
    border-radius: 16px;
    background: linear-gradient(120deg, rgba(23, 43, 66, 0.94), rgba(18, 55, 59, 0.72));
    animation: rise-in 360ms ease-out both;
}
div[data-testid="stMetricLabel"] { color: #b9c8dc; font-size: 0.82rem; letter-spacing: 0.07em; text-transform: uppercase; }
div[data-testid="stMetricValue"] { color: var(--yt-ink); font-size: clamp(2.3rem, 5vw, 3.4rem); font-weight: 760; letter-spacing: -0.035em; }
.result-kicker { margin: 0.25rem 0 0.85rem; }
.info-card-title { margin-bottom: 0.35rem; color: var(--yt-accent); font-size: 0.72rem; font-weight: 750; letter-spacing: 0.12em; text-transform: uppercase; }
.info-card-copy { margin: 0; color: #ced9e7; font-size: 0.91rem; line-height: 1.55; }
[data-testid="stExpander"] { border: 1px solid var(--yt-border); border-radius: 15px; background: rgba(16, 28, 45, 0.56); }
[data-testid="stCaptionContainer"] { color: var(--yt-muted); }
@keyframes rise-in { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: translateY(0); } }
@keyframes accent-in { from { transform: scaleX(0.08); opacity: 0.4; } to { transform: scaleX(1); opacity: 1; } }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
}
</style>
"""


@st.cache_resource(show_spinner="Loading the saved model...")
def cached_model_bundle():
    return load_model_bundle()


def section_heading(number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="section-number">{number}</span>
            <div>
                <div class="section-title">{title}</div>
                <p class="section-copy">{description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def snapshot_card(label: str, value: str) -> str:
    return (
        '<div class="snapshot-card">'
        f'<div class="snapshot-label">{label}</div>'
        f'<div class="snapshot-value">{value}</div>'
        '</div>'
    )


def about_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="info-card-title">{title}</div><p class="info-card-copy">{body}</p>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="hero-card">
            <div class="hero-eyebrow">FINAL MODEL &middot; GROUP B</div>
            <h1 class="hero-title">YouTube Trending Video View Count Estimation</h1>
            <p class="hero-subtitle">Machine Learning-Based Estimation Using Trending Snapshot Data</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <aside class="method-card">
            <span class="method-label">Methodology note</span>
            This application estimates observed view counts for videos in dated YouTube trending snapshots. It is not a pre-publication forecasting model and should not be interpreted as predicting future views for a newly uploaded video.
        </aside>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    for column, label, value in zip(
        cards,
        ("Model", "Target", "Features", "Data Context"),
        ("XGBoost", "Log Observed Views", "13", "Trending Snapshots"),
    ):
        column.markdown(snapshot_card(label, value), unsafe_allow_html=True)

    try:
        bundle = cached_model_bundle()
    except FileNotFoundError:
        LOGGER.exception("The saved model or metadata could not be found")
        st.error("The saved model files are unavailable. Please restore the model artifact and metadata, then restart the application.")
        st.stop()
    except Exception:
        LOGGER.exception("Could not load the saved prediction pipeline")
        st.error("The saved model could not be loaded. Please contact the project maintainer.")
        st.stop()

    with st.expander("Model Details"):
        model_columns = st.columns(3)
        model_items = (
            ("Algorithm", "XGBoost Regressor"),
            ("Target", "log_observed_views"),
            ("Feature Group", "Group B"),
            ("Input Features", "13"),
            ("Training Target Representation", "log1p(observed_views)"),
            ("Displayed Prediction", "expm1(predicted log-view value)"),
        )
        for index, (label, value) in enumerate(model_items):
            model_columns[index % 3].markdown(
                f'<div class="snapshot-card"><div class="snapshot-label">{label}</div><div class="snapshot-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Held-out chronological test metrics: MAE 964,651.57 | RMSE 4,093,751.92 | R-squared 0.062105 | MedianAE 130,516.76 | Log MAE 1.100355.")

    regions = list(bundle.regions)
    categories = list(bundle.categories)
    default_region = regions.index("US") if "US" in regions else 0
    default_category = categories.index("Education") if "Education" in categories else 0

    with st.form("estimate_form"):
        with st.container(border=True):
            section_heading("01", "VIDEO METADATA", "Text-derived characteristics of the video")
            left, right = st.columns(2)
            with left:
                title_length = st.number_input("Title Length", min_value=0, value=54, step=1)
                description_length = st.number_input("Description Length", min_value=0, value=55, step=1)
                tag_count = st.number_input("Tag Count", min_value=0, value=4, step=1)
                region = st.selectbox("Region", options=regions, index=default_region)
            with right:
                title_word_count = st.number_input("Title Word Count", min_value=0, value=9, step=1)
                description_word_count = st.number_input("Description Word Count", min_value=0, value=9, step=1)
                description_missing = st.checkbox("Description Missing", value=False)
                category_name = st.selectbox("Category", options=categories, index=default_category)

        with st.container(border=True):
            section_heading("02", "PUBLICATION INFORMATION", "When the video was published")
            left, right = st.columns(2)
            with left:
                publish_year = st.number_input(
                    "Publish Year",
                    min_value=TRAINING_YEAR_RANGE[0],
                    max_value=TRAINING_YEAR_RANGE[1],
                    value=min(2018, TRAINING_YEAR_RANGE[1]),
                    step=1,
                )
                publish_month = st.number_input("Publish Month", min_value=1, max_value=12, value=5, step=1)
            with right:
                publish_day_of_week = st.selectbox(
                    "Publish Day of Week",
                    options=list(WEEKDAY_LABELS),
                    index=2,
                    format_func=lambda value: WEEKDAY_LABELS[value],
                    help="Training encoding: Monday=0 through Sunday=6.",
                )
                publish_hour = st.number_input("Publish Hour", min_value=0, max_value=23, value=14, step=1)

        with st.container(border=True):
            section_heading("03", "TRENDING SNAPSHOT CONTEXT", "Context of the dated trending observation")
            context_col, note_col = st.columns([1, 2])
            with context_col:
                days_to_trending = st.number_input(
                    "Days to Trending",
                    min_value=0,
                    value=2,
                    step=1,
                    help="Number of days between publication and the recorded trending snapshot.",
                )
            with note_col:
                st.caption("This value describes the dated snapshot context; it is not a future prediction horizon.")

        with st.container(border=True):
            section_heading("04", "PREDICTION", "Estimate the observed count for this snapshot")
            submitted = st.form_submit_button("Estimate Observed Views", type="primary", use_container_width=True)

    if submitted:
        record = {
            "region": region,
            "category_name": category_name,
            "title_length": title_length,
            "title_word_count": title_word_count,
            "description_length": description_length,
            "description_word_count": description_word_count,
            "tag_count": tag_count,
            "publish_year": publish_year,
            "publish_month": publish_month,
            "publish_day_of_week": publish_day_of_week,
            "publish_hour": publish_hour,
            "description_missing": description_missing,
            "days_to_trending": days_to_trending,
        }
        try:
            _, predicted_views = predict_record(record, bundle)
            with st.container(border=True):
                st.markdown('<div class="result-kicker">&#10003; Prediction completed</div>', unsafe_allow_html=True)
                st.metric("Estimated Observed Views", f"{predicted_views:,.0f}")
                st.caption("Views in the dated trending snapshot.")
                st.caption("Estimated from the supplied video metadata, publication information, and trending snapshot context.")
        except InputValidationError as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Prediction failed")
            st.error("The estimate could not be generated. Check the entered values and try again.")

    with st.expander("About This Project"):
        about_columns = st.columns(2)
        with about_columns[0]:
            with st.container(border=True):
                about_card("Purpose", "Estimate observed views for videos represented in dated YouTube trending snapshots.")
        with about_columns[1]:
            with st.container(border=True):
                about_card("Dataset", "Historical, region-specific trending records. The sample includes videos selected for trending and does not represent all uploads.")
        with about_columns[0]:
            with st.container(border=True):
                about_card("Methodology", "The saved Group B XGBoost pipeline uses the existing chronological train, validation, and test split. Group B includes days_to_trending, which is available only with snapshot context.")
        with about_columns[1]:
            with st.container(border=True):
                about_card("Limitations", "The held-out period is one historical test window. The high-view tail remains difficult and is underpredicted; estimates are not calibrated uncertainty intervals or universal YouTube performance claims.")


if __name__ == "__main__":
    main()
