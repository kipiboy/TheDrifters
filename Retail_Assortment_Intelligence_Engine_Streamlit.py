import streamlit as st
import pandas as pd
import math
import altair as alt
from snowflake.snowpark.context import get_active_session

# ─── SESSION & CONFIG ──────────────────────────────────────────────────────────
session = get_active_session()

st.set_page_config(
    page_title="Retail Assortment Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

SCHEMA = "RAIE_ANALYTICS.CORE_TABLES"

# ─── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* KPI Cards — flexible, spacious, sidebar-safe */
div[data-testid="metric-container"] {
    background: #f7f8fc;
    border-radius: 12px;
    padding: 18px 20px 16px 20px;
    border-left: 4px solid #1a73e8;
    min-width: 0;
    overflow: hidden;
}
div[data-testid="metric-container"] label {
    font-size: 12px !important;
    white-space: normal !important;
    word-break: break-word;
    line-height: 1.4;
    color: #5a6a85 !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
    white-space: normal !important;
    word-break: break-word;
    line-height: 1.3;
}
div[data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 12px !important;
}

/* AI Advisor box */
.ai-narrative-box {
    background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%);
    border: 1.5px solid #7c3aed;
    border-radius: 14px;
    padding: 22px 26px;
    margin-top: 10px;
    line-height: 1.75;
    color: #1e1b4b;
    font-size: 15px;
}

/* Status pill badges */
.badge-green  { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-red    { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-yellow { background:#fef9c3; color:#78350f; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* Section divider */
.section-title {
    font-size:17px; font-weight:700; color:#1a1a2e;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px; margin-bottom: 14px;
}

/* Sidebar store info card */
.store-info-card {
    background:#f0f4ff; border-radius:10px;
    padding:12px 14px; font-size:13px; color:#334155;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADERS ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_regions():
    return session.sql(
        f"SELECT REGION_ID, REGION_NAME FROM {SCHEMA}.DIM_REGION ORDER BY REGION_NAME"
    ).to_pandas()

@st.cache_data(ttl=600)
def load_stores():
    return session.sql(
        f"SELECT STORE_ID, STORE_NAME, CITY, REGION_ID, STORE_TYPE FROM {SCHEMA}.DIM_STORES ORDER BY STORE_NAME"
    ).to_pandas()

@st.cache_data(ttl=600)
def load_categories():
    return session.sql(
        f"SELECT DISTINCT CATEGORY FROM {SCHEMA}.DIM_PRODUCT ORDER BY CATEGORY"
    ).to_pandas()

@st.cache_data(ttl=600)
def load_body_profiles():
    return session.sql(f"SELECT * FROM {SCHEMA}.DIM_BODY_PROFILE").to_pandas()

@st.cache_data(ttl=600)
def load_region_body_dist():
    return session.sql(f"SELECT * FROM {SCHEMA}.REGION_BODY_PROFILE_DISTRIBUTION").to_pandas()

@st.cache_data(ttl=600)
def load_size_demand():
    return session.sql(f"SELECT * FROM {SCHEMA}.SIZE_DEMAND_BY_BODY_PROFILE").to_pandas()

@st.cache_data(ttl=600)
def load_size_map():
    return session.sql(
        f"SELECT SIZE_ID, SIZE FROM {SCHEMA}.DIM_SIZE ORDER BY SIZE_ID"
    ).to_pandas()

@st.cache_data(ttl=600)
def load_date_bounds():
    return session.sql(
        f"SELECT MIN(DATE) AS MIN_DATE, MAX(DATE) AS MAX_DATE FROM {SCHEMA}.DIM_DATE"
    ).to_pandas()


def load_sales(store_id: str, category: str, start_date, end_date) -> pd.DataFrame:
    """
    Derives GENDER by joining to the dominant body profile
    per REGION + CATEGORY + SIZE_ID (highest DEMAND_WEIGHT).
    FACT_SALES has no GENDER column — this is the intelligent attribution.
    """
    sql = f"""
    WITH dominant_profile AS (
        SELECT
            REGION_ID,
            CATEGORY,
            SIZE_ID,
            BODY_PROFILE_ID,
            ROW_NUMBER() OVER (
                PARTITION BY REGION_ID, CATEGORY, SIZE_ID
                ORDER BY DEMAND_WEIGHT DESC
            ) AS rn
        FROM {SCHEMA}.SIZE_DEMAND_BY_BODY_PROFILE
    )
    SELECT
        dd.DATE,
        dd.MONTH,
        dd.SEASON,
        dsz.SIZE,
        fs.QTY_SOLD,
        COALESCE(bp.GENDER, 'Unknown') AS GENDER
    FROM {SCHEMA}.FACT_SALES     fs
    JOIN {SCHEMA}.DIM_STORES     ds   ON fs.STORE_ID   = ds.STORE_ID
    JOIN {SCHEMA}.DIM_PRODUCT    dp   ON fs.PRODUCT_ID = dp.PRODUCT_ID
    JOIN {SCHEMA}.DIM_DATE       dd   ON fs.DATE_KEY   = dd.DATE_KEY
    JOIN {SCHEMA}.DIM_SIZE       dsz  ON fs.SIZE_ID    = dsz.SIZE_ID
    LEFT JOIN dominant_profile   dpr
        ON  ds.REGION_ID    = dpr.REGION_ID
        AND dp.CATEGORY     = dpr.CATEGORY
        AND fs.SIZE_ID      = dpr.SIZE_ID
        AND dpr.rn          = 1
    LEFT JOIN {SCHEMA}.DIM_BODY_PROFILE bp ON dpr.BODY_PROFILE_ID = bp.BODY_PROFILE_ID
    WHERE fs.STORE_ID  = '{store_id}'
      AND dp.CATEGORY  = '{category}'
      AND dd.DATE BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def load_inventory(store_id: str, category: str) -> pd.DataFrame:
    sql = f"""
    SELECT
        dsz.SIZE,
        dsz.SIZE_ID,
        SUM(fi.STOCK_QTY) AS STOCK_QTY
    FROM {SCHEMA}.FACT_INVENTORY  fi
    JOIN {SCHEMA}.DIM_PRODUCT     dp  ON fi.PRODUCT_ID = dp.PRODUCT_ID
    JOIN {SCHEMA}.DIM_SIZE        dsz ON fi.SIZE_ID    = dsz.SIZE_ID
    WHERE fi.STORE_ID  = '{store_id}'
      AND dp.CATEGORY  = '{category}'
    GROUP BY dsz.SIZE, dsz.SIZE_ID
    ORDER BY dsz.SIZE_ID
    """
    return session.sql(sql).to_pandas()


def load_returns_sentiment(store_id: str, category: str) -> pd.DataFrame:
    """
    Calls SNOWFLAKE.CORTEX.SENTIMENT() on each unique RETURN_REASON.
    Uses a CTE so Cortex runs once per distinct reason, not per row.
    """
    sql = f"""
    WITH base AS (
        SELECT
            fr.RETURN_REASON,
            dsz.SIZE,
            COUNT(*) AS RETURN_COUNT
        FROM {SCHEMA}.FACT_RETURNS  fr
        JOIN {SCHEMA}.DIM_PRODUCT   dp  ON fr.PRODUCT_ID = dp.PRODUCT_ID
        JOIN {SCHEMA}.DIM_SIZE      dsz ON fr.SIZE_ID    = dsz.SIZE_ID
        WHERE fr.STORE_ID  = '{store_id}'
          AND dp.CATEGORY  = '{category}'
        GROUP BY fr.RETURN_REASON, dsz.SIZE
    ),
    with_sentiment AS (
        SELECT
            RETURN_REASON,
            SIZE,
            RETURN_COUNT,
            SNOWFLAKE.CORTEX.SENTIMENT(RETURN_REASON) AS SENTIMENT_SCORE
        FROM base
    )
    SELECT * FROM with_sentiment
    ORDER BY RETURN_COUNT DESC
    LIMIT 20
    """
    return session.sql(sql).to_pandas()


# ─── CORTEX COMPLETE — AI NARRATIVE ────────────────────────────────────────────

def get_ai_recommendation(
    store_name: str,
    region_name: str,
    category: str,
    assortment_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    top_sizes: list,
    clothes_space: int
) -> str:
    assortment_summary = assortment_df[["GENDER", "SIZE", "ALLOCATED_UNITS"]].to_string(index=False)

    if not sentiment_df.empty:
        sentiment_summary = sentiment_df[
            ["RETURN_REASON", "SENTIMENT_SCORE", "SIZE", "RETURN_COUNT"]
        ].head(5).to_string(index=False)
    else:
        sentiment_summary = "No returns data available for this store/category."

    top_sizes_str = ", ".join(top_sizes) if top_sizes else "N/A"

    prompt = f"""
You are a senior retail merchandise planner AI assistant specialised in fashion retail assortment optimisation.
Based on the data below, write a concise, professional, and actionable assortment buying brief for the store buyer.

Store: {store_name}
Region: {region_name}
Category: {category}
Total Units to Plan: {clothes_space}

Recommended Assortment Plan (derived from regional body profile distribution):
{assortment_summary}

Top 3 Historically Selling Sizes: {top_sizes_str}

Customer Return Sentiment Analysis (fit-related returns scored by Cortex AI):
{sentiment_summary}

Write the brief with the following structure:
1. REGIONAL PROFILE INSIGHT: Key body profile characteristics driving this assortment.
2. SIZE PRIORITIES: Which sizes to prioritise and why, referencing the allocation data.
3. FIT & RETURNS FLAGS: Any fit issues surfaced in the sentiment data that need attention.
4. BUYING RECOMMENDATIONS: 2-3 specific, immediately actionable recommendations.

Keep the total response under 220 words. Be direct, confident, and data-driven.
Do not repeat the raw numbers back — interpret them.
""".strip()

    # Escape single quotes for Snowflake SQL
    escaped_prompt = prompt.replace("'", "''")

    sql = f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'snowflake-arctic',
        '{escaped_prompt}'
    ) AS RECOMMENDATION
    """
    try:
        result = session.sql(sql).to_pandas()
        return result["RECOMMENDATION"].iloc[0]
    except Exception as e:
        return f"Unable to generate AI recommendation. Error: {str(e)}"


# ─── LARGEST REMAINDER ALLOCATION ALGORITHM ───────────────────────────────────

def allocate_units(df: pd.DataFrame, ratio_col: str, total_units: int) -> pd.DataFrame:
    """
    Largest Remainder Method — guarantees the sum of ALLOCATED_UNITS
    always equals total_units exactly. Fixes the .round(0) bug in the
    original app which frequently produced totals off by ±1 or more.
    """
    df = df.copy()
    df["RAW"]       = df[ratio_col] * total_units
    df["ALLOCATED"] = df["RAW"].apply(math.floor).astype(int)
    df["REMAINDER"] = df["RAW"] - df["ALLOCATED"]
    leftover        = total_units - df["ALLOCATED"].sum()
    if leftover > 0:
        top_idx = df["REMAINDER"].nlargest(leftover).index
        df.loc[top_idx, "ALLOCATED"] += 1
    return df


def gender_label(gender_str: str) -> str:
    """Prepend gender symbol to a gender string for display in KPI tiles."""
    g = str(gender_str).strip().lower()
    if g in ("male", "men", "man"):
        return f"♂ {gender_str}"
    elif g in ("female", "women", "woman"):
        return f"♀ {gender_str}"
    return gender_str



regions_df          = load_regions()
stores_df           = load_stores()
categories_df       = load_categories()
size_map_df         = load_size_map()
region_body_dist_df = load_region_body_dist()
size_demand_df      = load_size_demand()
body_profiles_df    = load_body_profiles()
date_bounds_df      = load_date_bounds()

min_date = date_bounds_df["MIN_DATE"].iloc[0]
max_date = date_bounds_df["MAX_DATE"].iloc[0]


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ RAIE")
    st.markdown("---")

    # ── Navigation — sidebar radio gives Python full active-tab awareness ────
    TAB_OPTIONS = [
        "📊  Current Data",
        "🔮  Future Predictions",
        "🧠  Assortment Insights",
        "🤖  AI Advisor",
    ]
    active_tab = st.radio(
        "Navigate",
        TAB_OPTIONS,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filters**")

    # Region
    region_map           = dict(zip(regions_df["REGION_NAME"], regions_df["REGION_ID"]))
    selected_region_name = st.selectbox("📍 Region", list(region_map.keys()))
    selected_region      = region_map[selected_region_name]

    # Store (filtered by region)
    region_stores        = stores_df[stores_df["REGION_ID"] == selected_region]
    store_map            = dict(zip(region_stores["STORE_NAME"], region_stores["STORE_ID"]))
    selected_store_name  = st.selectbox("🏪 Store", list(store_map.keys()))
    selected_store       = store_map[selected_store_name]

    # Category
    selected_category = st.selectbox("👗 Category", categories_df["CATEGORY"].tolist())

    # Date range — active only on Tab 1, greyed out (disabled) on all others
    is_tab1          = (active_tab == "📊  Current Data")
    date_range       = st.date_input(
        "📅 Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        disabled=not is_tab1,
        help="Applies to Current Data tab only. Greyed out on other tabs."
    )
    if not is_tab1:
        st.caption("📅 Date filter applies to Tab 1 only.")

    st.markdown("---")

    # Open-to-Buy — retail term for planned intake quantity, used in Tab 3
    clothes_space = st.number_input(
        "🛒 Open-to-Buy Units",
        min_value=10,
        max_value=50000,
        value=100,
        step=10,
        help="Total planned purchase units for this store. The Assortment Insights tab distributes exactly this quantity across sizes and genders based on regional body profiles."
    )

    # Store info card
    st.markdown("---")
    store_row = region_stores[region_stores["STORE_ID"] == selected_store]
    if not store_row.empty:
        city       = store_row["CITY"].values[0]
        store_type = store_row["STORE_TYPE"].values[0]
        st.markdown(f"""
        <div class="store-info-card">
            <strong>{selected_store_name}</strong><br>
            📌 {city} &nbsp;|&nbsp; 🏬 {store_type}<br>
            🌐 {selected_region_name}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by Snowflake Cortex AI")


# ─── PAGE HEADER ───────────────────────────────────────────────────────────────
st.title("🛍️ Retail Assortment Intelligence Engine")
st.caption(
    f"**{selected_store_name}** · {selected_region_name} · {selected_category} "
    f"· Powered by Snowflake Cortex"
)
st.markdown("---")

# Date range guard (only matters on Tab 1)
if is_tab1:
    if len(date_range) != 2:
        st.warning("⚠️ Please select a valid start and end date in the sidebar.")
        st.stop()
    start_date, end_date = date_range
else:
    # Use full range as safe defaults for any queries that need dates on other tabs
    start_date, end_date = min_date, max_date


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CURRENT DATA
# ═══════════════════════════════════════════════════════════════════════════════
if active_tab == "📊  Current Data":

    # ── Size / month / color helpers ─────────────────────────────────────────
    SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
    MONTH_ORDER = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    GENDER_COLORS = {
        "Male": "#4A90D9", "Men": "#4A90D9",
        "Female": "#E8607A", "Women": "#E8607A",
    }

    # ── Section header helper ─────────────────────────────────────────────────
    def section_banner(icon: str, title: str, subtitle: str,
                       bg: str = "#f0f4ff", border: str = "#1a73e8",
                       title_color: str = "#1a1a2e", sub_color: str = "#5a6a85"):
        return f"""
        <div style="background:{bg};border-left:5px solid {border};
                    border-radius:0 10px 10px 0;padding:14px 20px;
                    margin:24px 0 18px 0;">
          <div style="font-size:20px;font-weight:700;color:{title_color};
                      display:flex;align-items:center;gap:10px;">
            <span style="font-size:22px;">{icon}</span>{title}
          </div>
          <div style="font-size:13px;color:{sub_color};margin-top:3px;">{subtitle}</div>
        </div>"""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — SALES PERFORMANCE
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(section_banner(
        "📊", "Sales Performance",
        f"{selected_store_name} · {selected_region_name} · {selected_category} "
        f"· {start_date} → {end_date}",
        bg="#f0f4ff", border="#1a73e8"
    ), unsafe_allow_html=True)

    with st.spinner("Loading sales data..."):
        sales_df = load_sales(selected_store, selected_category, start_date, end_date)

    if sales_df.empty:
        st.warning("No sales data found for the selected filters. Try adjusting the date range or category.")
        st.stop()

    present_sizes  = [s for s in SIZE_ORDER if s in sales_df["SIZE"].unique()]
    extra_sizes    = [s for s in sales_df["SIZE"].unique() if s not in SIZE_ORDER]
    ordered_sizes  = present_sizes + extra_sizes
    present_months = [m for m in MONTH_ORDER if m in sales_df["MONTH"].unique()]
    gender_vals    = sales_df["GENDER"].unique().tolist()
    color_domain   = gender_vals
    color_range    = [GENDER_COLORS.get(g, "#7c3aed") for g in gender_vals]

    # ── Sales KPIs ────────────────────────────────────────────────────────────
    total_units     = int(sales_df["QTY_SOLD"].sum())
    unique_sizes    = len(ordered_sizes)
    top_size        = sales_df.groupby("SIZE")["QTY_SOLD"].sum().idxmax()
    top_gender      = (
        sales_df.groupby("GENDER")["QTY_SOLD"].sum().idxmax()
        if "GENDER" in sales_df.columns and sales_df["GENDER"].notna().any() else "N/A"
    )
    top_month       = (
        sales_df.groupby("MONTH")["QTY_SOLD"].sum().idxmax()
        if "MONTH" in sales_df.columns else "N/A"
    )
    seasons_covered = sales_df["SEASON"].nunique() if "SEASON" in sales_df.columns else "N/A"

    k1, k2, k3, k4, k5, k6 = st.columns([1.2, 1, 1, 1.3, 1.1, 1])
    k1.metric("Total Units Sold",  f"{total_units:,}")
    k2.metric("Size Variants",     unique_sizes)
    k3.metric("Top Selling Size",  top_size)
    k4.metric("Dominant Gender",   gender_label(top_gender))
    k5.metric("Peak Month",        top_month)
    k6.metric("Seasons Covered",   seasons_covered)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Units by Gender | Units by Size ───────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Units Sold by Gender**")
        gender_agg   = sales_df.groupby("GENDER")["QTY_SOLD"].sum().reset_index()
        gender_chart = (
            alt.Chart(gender_agg)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("GENDER:N", title="Gender", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("QTY_SOLD:Q", title="Units Sold"),
                color=alt.Color("GENDER:N",
                    scale=alt.Scale(domain=color_domain, range=color_range),
                    legend=None),
                tooltip=[alt.Tooltip("GENDER:N", title="Gender"),
                         alt.Tooltip("QTY_SOLD:Q", title="Units Sold", format=",")]
            )
            .properties(height=280)
            .configure_axis(grid=False).configure_view(strokeWidth=0)
        )
        st.altair_chart(gender_chart, use_container_width=True)

    with col2:
        st.markdown("**Units Sold by Size** *(XS → XL)*")
        size_agg = (
            sales_df.groupby("SIZE")["QTY_SOLD"].sum()
            .reindex(ordered_sizes).reset_index()
            .rename(columns={"index": "SIZE"}).dropna()
        )
        size_chart = (
            alt.Chart(size_agg)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("SIZE:N", title="Size", sort=ordered_sizes,
                         axis=alt.Axis(labelAngle=0)),
                y=alt.Y("QTY_SOLD:Q", title="Units Sold"),
                color=alt.value("#4A90D9"),
                tooltip=[alt.Tooltip("SIZE:N", title="Size"),
                         alt.Tooltip("QTY_SOLD:Q", title="Units Sold", format=",")]
            )
            .properties(height=280)
            .configure_axis(grid=False).configure_view(strokeWidth=0)
        )
        st.altair_chart(size_chart, use_container_width=True)

    # ── Gender × Size breakdown ───────────────────────────────────────────────
    st.markdown(f"**{selected_category} — Gender × Size Breakdown** *(♂ Male vs ♀ Female per size)*")
    st.caption("Stacked bars show how units split between genders within each size, ordered XS → XL.")
    gs_agg = (
        sales_df.groupby(["SIZE","GENDER"])["QTY_SOLD"].sum().reset_index()
    )
    gs_agg = gs_agg[gs_agg["SIZE"].isin(ordered_sizes)]
    gs_chart = (
        alt.Chart(gs_agg)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("SIZE:N", title="Size", sort=ordered_sizes,
                     axis=alt.Axis(labelAngle=0)),
            y=alt.Y("QTY_SOLD:Q", title="Units Sold"),
            color=alt.Color("GENDER:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Gender", orient="top")),
            order=alt.Order("GENDER:N", sort="ascending"),
            tooltip=[alt.Tooltip("GENDER:N", title="Gender"),
                     alt.Tooltip("SIZE:N", title="Size"),
                     alt.Tooltip("QTY_SOLD:Q", title="Units Sold", format=",")]
        )
        .properties(height=300)
        .configure_axis(grid=False).configure_view(strokeWidth=0)
    )
    st.altair_chart(gs_chart, use_container_width=True)

    # ── Size Mix Trend by Month ────────────────────────────────────────────────
    st.markdown("**Size Mix Trend by Month** *(which sizes sell in which months)*")
    st.caption("Stacked bars reveal size composition of sales each month, ordered January → December.")
    month_size_agg = (
        sales_df.groupby(["MONTH","SIZE"])["QTY_SOLD"].sum().reset_index()
    )
    month_size_agg = month_size_agg[
        month_size_agg["MONTH"].isin(present_months) &
        month_size_agg["SIZE"].isin(ordered_sizes)
    ]
    SIZE_COLORS      = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B","#44BBA4","#E94F37"]
    size_color_range = SIZE_COLORS[:len(ordered_sizes)]
    size_trend_chart = (
        alt.Chart(month_size_agg).mark_bar()
        .encode(
            x=alt.X("MONTH:N", title="Month", sort=present_months,
                     axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("QTY_SOLD:Q", title="Units Sold", stack="zero"),
            color=alt.Color("SIZE:N", sort=ordered_sizes,
                scale=alt.Scale(domain=ordered_sizes, range=size_color_range),
                legend=alt.Legend(title="Size", orient="right")),
            order=alt.Order("SIZE:N", sort="ascending"),
            tooltip=[alt.Tooltip("MONTH:N", title="Month"),
                     alt.Tooltip("SIZE:N", title="Size"),
                     alt.Tooltip("QTY_SOLD:Q", title="Units Sold", format=",")]
        )
        .properties(height=320)
        .configure_axis(grid=False).configure_view(strokeWidth=0)
    )
    st.altair_chart(size_trend_chart, use_container_width=True)

    # ── Units by Season ────────────────────────────────────────────────────────
    if "SEASON" in sales_df.columns:
        st.markdown("**Units by Season**")
        season_agg   = sales_df.groupby("SEASON")["QTY_SOLD"].sum().reset_index()
        season_chart = (
            alt.Chart(season_agg)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("SEASON:N", title="Season", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("QTY_SOLD:Q", title="Units Sold"),
                color=alt.value("#44BBA4"),
                tooltip=[alt.Tooltip("SEASON:N", title="Season"),
                         alt.Tooltip("QTY_SOLD:Q", title="Units Sold", format=",")]
            )
            .properties(height=260)
            .configure_axis(grid=False).configure_view(strokeWidth=0)
        )
        st.altair_chart(season_chart, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — REGIONAL DEMOGRAPHICS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(section_banner(
        "🧬", "Regional Body Profile Intelligence",
        f"Population body profile distribution for {selected_region_name} — "
        "the demographic foundation powering your size assortment decisions.",
        bg="#f0fdf4", border="#059669"
    ), unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;
                padding:13px 18px;font-size:13px;color:#065f46;margin-bottom:18px;">
        <strong>Why does this matter?</strong> The size assortment recommended by RAIE
        is directly derived from the body profile distribution below. Regions with a
        higher concentration of taller or heavier body profiles receive a different
        size mix to regions with petite or lighter profiles — ensuring the right sizes
        are in the right stores at the right time.
    </div>
    """, unsafe_allow_html=True)

    # ── Compute regional demographic aggregates ───────────────────────────────
    # Both dataframes already in memory — no extra SQL
    region_demo = (
        region_body_dist_df[region_body_dist_df["REGION_ID"] == selected_region]
        .merge(body_profiles_df, on="BODY_PROFILE_ID")
    )

    if region_demo.empty:
        st.info("No demographic data available for the selected region.")
    else:
        # ── Demographic KPIs ──────────────────────────────────────────────────
        dom_height = (
            region_demo.groupby("HEIGHT_CATEGORY")["POPULATION_PCT"].sum().idxmax()
        )
        dom_weight = (
            region_demo.groupby("WEIGHT_RANGE_KG")["POPULATION_PCT"].sum().idxmax()
        )
        dom_body_type = (
            region_demo.groupby("BODY_TYPE")["POPULATION_PCT"].sum().idxmax()
        )
        dom_age_group = (
            region_demo.groupby("AGE_GROUP")["POPULATION_PCT"].sum().idxmax()
        )
        n_profiles = region_demo["BODY_PROFILE_ID"].nunique()

        dk1, dk2, dk3, dk4, dk5 = st.columns([1.2, 1.2, 1.2, 1.2, 1])
        dk1.metric("📏 Dominant Height",    dom_height,
                   help="Height category representing the largest share of this region's population.")
        dk2.metric("⚖️ Dominant Weight",    dom_weight,
                   help="Weight range category representing the largest share of this region's population.")
        dk3.metric("🏃 Dominant Body Type", dom_body_type,
                   help="Most prevalent body type driving size recommendations for this region.")
        dk4.metric("👥 Peak Age Group",     dom_age_group,
                   help="Age bracket with the highest population concentration in this region.")
        dk5.metric("🔬 Body Profiles",      n_profiles,
                   help="Number of distinct body profile segments active in this region.")

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        # ── Chart 1: Height Category Distribution ─────────────────────────────
        col_h, col_w = st.columns(2)

        height_agg = (
            region_demo.groupby(["HEIGHT_CATEGORY", "GENDER"])["POPULATION_PCT"]
            .sum().reset_index()
        )
        # Canonical height order: Petite → Short → Average → Tall → Very Tall
        HEIGHT_ORDER  = ["Petite", "Short", "Below Average", "Average",
                          "Above Average", "Tall", "Very Tall"]
        h_present     = [h for h in HEIGHT_ORDER if h in height_agg["HEIGHT_CATEGORY"].unique()]
        h_extra       = [h for h in height_agg["HEIGHT_CATEGORY"].unique() if h not in HEIGHT_ORDER]
        h_order       = h_present + h_extra

        dem_gender_vals  = region_demo["GENDER"].unique().tolist()
        dem_color_range  = [GENDER_COLORS.get(g, "#7c3aed") for g in dem_gender_vals]

        with col_h:
            st.markdown("**📏 Height Profile Distribution**")
            st.caption(f"How {selected_region_name}'s population is distributed across height categories by gender")

            height_chart = (
                alt.Chart(height_agg).mark_bar()
                .encode(
                    x=alt.X("HEIGHT_CATEGORY:N", title="Height Category",
                             sort=h_order, axis=alt.Axis(labelAngle=-20)),
                    y=alt.Y("POPULATION_PCT:Q", title="Population %",
                             stack="zero",
                             axis=alt.Axis(format=".1%")),
                    color=alt.Color("GENDER:N",
                        scale=alt.Scale(domain=dem_gender_vals, range=dem_color_range),
                        legend=alt.Legend(title="Gender", orient="top")),
                    order=alt.Order("GENDER:N", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("GENDER:N",        title="Gender"),
                        alt.Tooltip("HEIGHT_CATEGORY:N", title="Height Category"),
                        alt.Tooltip("POPULATION_PCT:Q",  title="Population %", format=".1%"),
                    ]
                )
                .properties(height=300)
                .configure_axis(grid=False).configure_view(strokeWidth=0)
            )
            st.altair_chart(height_chart, use_container_width=True)

        # ── Chart 2: Weight Range Distribution ───────────────────────────────
        weight_agg = (
            region_demo.groupby(["WEIGHT_RANGE_KG", "GENDER"])["POPULATION_PCT"]
            .sum().reset_index()
        )
        # Sort weight ranges numerically by extracting first number
        import re as _re
        def _w_sort_key(s):
            nums = _re.findall(r'\d+', str(s))
            return int(nums[0]) if nums else 9999
        w_unique  = weight_agg["WEIGHT_RANGE_KG"].unique().tolist()
        w_order   = sorted(w_unique, key=_w_sort_key)

        with col_w:
            st.markdown("**⚖️ Weight Range Distribution**")
            st.caption(f"Population concentration across weight bands in {selected_region_name} by gender")

            weight_chart = (
                alt.Chart(weight_agg).mark_bar()
                .encode(
                    y=alt.Y("WEIGHT_RANGE_KG:N", title="Weight Range (kg)",
                             sort=w_order, axis=alt.Axis(labelAngle=0)),
                    x=alt.X("POPULATION_PCT:Q", title="Population %",
                             stack="zero",
                             axis=alt.Axis(format=".1%")),
                    color=alt.Color("GENDER:N",
                        scale=alt.Scale(domain=dem_gender_vals, range=dem_color_range),
                        legend=alt.Legend(title="Gender", orient="top")),
                    order=alt.Order("GENDER:N", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("GENDER:N",        title="Gender"),
                        alt.Tooltip("WEIGHT_RANGE_KG:N", title="Weight Range"),
                        alt.Tooltip("POPULATION_PCT:Q",  title="Population %", format=".1%"),
                    ]
                )
                .properties(height=300)
                .configure_axis(grid=False).configure_view(strokeWidth=0)
            )
            st.altair_chart(weight_chart, use_container_width=True)

        # ── Chart 3: Body Type breakdown + Age Group split ────────────────────
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        col_bt, col_ag = st.columns(2)

        body_type_agg = (
            region_demo.groupby(["BODY_TYPE", "GENDER"])["POPULATION_PCT"]
            .sum().reset_index()
        )
        bt_totals = body_type_agg.groupby("BODY_TYPE")["POPULATION_PCT"].sum()
        bt_order  = bt_totals.sort_values(ascending=False).index.tolist()

        with col_bt:
            st.markdown("**🏃 Body Type Breakdown**")
            st.caption("Distribution of body types informing fit and size allocation strategy")
            bt_chart = (
                alt.Chart(body_type_agg).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("BODY_TYPE:N", title="Body Type",
                             sort=bt_order, axis=alt.Axis(labelAngle=-20)),
                    y=alt.Y("POPULATION_PCT:Q", title="Population %",
                             stack="zero", axis=alt.Axis(format=".1%")),
                    color=alt.Color("GENDER:N",
                        scale=alt.Scale(domain=dem_gender_vals, range=dem_color_range),
                        legend=alt.Legend(title="Gender", orient="top")),
                    order=alt.Order("GENDER:N", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("GENDER:N",       title="Gender"),
                        alt.Tooltip("BODY_TYPE:N",     title="Body Type"),
                        alt.Tooltip("POPULATION_PCT:Q", title="Population %", format=".1%"),
                    ]
                )
                .properties(height=280)
                .configure_axis(grid=False).configure_view(strokeWidth=0)
            )
            st.altair_chart(bt_chart, use_container_width=True)

        age_agg = (
            region_demo.groupby(["AGE_GROUP", "GENDER"])["POPULATION_PCT"]
            .sum().reset_index()
        )
        AGE_ORDER  = ["Teen","Young Adult","Adult","Middle-Aged","Senior","Elderly"]
        ag_present = [a for a in AGE_ORDER if a in age_agg["AGE_GROUP"].unique()]
        ag_extra   = [a for a in age_agg["AGE_GROUP"].unique() if a not in AGE_ORDER]
        ag_order   = ag_present + ag_extra

        with col_ag:
            st.markdown("**👥 Age Group Distribution**")
            st.caption("Age profile of the regional customer base — shapes seasonal and style preferences")
            age_chart = (
                alt.Chart(age_agg).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("AGE_GROUP:N", title="Age Group",
                             sort=ag_order, axis=alt.Axis(labelAngle=-20)),
                    y=alt.Y("POPULATION_PCT:Q", title="Population %",
                             stack="zero", axis=alt.Axis(format=".1%")),
                    color=alt.Color("GENDER:N",
                        scale=alt.Scale(domain=dem_gender_vals, range=dem_color_range),
                        legend=alt.Legend(title="Gender", orient="top")),
                    order=alt.Order("GENDER:N", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("GENDER:N",       title="Gender"),
                        alt.Tooltip("AGE_GROUP:N",     title="Age Group"),
                        alt.Tooltip("POPULATION_PCT:Q", title="Population %", format=".1%"),
                    ]
                )
                .properties(height=280)
                .configure_axis(grid=False).configure_view(strokeWidth=0)
            )
            st.altair_chart(age_chart, use_container_width=True)

        # ── Demographic insight callout ────────────────────────────────────────
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                    padding:16px 20px;margin-top:10px;font-size:13px;color:#166534;">
            <strong>🔍 {selected_region_name} Profile Summary:</strong>
            The dominant customer in this region is
            <strong>{dom_height}</strong> in height,
            in the <strong>{dom_weight}</strong> weight range,
            with a <strong>{dom_body_type}</strong> body type,
            primarily in the <strong>{dom_age_group}</strong> age bracket.
            These signals directly inform the size ratios recommended in the
            <em>Assortment Insights</em> tab.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FUTURE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🔮  Future Predictions":

    # ── Size ordering (consistent with Tab 1) ────────────────────────────────
    SIZE_ORDER    = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
    GENDER_COLORS = {
        "Male": "#4A90D9", "Men": "#4A90D9",
        "Female": "#E8607A", "Women": "#E8607A",
    }

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">📈 Demand Outlook — {selected_category}</div>',
        unsafe_allow_html=True
    )

    # ── Controls row — horizon + store/region toggle ──────────────────────────
    ctl1, ctl2, ctl3 = st.columns([2, 1.5, 2])

    with ctl1:
        horizon_label = st.select_slider(
            "🗓️ Planning Horizon",
            options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
            value="3 Months",
            help="How far ahead do you want to plan?"
        )

    with ctl2:
        pred_level = st.radio(
            "📍 View Level",
            ["Store", "Region"],
            horizontal=True,
            help="Toggle between store-level and region-level demand forecast."
        )

    # Dynamic scope label — now pred_level is defined
    scope_label_header = selected_store_name if pred_level == "Store" else selected_region_name
    st.caption(
        f"📍 Viewing: **{scope_label_header}** · {selected_category} · "
        f"Horizon: **{horizon_label}** · "
        "Powered by regional body profile demand signals"
    )

    # Weeks mapping for each horizon
    HORIZON_WEEKS = {
        "1 Month":  4,
        "3 Months": 13,
        "6 Months": 26,
        "1 Year":   52,
        "2 Years":  104,
    }
    horizon_weeks = HORIZON_WEEKS[horizon_label]

    st.markdown("---")

    # ── Compute historical weekly run rate ────────────────────────────────────
    # This anchors forecasted units to real observed velocity from FACT_SALES
    if pred_level == "Store":
        weekly_sql = f"""
        SELECT
            dp.CATEGORY,
            AVG(weekly_qty) AS AVG_WEEKLY_UNITS
        FROM (
            SELECT
                fs.STORE_ID,
                dp.CATEGORY,
                DATE_TRUNC('week', dd.DATE) AS WEEK_START,
                SUM(fs.QTY_SOLD)            AS weekly_qty
            FROM {SCHEMA}.FACT_SALES  fs
            JOIN {SCHEMA}.DIM_DATE    dd ON fs.DATE_KEY   = dd.DATE_KEY
            JOIN {SCHEMA}.DIM_PRODUCT dp ON fs.PRODUCT_ID = dp.PRODUCT_ID
            WHERE fs.STORE_ID  = '{selected_store}'
              AND dp.CATEGORY  = '{selected_category}'
            GROUP BY fs.STORE_ID, dp.CATEGORY, WEEK_START
        ) w
        JOIN {SCHEMA}.DIM_PRODUCT dp ON w.CATEGORY = dp.CATEGORY
        GROUP BY dp.CATEGORY
        """
        scope_label = selected_store_name
    else:
        weekly_sql = f"""
        SELECT
            dp.CATEGORY,
            AVG(weekly_qty) AS AVG_WEEKLY_UNITS
        FROM (
            SELECT
                ds.REGION_ID,
                dp.CATEGORY,
                DATE_TRUNC('week', dd.DATE) AS WEEK_START,
                SUM(fs.QTY_SOLD)            AS weekly_qty
            FROM {SCHEMA}.FACT_SALES  fs
            JOIN {SCHEMA}.DIM_DATE    dd ON fs.DATE_KEY   = dd.DATE_KEY
            JOIN {SCHEMA}.DIM_PRODUCT dp ON fs.PRODUCT_ID = dp.PRODUCT_ID
            JOIN {SCHEMA}.DIM_STORES  ds ON fs.STORE_ID   = ds.STORE_ID
            WHERE ds.REGION_ID = '{selected_region}'
              AND dp.CATEGORY  = '{selected_category}'
            GROUP BY ds.REGION_ID, dp.CATEGORY, WEEK_START
        ) w
        JOIN {SCHEMA}.DIM_PRODUCT dp ON w.CATEGORY = dp.CATEGORY
        GROUP BY dp.CATEGORY
        """
        scope_label = selected_region_name

    try:
        weekly_df      = session.sql(weekly_sql).to_pandas()
        avg_weekly_units = float(weekly_df["AVG_WEEKLY_UNITS"].iloc[0]) if not weekly_df.empty else 50.0
    except Exception:
        avg_weekly_units = 50.0   # safe fallback

    total_projected_units = round(avg_weekly_units * horizon_weeks)

    # ── Weighted demand ratios (demographic engine) ───────────────────────────
    demand_df = size_demand_df[
        (size_demand_df["REGION_ID"] == selected_region) &
        (size_demand_df["CATEGORY"]  == selected_category)
    ]
    pred_merged = (
        demand_df
        .merge(
            region_body_dist_df[region_body_dist_df["REGION_ID"] == selected_region][
                ["BODY_PROFILE_ID", "POPULATION_PCT"]
            ],
            on="BODY_PROFILE_ID"
        )
        .merge(body_profiles_df[["BODY_PROFILE_ID", "GENDER"]], on="BODY_PROFILE_ID")
        .merge(size_map_df, on="SIZE_ID")
    )

    if pred_merged.empty:
        st.warning("No demand data available for the selected region and category.")
        st.stop()

    pred_merged["WEIGHTED"] = pred_merged["DEMAND_WEIGHT"] * pred_merged["POPULATION_PCT"]

    size_pred = (
        pred_merged
        .groupby(["SIZE", "SIZE_ID", "GENDER"])["WEIGHTED"]
        .sum().reset_index()
    )
    total_w            = size_pred["WEIGHTED"].sum()
    size_pred["RATIO"] = size_pred["WEIGHTED"] / total_w

    # Scale ratios → projected units using historical weekly run rate
    size_pred = allocate_units(size_pred, "RATIO", total_projected_units)
    size_pred = size_pred.rename(columns={"ALLOCATED": "PROJ_UNITS"})

    # Enforce canonical size order
    present_sizes  = [s for s in SIZE_ORDER if s in size_pred["SIZE"].unique()]
    extra_sizes    = [s for s in size_pred["SIZE"].unique() if s not in SIZE_ORDER]
    ordered_sizes  = present_sizes + extra_sizes
    size_pred      = size_pred[size_pred["SIZE"].isin(ordered_sizes)]

    # Aggregate for size-level summaries
    size_summary = (
        size_pred.groupby("SIZE")
        .agg(PROJ_UNITS=("PROJ_UNITS", "sum"), RATIO=("RATIO", "sum"))
        .reindex(ordered_sizes).dropna().reset_index()
    )

    # ── Business KPIs ─────────────────────────────────────────────────────────
    top_size_row      = size_summary.loc[size_summary["PROJ_UNITS"].idxmax()]
    top_size_name     = top_size_row["SIZE"]
    top_size_units    = int(top_size_row["PROJ_UNITS"])

    bottom_size_row   = size_summary.loc[size_summary["PROJ_UNITS"].idxmin()]
    slowest_size      = bottom_size_row["SIZE"]

    top_gender        = (
        size_pred.groupby("GENDER")["PROJ_UNITS"].sum().idxmax()
        if not size_pred.empty else "N/A"
    )
    gender_vals   = size_pred["GENDER"].unique().tolist()
    color_range_g = [GENDER_COLORS.get(g, "#7c3aed") for g in gender_vals]

    # Stockout risk: sizes where projected units > 2× current inventory
    try:
        inv_df = load_inventory(selected_store, selected_category)
        if not inv_df.empty:
            risk_df  = size_summary.merge(inv_df[["SIZE", "STOCK_QTY"]], on="SIZE", how="left").fillna({"STOCK_QTY": 0})
            risk_df["STOCK_QTY"] = risk_df["STOCK_QTY"].astype(int)
            risk_df["COVER_WKS"] = (risk_df["STOCK_QTY"] / (risk_df["PROJ_UNITS"] / horizon_weeks)).round(1)
            at_risk_sizes = risk_df[risk_df["COVER_WKS"] < (horizon_weeks * 0.3)]["SIZE"].tolist()
            stockout_flag = ", ".join(at_risk_sizes) if at_risk_sizes else "None ✅"
        else:
            stockout_flag = "No inventory data"
            risk_df = pd.DataFrame()
    except Exception:
        stockout_flag = "N/A"
        risk_df = pd.DataFrame()

    # ── KPI tiles ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns([1.3, 1.2, 1.2, 1.3, 1.3])
    k1.metric(
        f"📦 Projected Units ({horizon_label})",
        f"{total_projected_units:,}",
        help=f"Total units expected to sell across all sizes over {horizon_label}, based on historical weekly sales rate of {avg_weekly_units:.0f} units/week."
    )
    k2.metric(
        "🏆 Top Size to Stock",
        top_size_name,
        delta=f"{top_size_units:,} units projected",
        help="The single size expected to drive the most demand in this period."
    )
    k3.metric(
        "👥 Dominant Gender",
        gender_label(top_gender),
        help="Gender segment representing the largest share of projected demand."
    )
    k4.metric(
        "⚠️ Stockout Risk",
        stockout_flag,
        help="Sizes where current stock covers less than 30% of projected demand horizon."
    )
    k5.metric(
        "🐢 Slowest Moving Size",
        slowest_size,
        help="Size with lowest projected demand — consider reducing intake for this period."
    )

    st.markdown("---")

    # ── Projected units chart — clean, business-readable ─────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Projected Units by Size — Next {horizon_label}**")
        st.caption(f"Based on {scope_label} demand profile · {avg_weekly_units:.0f} avg units/week historically")

        proj_chart = (
            alt.Chart(size_summary)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("SIZE:N", sort=ordered_sizes, title="Size",
                         axis=alt.Axis(labelAngle=0)),
                y=alt.Y("PROJ_UNITS:Q", title="Projected Units"),
                color=alt.condition(
                    alt.datum.SIZE == top_size_name,
                    alt.value("#1a73e8"),
                    alt.value("#93c4f4")
                ),
                tooltip=[
                    alt.Tooltip("SIZE:N",      title="Size"),
                    alt.Tooltip("PROJ_UNITS:Q", title="Projected Units", format=","),
                ]
            )
            .properties(height=300)
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(proj_chart, use_container_width=True)

    with col2:
        st.markdown(f"**Gender Split — Next {horizon_label}**")
        st.caption("How projected demand splits between Male and Female across all sizes")

        gender_proj = (
            size_pred.groupby("GENDER")["PROJ_UNITS"]
            .sum().reset_index()
        )
        gender_proj_chart = (
            alt.Chart(size_pred)
            .mark_bar()
            .encode(
                x=alt.X("SIZE:N", sort=ordered_sizes, title="Size",
                         axis=alt.Axis(labelAngle=0)),
                y=alt.Y("PROJ_UNITS:Q", title="Projected Units", stack="zero"),
                color=alt.Color(
                    "GENDER:N",
                    scale=alt.Scale(domain=gender_vals, range=color_range_g),
                    legend=alt.Legend(title="Gender", orient="top")
                ),
                order=alt.Order("GENDER:N", sort="ascending"),
                tooltip=[
                    alt.Tooltip("GENDER:N",    title="Gender"),
                    alt.Tooltip("SIZE:N",       title="Size"),
                    alt.Tooltip("PROJ_UNITS:Q", title="Projected Units", format=","),
                ]
            )
            .properties(height=300)
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(gender_proj_chart, use_container_width=True)

    # ── Stock coverage table (if inventory exists) ────────────────────────────
    if not risk_df.empty:
        st.markdown("---")
        st.markdown("### 🗂️ Stock Coverage Plan")
        st.caption(
            f"Shows how long your current inventory will last against projected demand "
            f"over the next {horizon_label}. Flag = reorder urgency."
        )

        def coverage_flag(wks):
            if wks < horizon_weeks * 0.3:   return "🔴 Reorder Now"
            elif wks < horizon_weeks * 0.6: return "🟡 Plan Reorder"
            else:                            return "🟢 Sufficient"

        risk_df["Coverage"]       = risk_df["COVER_WKS"].apply(
            lambda x: f"{x:.1f} weeks" if x < 999 else "∞"
        )
        risk_df["Action"]         = risk_df["COVER_WKS"].apply(coverage_flag)
        risk_df["Reorder Qty"]    = (risk_df["PROJ_UNITS"] - risk_df["STOCK_QTY"]).clip(lower=0).astype(int)

        st.dataframe(
            risk_df[["SIZE", "PROJ_UNITS", "STOCK_QTY", "Coverage", "Reorder Qty", "Action"]]
            .rename(columns={
                "SIZE":       "Size",
                "PROJ_UNITS": f"Projected Units ({horizon_label})",
                "STOCK_QTY":  "Current Stock",
                "Reorder Qty":"Suggested Reorder"
            }),
            use_container_width=True
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("🔴 Reorder Now",    (risk_df["Action"] == "🔴 Reorder Now").sum())
        r2.metric("🟡 Plan Reorder",   (risk_df["Action"] == "🟡 Plan Reorder").sum())
        r3.metric("🟢 Sufficient Stock",(risk_df["Action"] == "🟢 Sufficient").sum())

    # ── Plain-English AI Summary ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 What This Means For Your Buying Plan")
    st.caption("Plain-English summary generated by Snowflake Cortex AI")

    if st.button("✨ Generate Demand Summary", type="primary"):
        with st.spinner("Cortex is reading your demand signals..."):
            top3 = size_summary.nlargest(3, "PROJ_UNITS")[["SIZE","PROJ_UNITS"]].to_string(index=False)
            bot3 = size_summary.nsmallest(2, "PROJ_UNITS")[["SIZE","PROJ_UNITS"]].to_string(index=False)
            stock_note = (
                f"Stockout risk sizes: {stockout_flag}."
                if stockout_flag not in ("None ✅", "N/A", "No inventory data")
                else "All sizes appear adequately stocked for this horizon."
            )
            prompt = f"""
You are a retail buying advisor. Write a SHORT, plain-English demand summary
for a store buyer or merchandising director. No jargon, no technical terms.
Use clear sentences a non-technical person understands immediately.

Store / Scope: {scope_label}
Category: {selected_category}
Planning Period: {horizon_label}
Total Projected Units: {total_projected_units:,}
Historical Weekly Run Rate: {avg_weekly_units:.0f} units/week

Top 3 sizes by projected demand:
{top3}

Slowest 2 sizes:
{bot3}

Dominant gender: {top_gender}
{stock_note}

Write exactly 3 short paragraphs:
1. What the overall demand looks like for this period (1-2 sentences).
2. Which sizes to prioritise buying and which to buy less of (2-3 sentences).
3. One clear, specific action the buyer should take before the period starts (1-2 sentences).

Do not use bullet points. Do not mention numbers excessively — interpret them instead.
Keep total response under 150 words.
""".strip()

            escaped = prompt.replace("'", "''")
            try:
                result = session.sql(f"""
                    SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{escaped}') AS SUMMARY
                """).to_pandas()
                summary_text = result["SUMMARY"].iloc[0]
            except Exception as e:
                summary_text = f"Unable to generate summary. Error: {str(e)}"

        st.markdown(
            f'<div class="ai-narrative-box">{summary_text}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info(
            "👆 Click **Generate Demand Summary** above to get a plain-English "
            "interpretation of these demand signals from Cortex AI."
        )



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASSORTMENT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🧠  Assortment Insights":
    st.markdown(
        f'<div class="section-title">Smart Assortment Plan — {selected_store_name}</div>',
        unsafe_allow_html=True
    )
    st.caption(
        f"Distributing **{clothes_space:,} units** across sizes and genders based on "
        f"the regional body profile of **{selected_region_name}** · {selected_category}."
    )

    # ── Size ordering ─────────────────────────────────────────────────────────
    SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]

    # ── Compute assortment ────────────────────────────────────────────────────
    demand_df = size_demand_df[
        (size_demand_df["REGION_ID"] == selected_region) &
        (size_demand_df["CATEGORY"]  == selected_category)
    ]
    ins_merged = (
        demand_df
        .merge(
            region_body_dist_df[region_body_dist_df["REGION_ID"] == selected_region][
                ["BODY_PROFILE_ID", "POPULATION_PCT"]
            ], on="BODY_PROFILE_ID"
        )
        .merge(body_profiles_df[["BODY_PROFILE_ID", "GENDER"]], on="BODY_PROFILE_ID")
        .merge(size_map_df, on="SIZE_ID")
    )
    if ins_merged.empty:
        st.warning("No assortment data available for the selected filters.")
        st.stop()

    ins_merged["WEIGHTED"] = ins_merged["DEMAND_WEIGHT"] * ins_merged["POPULATION_PCT"]
    final = (
        ins_merged.groupby(["GENDER", "SIZE", "SIZE_ID"])["WEIGHTED"]
        .sum().reset_index()
    )
    final["RATIO"]         = final["WEIGHTED"] / final["WEIGHTED"].sum()
    final                  = allocate_units(final, "RATIO", clothes_space)
    final                  = final.rename(columns={"ALLOCATED": "ALLOCATED_UNITS"})
    final                  = final.sort_values(["GENDER", "SIZE_ID"]).reset_index(drop=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    biggest_row  = final.loc[final["ALLOCATED_UNITS"].idxmax()]
    male_total   = final[final["GENDER"].str.lower().isin(["male","men"])]["ALLOCATED_UNITS"].sum()
    female_total = final[final["GENDER"].str.lower().isin(["female","women"])]["ALLOCATED_UNITS"].sum()

    k1, k2, k3, k4, k5 = st.columns([1.2, 1, 1, 1, 1.3])
    k1.metric("Total Units Planned",       f"{final['ALLOCATED_UNITS'].sum():,}")
    k2.metric("♂ Male Units",              f"{male_total:,}")
    k3.metric("♀ Female Units",            f"{female_total:,}")
    k4.metric("Size Variants",             final["SIZE"].nunique())
    k5.metric("Largest Single Allocation",
              f"{biggest_row['ALLOCATED_UNITS']} units",
              delta=f"{biggest_row['GENDER']} · {biggest_row['SIZE']}")

    st.markdown("---")

    # ── Color gradient helper — professional steel-teal palette ──────────────
    # Darkest = most units required, lightest = fewest. Text flips at midpoint.
    TEAL_RAMP = [
        "#0A3D52", "#0D4F6C", "#1A6E8A", "#2E8FAD",
        "#5BAFC7", "#8ED0E0", "#C2E8F0", "#E8F7FA",
    ]
    def gradient_html_table(df_gender: pd.DataFrame, gender_label_str: str) -> str:
        """
        Renders a single-gender assortment table as HTML with a steel-teal
        color gradient: darkest row = highest units, lightest = lowest.
        Includes a TOTAL row pinned at the bottom.
        """
        df_sorted = df_gender.sort_values("ALLOCATED_UNITS", ascending=False).copy()
        df_sorted["Ratio (%)"] = (df_sorted["RATIO"] * 100).round(1)
        n          = len(df_sorted)
        ramp_steps = TEAL_RAMP[:n] if n <= len(TEAL_RAMP) else TEAL_RAMP + ["#F0FAFB"] * (n - len(TEAL_RAMP))

        rows_html = ""
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            bg      = ramp_steps[i]
            # Use white text for the 4 darkest shades, dark text for lighter
            fg      = "#FFFFFF" if i < 4 else "#0A3D52"
            rows_html += f"""
            <tr>
              <td style="background:{bg};color:{fg};padding:9px 14px;font-weight:600;
                         border-bottom:1px solid rgba(255,255,255,0.15);">{row['SIZE']}</td>
              <td style="background:{bg};color:{fg};padding:9px 14px;text-align:center;
                         font-weight:700;font-size:15px;
                         border-bottom:1px solid rgba(255,255,255,0.15);">{int(row['ALLOCATED_UNITS'])}</td>
              <td style="background:{bg};color:{fg};padding:9px 14px;text-align:center;
                         border-bottom:1px solid rgba(255,255,255,0.15);">{row['Ratio (%)']:.1f}%</td>
            </tr>"""

        total_units = int(df_sorted["ALLOCATED_UNITS"].sum())
        total_ratio = df_sorted["Ratio (%)"].sum()
        rows_html  += f"""
            <tr style="border-top:2px solid #0A3D52;">
              <td style="padding:9px 14px;font-weight:700;color:#0A3D52;background:#E8F7FA;">TOTAL</td>
              <td style="padding:9px 14px;text-align:center;font-weight:700;font-size:15px;
                         color:#0A3D52;background:#E8F7FA;">{total_units:,}</td>
              <td style="padding:9px 14px;text-align:center;font-weight:700;
                         color:#0A3D52;background:#E8F7FA;">{total_ratio:.1f}%</td>
            </tr>"""

        return f"""
        <div style="margin-bottom:6px;">
          <span style="font-size:13px;font-weight:700;color:#0A3D52;
                       background:#C2E8F0;padding:4px 12px;border-radius:20px;">
            {gender_label_str}
          </span>
        </div>
        <table style="width:100%;border-collapse:collapse;border-radius:10px;
                      overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);
                      margin-bottom:20px;font-size:14px;">
          <thead>
            <tr style="background:#0A3D52;color:#FFFFFF;">
              <th style="padding:10px 14px;text-align:left;font-weight:600;">Size</th>
              <th style="padding:10px 14px;text-align:center;font-weight:600;">Units</th>
              <th style="padding:10px 14px;text-align:center;font-weight:600;">Share</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>"""

    # ── Assortment tables — split by gender ───────────────────────────────────
    col1, col2 = st.columns(2)

    genders_present = final["GENDER"].unique().tolist()
    male_kws   = ["male", "men"]
    female_kws = ["female", "women"]

    male_df   = final[final["GENDER"].str.lower().isin(male_kws)]
    female_df = final[final["GENDER"].str.lower().isin(female_kws)]
    other_df  = final[~final["GENDER"].str.lower().isin(male_kws + female_kws)]

    with col1:
        st.markdown("**Recommended Assortment Plan**")
        st.caption("Sorted by highest units first · Darkest = most required")
        if not male_df.empty:
            st.markdown(
                gradient_html_table(male_df, "♂ Male"),
                unsafe_allow_html=True
            )
        if not female_df.empty:
            st.markdown(
                gradient_html_table(female_df, "♀ Female"),
                unsafe_allow_html=True
            )
        if not other_df.empty:
            st.markdown(
                gradient_html_table(other_df, "Other"),
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("**Unit Allocation by Gender × Size**")
        SIZE_COLORS_T3 = ["#0A3D52","#1A6E8A","#2E8FAD","#5BAFC7","#8ED0E0","#C2E8F0","#E8F7FA"]
        p_sizes   = [s for s in SIZE_ORDER if s in final["SIZE"].unique()]
        e_sizes   = [s for s in final["SIZE"].unique() if s not in SIZE_ORDER]
        ord_sizes = p_sizes + e_sizes
        sc_range  = SIZE_COLORS_T3[:len(ord_sizes)]

        alloc_chart = (
            alt.Chart(final)
            .mark_bar()
            .encode(
                x=alt.X("GENDER:N", title="Gender", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("ALLOCATED_UNITS:Q", title="Units", stack="zero"),
                color=alt.Color(
                    "SIZE:N",
                    sort=ord_sizes,
                    scale=alt.Scale(domain=ord_sizes, range=sc_range),
                    legend=alt.Legend(title="Size", orient="right")
                ),
                order=alt.Order("SIZE:N", sort="ascending"),
                tooltip=[
                    alt.Tooltip("GENDER:N",         title="Gender"),
                    alt.Tooltip("SIZE:N",            title="Size"),
                    alt.Tooltip("ALLOCATED_UNITS:Q", title="Units", format=","),
                ]
            )
            .properties(height=380)
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(alloc_chart, use_container_width=True)

    # ── Gap Analysis ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📦 Gap Analysis — Recommended vs Current Inventory")
    st.caption(
        "Positive gap = understocked (you need more). "
        "Negative gap = overstocked (you have excess). "
        "Gap % shows the variance relative to what's recommended."
    )

    with st.spinner("Fetching inventory data..."):
        inventory_df = load_inventory(selected_store, selected_category)

    if inventory_df.empty:
        st.info("No inventory data found for this store and category.")
    else:
        # Aggregate recommended at size level (across genders)
        rec_by_size = (
            final.groupby("SIZE")["ALLOCATED_UNITS"].sum().reset_index()
        )
        gap_df = (
            rec_by_size
            .merge(inventory_df[["SIZE", "STOCK_QTY"]], on="SIZE", how="left")
            .fillna({"STOCK_QTY": 0})
        )
        gap_df["STOCK_QTY"] = gap_df["STOCK_QTY"].astype(int)
        gap_df["GAP"]       = gap_df["ALLOCATED_UNITS"] - gap_df["STOCK_QTY"]
        gap_df["GAP_PCT"]   = (
            (gap_df["GAP"] / gap_df["ALLOCATED_UNITS"].replace(0, 1)) * 100
        ).round(1)

        def gap_status(x):
            if -5 <= x <= 5:  return "🟢 Well Stocked"
            elif x > 0:       return "🔴 Understocked"
            else:             return "🟡 Overstocked"

        def gap_pct_badge(pct):
            if pct > 5:
                return f'<span style="background:#fee2e2;color:#991b1b;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600;">+{pct:.1f}% short</span>'
            elif pct < -5:
                return f'<span style="background:#fef9c3;color:#78350f;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600;">{pct:.1f}% excess</span>'
            else:
                return f'<span style="background:#d1fae5;color:#065f46;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600;">~{pct:.1f}% on target</span>'

        gap_df["Status"]  = gap_df["GAP"].apply(gap_status)
        gap_df["Gap %"]   = gap_df["GAP_PCT"].apply(gap_pct_badge)

        # ── Gap HTML table with totals row ────────────────────────────────────
        gap_rows_html = ""
        for _, row in gap_df.iterrows():
            gap_color = "#fee2e2" if row["GAP"] > 5 else ("#fef9c3" if row["GAP"] < -5 else "#f0fdf4")
            gap_text  = "#991b1b" if row["GAP"] > 5 else ("#78350f" if row["GAP"] < -5 else "#065f46")
            gap_sign  = f"+{int(row['GAP'])}" if row["GAP"] > 0 else str(int(row["GAP"]))
            gap_rows_html += f"""
            <tr style="border-bottom:1px solid #e2e8f0;">
              <td style="padding:9px 14px;font-weight:600;color:#1a1a2e;">{row['SIZE']}</td>
              <td style="padding:9px 14px;text-align:center;color:#1a1a2e;">{int(row['ALLOCATED_UNITS']):,}</td>
              <td style="padding:9px 14px;text-align:center;color:#1a1a2e;">{int(row['STOCK_QTY']):,}</td>
              <td style="padding:9px 14px;text-align:center;background:{gap_color};
                         color:{gap_text};font-weight:700;">{gap_sign}</td>
              <td style="padding:9px 14px;text-align:center;">{row['Gap %']}</td>
              <td style="padding:9px 14px;text-align:center;">{row['Status']}</td>
            </tr>"""

        # Totals row
        tot_rec   = int(gap_df["ALLOCATED_UNITS"].sum())
        tot_stock = int(gap_df["STOCK_QTY"].sum())
        tot_gap   = tot_rec - tot_stock
        tot_sign  = f"+{tot_gap}" if tot_gap > 0 else str(tot_gap)
        tot_pct   = round((tot_gap / tot_rec * 100) if tot_rec else 0, 1)
        tot_pct_badge = gap_pct_badge(tot_pct)
        gap_rows_html += f"""
            <tr style="border-top:2px solid #0A3D52;background:#E8F7FA;">
              <td style="padding:10px 14px;font-weight:700;color:#0A3D52;">TOTAL</td>
              <td style="padding:10px 14px;text-align:center;font-weight:700;color:#0A3D52;">{tot_rec:,}</td>
              <td style="padding:10px 14px;text-align:center;font-weight:700;color:#0A3D52;">{tot_stock:,}</td>
              <td style="padding:10px 14px;text-align:center;font-weight:700;color:#0A3D52;">{tot_sign}</td>
              <td style="padding:10px 14px;text-align:center;">{tot_pct_badge}</td>
              <td style="padding:10px 14px;text-align:center;">—</td>
            </tr>"""

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;border-radius:10px;
                      overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);
                      font-size:14px;margin-bottom:16px;">
          <thead>
            <tr style="background:#0A3D52;color:#FFFFFF;">
              <th style="padding:10px 14px;text-align:left;">Size</th>
              <th style="padding:10px 14px;text-align:center;">Recommended</th>
              <th style="padding:10px 14px;text-align:center;">Current Stock</th>
              <th style="padding:10px 14px;text-align:center;">Gap (units)</th>
              <th style="padding:10px 14px;text-align:center;">Gap %</th>
              <th style="padding:10px 14px;text-align:center;">Status</th>
            </tr>
          </thead>
          <tbody>{gap_rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        # Summary KPIs
        n_under = (gap_df["GAP"] > 5).sum()
        n_over  = (gap_df["GAP"] < -5).sum()
        n_ok    = len(gap_df) - n_under - n_over
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("🔴 Understocked Sizes",  n_under)
        g2.metric("🟡 Overstocked Sizes",   n_over)
        g3.metric("🟢 Well Stocked Sizes",  n_ok)
        g4.metric("📦 Total Units Needed",
                  f"+{tot_gap:,}" if tot_gap > 0 else f"{tot_gap:,}",
                  delta="units to reorder" if tot_gap > 0 else "units excess",
                  delta_color="inverse" if tot_gap < 0 else "normal")

    # ── Persist for Tab 4 ─────────────────────────────────────────────────────
    st.session_state["assortment_df"] = final
    st.session_state["top_sizes"]     = (
        final.nlargest(3, "ALLOCATED_UNITS")["SIZE"].tolist()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🤖  AI Advisor":
    st.markdown(
        '<div class="section-title">🤖 Cortex AI Buying Advisor</div>',
        unsafe_allow_html=True
    )
    st.caption("Powered by **Snowflake Cortex** · `snowflake-arctic` · COMPLETE() + SENTIMENT()")
    st.markdown("---")

    # ── Store context card ────────────────────────────────────────────────────
    store_row_t4 = stores_df[stores_df["STORE_ID"] == selected_store]
    city_t4      = store_row_t4["CITY"].values[0]      if not store_row_t4.empty else "—"
    stype_t4     = store_row_t4["STORE_TYPE"].values[0] if not store_row_t4.empty else "—"

    st.markdown(f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:18px;">
      <div style="background:#f0f4ff;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;">
        <div style="font-size:11px;color:#5a6a85;font-weight:600;text-transform:uppercase;
                    letter-spacing:.5px;">Store</div>
        <div style="font-size:16px;font-weight:700;color:#1a1a2e;margin-top:4px;">
            {selected_store_name}</div>
        <div style="font-size:12px;color:#5a6a85;">📌 {city_t4} · {stype_t4}</div>
      </div>
      <div style="background:#f0f4ff;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;">
        <div style="font-size:11px;color:#5a6a85;font-weight:600;text-transform:uppercase;
                    letter-spacing:.5px;">Region</div>
        <div style="font-size:16px;font-weight:700;color:#1a1a2e;margin-top:4px;">
            {selected_region_name}</div>
      </div>
      <div style="background:#f0f4ff;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;">
        <div style="font-size:11px;color:#5a6a85;font-weight:600;text-transform:uppercase;
                    letter-spacing:.5px;">Category</div>
        <div style="font-size:16px;font-weight:700;color:#1a1a2e;margin-top:4px;">
            {selected_category}</div>
      </div>
      <div style="background:#e8f0fe;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;">
        <div style="font-size:11px;color:#1a56db;font-weight:600;text-transform:uppercase;
                    letter-spacing:.5px;">Open-to-Buy</div>
        <div style="font-size:16px;font-weight:700;color:#1a1a2e;margin-top:4px;">
            {clothes_space:,} units</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Generate button + what Cortex analyses ────────────────────────────────
    col_btn, col_info = st.columns([1, 2])

    with col_btn:
        run_ai = st.button(
            "✨ Generate AI Buying Brief",
            type="primary",
            use_container_width=True,
            help="Calls Snowflake Cortex COMPLETE() with your store context"
        )
        st.markdown("""
        <div style="background:#f7f8fc;border-radius:10px;padding:14px 16px;
                    margin-top:12px;font-size:13px;color:#334155;">
            <div style="font-weight:700;margin-bottom:8px;color:#1a1a2e;">
                What Cortex analyses:</div>
            🏙️ Regional body profile distribution<br>
            📦 Recommended assortment allocation<br>
            📈 Top selling sizes (historical)<br>
            💬 Customer return reasons & sentiment<br>
            🗺️ Store location & type context
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.info(
            "💡 **Tip:** Visit **Assortment Insights** first — the AI uses the "
            "generated allocation plan as a key input for its buying brief."
        )
        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                    padding:12px 16px;font-size:13px;color:#166534;margin-top:8px;">
            <strong>How it works:</strong> Cortex reads your store's demographic profile,
            the optimised size allocation, and fit-related return reasons — then writes a
            personalised buying brief a store director can act on immediately.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if run_ai:
        with st.spinner("🧠 Cortex AI is analysing regional profiles and return sentiment..."):

            assortment_df = st.session_state.get("assortment_df", pd.DataFrame())
            top_sizes     = st.session_state.get("top_sizes", [])

            if assortment_df.empty:
                demand_df = size_demand_df[
                    (size_demand_df["REGION_ID"] == selected_region) &
                    (size_demand_df["CATEGORY"]  == selected_category)
                ]
                recomp = (
                    demand_df
                    .merge(
                        region_body_dist_df[region_body_dist_df["REGION_ID"] == selected_region][
                            ["BODY_PROFILE_ID","POPULATION_PCT"]
                        ], on="BODY_PROFILE_ID"
                    )
                    .merge(body_profiles_df[["BODY_PROFILE_ID","GENDER"]], on="BODY_PROFILE_ID")
                    .merge(size_map_df, on="SIZE_ID")
                )
                recomp["WEIGHTED"] = recomp["DEMAND_WEIGHT"] * recomp["POPULATION_PCT"]
                recomp_agg         = (
                    recomp.groupby(["GENDER","SIZE","SIZE_ID"])["WEIGHTED"]
                    .sum().reset_index()
                )
                recomp_agg["RATIO"]     = recomp_agg["WEIGHTED"] / recomp_agg["WEIGHTED"].sum()
                recomp_agg              = allocate_units(recomp_agg, "RATIO", clothes_space)
                recomp_agg              = recomp_agg.rename(columns={"ALLOCATED":"ALLOCATED_UNITS"})
                assortment_df           = recomp_agg
                top_sizes               = recomp_agg.nlargest(3,"ALLOCATED_UNITS")["SIZE"].tolist()

            try:
                sentiment_df = load_returns_sentiment(selected_store, selected_category)
            except Exception:
                sentiment_df = pd.DataFrame()

            narrative = get_ai_recommendation(
                selected_store_name, selected_region_name, selected_category,
                assortment_df, sentiment_df, top_sizes, clothes_space
            )

        # ── AI Narrative ──────────────────────────────────────────────────────
        st.markdown("#### 📝 AI Buying Brief")
        st.markdown(f"""
        <div class="ai-narrative-box">
            {narrative}
            <div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(124,58,237,0.2);
                        font-size:11px;color:#7c3aed;opacity:0.7;">
                ⚡ Generated by Snowflake Cortex · snowflake-arctic · COMPLETE()
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Sentiment section ──────────────────────────────────────────────────
        if not sentiment_df.empty:
            st.markdown("---")
            st.markdown("#### 💬 Return Sentiment Analysis")
            st.caption(
                "Cortex SENTIMENT() scored each return reason. "
                "Score: −1.0 = very negative · 0 = neutral · +1.0 = very positive."
            )

            # Sentiment overview bar
            neg = int((sentiment_df["SENTIMENT_SCORE"] < -0.2).sum())
            neu = int(((sentiment_df["SENTIMENT_SCORE"] >= -0.2) & (sentiment_df["SENTIMENT_SCORE"] < 0.2)).sum())
            pos = int((sentiment_df["SENTIMENT_SCORE"] >= 0.2).sum())
            total_s = neg + neu + pos
            if total_s > 0:
                neg_pct = round(neg / total_s * 100)
                neu_pct = round(neu / total_s * 100)
                pos_pct = 100 - neg_pct - neu_pct
                st.markdown(f"""
                <div style="margin:10px 0 18px;">
                  <div style="font-size:12px;color:#5a6a85;margin-bottom:6px;font-weight:600;">
                      Overall Sentiment Distribution</div>
                  <div style="display:flex;height:28px;border-radius:14px;overflow:hidden;
                              box-shadow:0 1px 4px rgba(0,0,0,0.1);">
                    <div style="width:{neg_pct}%;background:#fca5a5;display:flex;align-items:center;
                                justify-content:center;font-size:11px;font-weight:700;color:#991b1b;">
                        {neg_pct}%</div>
                    <div style="width:{neu_pct}%;background:#fde68a;display:flex;align-items:center;
                                justify-content:center;font-size:11px;font-weight:700;color:#78350f;">
                        {neu_pct}%</div>
                    <div style="width:{pos_pct}%;background:#6ee7b7;display:flex;align-items:center;
                                justify-content:center;font-size:11px;font-weight:700;color:#065f46;">
                        {pos_pct}%</div>
                  </div>
                  <div style="display:flex;gap:16px;margin-top:6px;font-size:11px;color:#5a6a85;">
                    <span>😡 Negative ({neg})</span>
                    <span>😐 Neutral ({neu})</span>
                    <span>😊 Positive ({pos})</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            s1, s2, s3 = st.columns(3)
            s1.metric("😡 Negative Returns", neg)
            s2.metric("😐 Neutral Returns",  neu)
            s3.metric("😊 Positive Returns", pos)

            sentiment_display = sentiment_df.copy()
            sentiment_display["SENTIMENT_SCORE"] = sentiment_display["SENTIMENT_SCORE"].round(3)
            sentiment_display["Sentiment Label"] = sentiment_display["SENTIMENT_SCORE"].apply(
                lambda x: "😡 Negative" if x < -0.2
                else ("😐 Neutral" if x < 0.2 else "😊 Positive")
            )
            st.dataframe(
                sentiment_display[[
                    "RETURN_REASON","SIZE","RETURN_COUNT",
                    "SENTIMENT_SCORE","Sentiment Label"
                ]].rename(columns={
                    "RETURN_REASON":  "Return Reason",
                    "SIZE":           "Size",
                    "RETURN_COUNT":   "No. of Returns",
                    "SENTIMENT_SCORE":"Sentiment Score"
                }),
                use_container_width=True
            )
        else:
            st.info("No return data found for this store and category — sentiment analysis skipped.")

    else:
        st.markdown("""
        <div style="background:#f7f8fc;border-radius:12px;padding:24px 28px;
                    text-align:center;color:#5a6a85;margin-top:12px;">
            <div style="font-size:32px;margin-bottom:10px;">🤖</div>
            <div style="font-size:16px;font-weight:600;color:#1a1a2e;margin-bottom:6px;">
                Ready to generate your AI buying brief</div>
            <div style="font-size:13px;">
                Set your filters and click
                <strong>✨ Generate AI Buying Brief</strong> above.
            </div>
        </div>
        """, unsafe_allow_html=True)
