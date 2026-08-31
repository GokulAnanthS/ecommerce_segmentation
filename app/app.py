import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

SEGMENT_COLORS = {
    "VIP Outlier": "#c44e52",
    "Active High-Value Core": "#55a868",
    "Lapsed Low-Value": "#8c8c8c",
    "International": "#4c72b0",
}

PERSONAS = {
    "VIP Outlier": {
        "blurb": "DBSCAN-flagged density outliers — too large or unusual to belong to any "
                 "\"normal\" cluster. Only 3.2% of customers but ~30% of revenue.",
        "action": "Dedicated account management / personal outreach, not a generic campaign.",
    },
    "Active High-Value Core": {
        "blurb": "Recent, frequent, reliable — the main revenue engine.",
        "action": "Loyalty/rewards program to retain and nudge toward VIP status.",
    },
    "Lapsed Low-Value": {
        "blurb": "Largest group by count, smallest by revenue. Long recency, few orders.",
        "action": "Low-cost automated win-back email with a discount; don't over-invest.",
    },
    "International": {
        "blurb": "Ships outside the UK. Lower purchase frequency than the domestic core "
                 "despite similar recency.",
        "action": "Investigate shipping/currency friction; consider localized offers.",
    },
}


@st.cache_data
def load_data():
    raw = pd.read_parquet(DATA_DIR / "rfm_features.parquet")
    scaled = pd.read_parquet(DATA_DIR / "rfm_features_scaled.parquet")
    segments = pd.read_parquet(DATA_DIR / "final_segments.parquet")
    df = raw.join(segments)
    return df, scaled


@st.cache_data
def compute_pca(scaled_df: pd.DataFrame):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_df)
    coords_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=scaled_df.index)
    return coords_df, pca.explained_variance_ratio_


st.set_page_config(page_title="Customer Segmentation Explorer", layout="wide")
st.title("E-commerce Customer Segmentation Explorer")
st.caption(
    "Online Retail II — segments from a K-Means base (3 clusters) with DBSCAN's "
    "outlier flag layered on top as a 4th, higher-priority segment. See the README "
    "for the full algorithm comparison."
)

df, scaled = load_data()
pca_coords, variance_ratio = compute_pca(scaled)

# ---------------- Sidebar ----------------
st.sidebar.header("Filter")
all_segments = sorted(df["segment"].unique())
selected_segments = st.sidebar.multiselect("Segments", all_segments, default=all_segments)

st.sidebar.header("Look up a customer")
customer_ids = sorted(df.index[df["segment"].isin(selected_segments)])
selected_customer = st.sidebar.selectbox(
    "Customer ID", options=[None] + customer_ids, format_func=lambda x: "—" if x is None else str(x)
)

filtered = df[df["segment"].isin(selected_segments)]

# ---------------- KPI row ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers (filtered)", f"{len(filtered):,}")
col2.metric("Total revenue (filtered)", f"£{filtered['monetary'].sum():,.0f}")
col3.metric("Avg spend / customer", f"£{filtered['monetary'].mean():,.0f}")
col4.metric(
    "Selected customer segment",
    df.loc[selected_customer, "segment"] if selected_customer else "—",
)

# ---------------- Segment overview ----------------
st.subheader("Segment overview")
overview_col1, overview_col2 = st.columns(2)

counts = df["segment"].value_counts().rename_axis("segment").reset_index(name="n_customers")
chart_counts = (
    alt.Chart(counts)
    .mark_bar()
    .encode(
        x=alt.X("n_customers:Q", title="Customers"),
        y=alt.Y("segment:N", sort="-x", title=None),
        color=alt.Color("segment:N", scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                       range=list(SEGMENT_COLORS.values())), legend=None),
        tooltip=["segment", "n_customers"],
    )
    .properties(height=200)
)
overview_col1.altair_chart(chart_counts, use_container_width=True)

revenue = df.groupby("segment")["monetary"].sum().rename("revenue").reset_index()
revenue["pct_of_total"] = revenue["revenue"] / revenue["revenue"].sum() * 100
chart_revenue = (
    alt.Chart(revenue)
    .mark_bar()
    .encode(
        x=alt.X("pct_of_total:Q", title="% of total revenue"),
        y=alt.Y("segment:N", sort="-x", title=None),
        color=alt.Color("segment:N", scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                       range=list(SEGMENT_COLORS.values())), legend=None),
        tooltip=["segment", alt.Tooltip("pct_of_total:Q", format=".1f")],
    )
    .properties(height=200)
)
overview_col2.altair_chart(chart_revenue, use_container_width=True)

# ---------------- PCA scatter ----------------
st.subheader("PCA scatter (all customers, colored by segment)")
st.caption(f"2 components explain {variance_ratio.sum()*100:.1f}% of variance — a rough sketch, not the full picture.")

plot_df = pca_coords.join(df[["segment", "monetary"]])
plot_df["is_selected"] = plot_df.index == selected_customer if selected_customer else False
plot_df = plot_df[plot_df["segment"].isin(selected_segments)]

base_scatter = (
    alt.Chart(plot_df.reset_index())
    .mark_circle(size=40, opacity=0.6)
    .encode(
        x="PC1:Q",
        y="PC2:Q",
        color=alt.Color("segment:N", scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                       range=list(SEGMENT_COLORS.values()))),
        tooltip=["Customer_ID", "segment", alt.Tooltip("monetary:Q", format=",.0f")],
    )
    .properties(height=420)
)

if selected_customer is not None:
    highlight = (
        alt.Chart(plot_df.reset_index()[plot_df.reset_index()["Customer_ID"] == selected_customer])
        .mark_point(size=300, shape="cross", color="black", strokeWidth=3)
        .encode(x="PC1:Q", y="PC2:Q")
    )
    st.altair_chart(base_scatter + highlight, use_container_width=True)
else:
    st.altair_chart(base_scatter, use_container_width=True)

# ---------------- Customer detail ----------------
if selected_customer is not None:
    st.subheader(f"Customer {selected_customer}")
    row = df.loc[selected_customer]
    segment_avg = df[df["segment"] == row["segment"]].mean(numeric_only=True)

    detail = pd.DataFrame({
        "This customer": row[["recency", "frequency", "monetary", "avg_basket_size", "category_diversity"]],
        f"{row['segment']} average": segment_avg[["recency", "frequency", "monetary", "avg_basket_size", "category_diversity"]],
    }).round(1)
    st.dataframe(detail, use_container_width=True)

    persona = PERSONAS[row["segment"]]
    st.info(f"**{row['segment']}** — {persona['blurb']}\n\n**Suggested action:** {persona['action']}")

# ---------------- Segment table ----------------
st.subheader("Customers in filtered segments")
display_cols = ["segment", "recency", "frequency", "monetary", "avg_basket_size", "category_diversity", "is_international"]
st.dataframe(
    filtered[display_cols].sort_values("monetary", ascending=False),
    use_container_width=True,
)
