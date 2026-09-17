"""
S&P 500 Stock Clustering Explorer
----------------------------------
Built on top of the Imperial DSA-with-Python Homework 3 dataset:
  - SP_500_firms.csv    : 504 S&P 500 tickers, names, GICS sectors
  - SP_500_close_2015.csv : daily close prices for 496 of those tickers, all 252
    trading days of 2015

The homework asked you to implement a greedy single-linkage clustering
algorithm (union-find over a correlation edge list, sorted by strength) to
group stocks by how similarly their prices moved that year. This app turns
that notebook exercise into an interactive tool: pick a sector, inspect
returns/volatility, explore the correlation structure, and run the
clustering algorithm live with an adjustable "how many edges to merge" (k)
slider so you can see clusters form from nothing to over-merged.

Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

st.set_page_config(page_title="S&P 500 Clustering Explorer", layout="wide")

DATA_DIR = "data"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@st.cache_data(show_spinner="Fetching current S&P 500 constituents from Wikipedia...")
def scrape_current_constituents():
    """Live scrape of today's S&P 500 membership: requests to fetch the page,
    BeautifulSoup to parse the constituents table, find_all to pull rows out
    of it -- the same pattern as the DSA-with-Python scraping notebook, on a
    real page instead of a supplied one. Falls back to a bundled snapshot
    (fetched at build time) if Wikipedia can't be reached."""
    try:
        r = requests.get(WIKI_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        rows = table.find_all("tr")
        records = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            records.append({
                "Symbol": cells[0].get_text(strip=True),
                "Security": cells[1].get_text(strip=True),
                "Sector": cells[2].get_text(strip=True),
                "Sub-Industry": cells[3].get_text(strip=True),
                "Date added": cells[5].get_text(strip=True) if len(cells) > 5 else "",
            })
        df = pd.DataFrame(records)
        if len(df) < 400:
            raise ValueError("Scrape returned an implausibly small table, falling back")
        return df, "live scrape (en.wikipedia.org, requests + BeautifulSoup)"
    except Exception:
        df = pd.read_csv(f"{DATA_DIR}/sp500_current_snapshot.csv")
        return df, "bundled snapshot (scraped at build time, offline fallback)"


@st.cache_data
def load_data():
    firms = pd.read_csv(f"{DATA_DIR}/SP_500_firms.csv")
    prices = pd.read_csv(f"{DATA_DIR}/SP_500_close_2015.csv", index_col=0, parse_dates=True)
    # Keep only tickers we have both a name/sector AND a price series for
    common = [t for t in prices.columns if t in set(firms["Symbol"])]
    prices = prices[common]
    firms = firms[firms["Symbol"].isin(common)].set_index("Symbol")
    return firms, prices


@st.cache_data
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


@st.cache_data
def compute_correlations(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


@st.cache_data
def build_edge_list(correl: pd.DataFrame):
    """Upper triangle of the correlation matrix as (weight, ticker_a, ticker_b) tuples,
    sorted by strongest correlation first — same structure the homework notebook builds
    before running the clustering algorithm."""
    tickers = list(correl.columns)
    mat = correl.to_numpy()
    edges = []
    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((mat[i, j], tickers[i], tickers[j]))
    edges.sort(key=lambda e: -e[0])
    return edges


# --------------------------------------------------------------------------
# Clustering algorithm (the "mystery algorithm" from the HW: a Kruskal-style
# single-linkage clustering via union-find with path-to-bottom pointers)
# --------------------------------------------------------------------------

def find_bottom(node, next_nodes):
    while next_nodes[node] != node:
        node = next_nodes[node]
    return node


def merge_sets(node1, node2, next_nodes, set_starters):
    b1 = find_bottom(node1, next_nodes)
    b2 = find_bottom(node2, next_nodes)
    if b1 != b2:
        next_nodes[b1] = b2
        set_starters.discard(b1)


def cluster_correlations(edge_list, firms, k):
    """Merge the k strongest-correlation edges via union-find.
    Larger k -> more merges -> fewer, bigger clusters."""
    next_nodes = {f: f for f in firms}
    set_starters = set(firms)
    for i in range(min(k, len(edge_list))):
        _, a, b = edge_list[i]
        merge_sets(a, b, next_nodes, set_starters)
    return next_nodes, set_starters


def construct_sets(set_starters, next_nodes):
    all_sets = {}
    for s in set_starters:
        cur = {s}
        p = s
        while next_nodes[p] != p:
            p = next_nodes[p]
            cur.add(p)
        if p not in all_sets:
            all_sets[p] = cur
        else:
            all_sets[p] |= cur
    return all_sets


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

firms, prices = load_data()
returns = compute_returns(prices)

st.title("S&P 500 Stock Clustering Explorer — 2015")
st.caption(
    "Built from Imperial DSA-with-Python HW3: 496 S&P 500 tickers, "
    "252 trading days of 2015 close prices, 10 GICS sectors."
)

tab_overview, tab_movers, tab_corr, tab_cluster, tab_turnover = st.tabs(
    ["Overview", "Returns & Volatility", "Correlations", "Clustering", "2015 vs Today"]
)

# ---- Overview -------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Universe")
        st.metric("Tickers", len(firms))
        st.metric("Trading days", len(prices))
        sector_counts = firms["Sector"].value_counts()
        fig = px.bar(
            sector_counts, orientation="h",
            labels={"value": "# companies", "index": "Sector"},
            title="Companies per GICS sector",
        )
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Normalised price paths")
        default_sector = "Information Technology"
        sector_pick = st.selectbox(
            "Sector", sorted(firms["Sector"].unique()),
            index=sorted(firms["Sector"].unique()).index(default_sector)
            if default_sector in firms["Sector"].unique() else 0,
        )
        tickers_in_sector = firms[firms["Sector"] == sector_pick].index.tolist()
        chosen = st.multiselect(
            "Tickers (normalised to 1.0 at start of year)",
            tickers_in_sector,
            default=tickers_in_sector[: min(6, len(tickers_in_sector))],
        )
        if chosen:
            norm = prices[chosen].divide(prices[chosen].iloc[0])
            fig2 = px.line(norm, labels={"value": "Price / start-of-year price", "Date": ""})
            fig2.update_layout(height=420, legend_title_text="Ticker")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pick at least one ticker.")

# ---- Returns & Volatility --------------------------------------------------
with tab_movers:
    st.subheader("Best and worst movers of 2015")
    yearly_return = (prices.iloc[-1] / prices.iloc[0] - 1).sort_values(ascending=False)
    volatility = (returns.std() * np.sqrt(252)).sort_values(ascending=False)

    pct_col = st.column_config.NumberColumn(format="%.1f%%")

    n = st.slider("How many to show", 5, 25, 10)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Highest yearly return**")
        top = yearly_return.head(n).to_frame("Yearly return")
        top["Yearly return"] = top["Yearly return"] * 100
        top["Name"] = firms.loc[top.index, "Name"]
        top["Sector"] = firms.loc[top.index, "Sector"]
        st.dataframe(top, use_container_width=True, column_config={"Yearly return": pct_col})
    with c2:
        st.markdown("**Lowest yearly return**")
        bottom = yearly_return.tail(n).sort_values().to_frame("Yearly return")
        bottom["Yearly return"] = bottom["Yearly return"] * 100
        bottom["Name"] = firms.loc[bottom.index, "Name"]
        bottom["Sector"] = firms.loc[bottom.index, "Sector"]
        st.dataframe(bottom, use_container_width=True, column_config={"Yearly return": pct_col})

    st.markdown("**Highest annualised volatility**")
    vol_top = volatility.head(n).to_frame("Annualised volatility")
    vol_top["Annualised volatility"] = vol_top["Annualised volatility"] * 100
    vol_top["Name"] = firms.loc[vol_top.index, "Name"]
    vol_top["Sector"] = firms.loc[vol_top.index, "Sector"]
    st.dataframe(vol_top, use_container_width=True, column_config={"Annualised volatility": pct_col})

# ---- Correlations -----------------------------------------------------------
with tab_corr:
    st.subheader("Correlation structure")
    all_tickers = list(prices.columns)
    default_set = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]
    default_set = [t for t in default_set if t in all_tickers] or all_tickers[:5]
    picked = st.multiselect("Tickers to compare", all_tickers, default=default_set)
    if len(picked) >= 2:
        sub_corr = compute_correlations(returns)[picked].loc[picked]
        fig3 = px.imshow(
            sub_corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
            title="Return correlation matrix",
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)

        focus = st.selectbox("Show top/bottom correlated companies for:", picked)
        full_corr = compute_correlations(returns)[focus].drop(focus).sort_values(ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Most correlated with {focus}**")
            st.dataframe(full_corr.head(10).to_frame("Correlation"), use_container_width=True)
        with c2:
            st.markdown(f"**Least correlated with {focus}**")
            st.dataframe(full_corr.tail(10).to_frame("Correlation"), use_container_width=True)
    else:
        st.info("Pick at least two tickers.")

# ---- Clustering ---------------------------------------------------------------
with tab_cluster:
    st.subheader("Run the clustering algorithm")
    st.caption(
        "Single-linkage clustering via union-find: sort every pairwise correlation, "
        "then merge the k strongest edges. Small k leaves most stocks as singletons; "
        "large k merges everything into one giant cluster."
    )
    k = st.slider("k — number of strongest edges to merge", 10, 3000, 400, step=10)

    correl = compute_correlations(returns)
    edges = build_edge_list(correl)
    next_nodes, set_starters = cluster_correlations(edges, list(correl.columns), k)
    clusters = construct_sets(set_starters, next_nodes)
    sizes = sorted(((len(v), root) for root, v in clusters.items()), reverse=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Number of clusters", len(clusters))
        st.metric("Largest cluster size", sizes[0][0] if sizes else 0)
        st.markdown("**Cluster sizes**")
        size_df = pd.DataFrame(sizes, columns=["size", "root"]).drop(columns="root")
        st.bar_chart(size_df.head(20).reset_index(drop=True))

    with c2:
        st.markdown("**Inspect a cluster**")
        big_clusters = [root for size, root in sizes if size >= 3][:15]
        if big_clusters:
            pick_root = st.selectbox(
                "Cluster (labelled by its largest member)", big_clusters,
                format_func=lambda r: f"{r} — {len(clusters[r])} stocks",
            )
            members = sorted(clusters[pick_root])
            member_df = firms.loc[[m for m in members if m in firms.index]]
            st.dataframe(member_df, use_container_width=True)

            norm = prices[members].divide(prices[members].iloc[0])
            fig4 = px.line(norm, labels={"value": "Normalised price", "Date": ""})
            fig4.update_layout(height=400, showlegend=len(members) <= 15)
            st.plotly_chart(fig4, use_container_width=True)

            sector_mix = member_df["Sector"].value_counts()
            st.caption("Sector mix of this cluster: " + ", ".join(f"{s} ({c})" for s, c in sector_mix.items()))
        else:
            st.info("Raise k to form clusters of 3+ stocks.")

# ---- 2015 vs Today (live scrape) --------------------------------------------
with tab_turnover:
    st.subheader("How much has the index turned over since 2015?")
    st.caption(
        "The HW3 dataset is a snapshot of the S&P 500 as it stood in 2015. This tab scrapes "
        "today's actual constituent list live from Wikipedia (requests + BeautifulSoup, the "
        "same pattern as the DSA-with-Python scraping notebook) and compares it against 2015, "
        "rather than just describing the 2015 data on its own."
    )

    current, source = scrape_current_constituents()
    st.caption(f"Source: {source}. {len(current)} current constituents scraped.")

    # Normalise ticker punctuation (Wikipedia uses BRK.B, the 2015 dataset uses BRK-B)
    current["Symbol_norm"] = current["Symbol"].str.replace(".", "-", regex=False)
    today_syms = set(current["Symbol_norm"])
    old_syms = set(firms.index)

    left_index = sorted(old_syms - today_syms)
    added_since = sorted(today_syms - old_syms)
    still_in = sorted(old_syms & today_syms)

    m1, m2, m3 = st.columns(3)
    m1.metric("Still in the index", len(still_in))
    m2.metric("Left since 2015", len(left_index))
    m3.metric("Added since 2015", len(added_since))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sector mix, 2015 (HW3 dataset)**")
        fig5 = px.pie(firms["Sector"].value_counts().reset_index(), names="Sector", values="count", hole=0.4)
        fig5.update_layout(height=380, showlegend=False)
        fig5.update_traces(textinfo="label+percent")
        st.plotly_chart(fig5, use_container_width=True)
    with c2:
        st.markdown("**Sector mix, today (live scrape)**")
        fig6 = px.pie(current["Sector"].value_counts().reset_index(), names="Sector", values="count", hole=0.4)
        fig6.update_layout(height=380, showlegend=False)
        fig6.update_traces(textinfo="label+percent")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**Companies that have left the index since 2015**")
    left_df = firms.loc[[t for t in left_index if t in firms.index]]
    st.dataframe(left_df, use_container_width=True)

    st.markdown("**Companies added to the index since 2015**")
    added_df = current[current["Symbol_norm"].isin(added_since)][["Symbol", "Security", "Sector", "Date added"]]
    added_df = added_df.sort_values("Date added", ascending=False)
    st.dataframe(added_df, use_container_width=True, hide_index=True)
