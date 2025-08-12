# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("your_data.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

cleaned_PBMZI = load_data()

# --- Page Navigation ---
page = st.sidebar.selectbox("Select Page", ["PBMZI (2018–2023)", "MST Overview"])

# ----------------------------------------------------------------------
# PAGE 1: PBMZI (2018–2023)
# ----------------------------------------------------------------------
if page == "PBMZI (2018–2023)":
    st.title("PBMZI (2018–2023)")

    # First Row Layout
    col1, col2, col3, col4 = st.columns([1, 2, 4, 4])

    with col1:
        st.markdown("### Year")
        years = list(range(2018, 2024))
        selected_years = [year for year in years if st.checkbox(str(year), True)]
    with col2:
        st.markdown("### Companies")
        st.write(list(cleaned_PBMZI.columns[1:]))

    # Filtered data
    filtered_df = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]

    with col3:
        st.subheader("Price Movement")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        for company in filtered_df.columns[1:]:
            ax1.plot(filtered_df['Date'], filtered_df[company], label=company)
        ax1.legend(fontsize=6)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (~M$)')
        st.pyplot(fig1)

    with col4:
        st.subheader("Return Movement")
        log_return = filtered_df[filtered_df.columns[1:]].apply(lambda col: np.log(col / col.shift(1)))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for company in log_return.columns:
            ax2.plot(filtered_df['Date'], log_return[company], label=company)
        ax2.legend(fontsize=6)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Log Return')
        st.pyplot(fig2)

    # Second Row Layout
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Volatility (60-Day Rolling)")
        volatility = filtered_df[filtered_df.columns[1:]].rolling(window=60).std()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        for company in volatility.columns:
            ax3.plot(filtered_df['Date'], volatility[company], label=company)
        ax3.legend(fontsize=6)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Rolling Std')
        st.pyplot(fig3)

    with col6:
        st.subheader("Correlation Matrix (Log Return)")
        log_return_filtered = filtered_df[filtered_df.columns[1:]].apply(lambda col: np.log(col / col.shift(1)))
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(log_return_filtered.corr(), cmap='coolwarm_r', square=True,
                    xticklabels=log_return_filtered.columns,
                    yticklabels=log_return_filtered.columns, vmin=-1, vmax=1, ax=ax4)
        ax4.set_title("Correlation Matrix of PBMZI Companies' Log Return", fontsize=10)
        st.pyplot(fig4)

# ----------------------------------------------------------------------
# PAGE 2: MST Overview
# ----------------------------------------------------------------------
if page == "MST Overview":
    st.title("MST Overview")

    # Function to plot MST for a given dataframe and title
    def plot_mst(df, title):
        row1, col1 = df.shape
        returns = np.log(df.iloc[1:row1, 1:col1].values / df.iloc[0:row1-1, 1:col1].values)
        returns_df = pd.DataFrame(returns, columns=df.columns[1:])
        corremat = np.corrcoef(returns_df.T)
        distmat = np.round(np.sqrt(2 * (1 - corremat)), 2)

        companies = returns_df.columns.tolist()
        G = nx.Graph()
        G.add_nodes_from(companies)
        for i in range(len(companies)):
            for j in range(i+1, len(companies)):
                G.add_edge(companies[i], companies[j], weight=distmat[i, j])

        MST = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
        node_values = dict(MST.degree())
        values = np.array(list(node_values.values()))
        norm_nodes = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        cmap = cm.get_cmap("summer_r")
        node_colors = [cmap(norm_nodes(val)) for val in values]
        node_sizes = [v * 300 for v in node_values.values()]
        weights = nx.get_edge_attributes(MST, "weight").values()
        norm = mcolors.Normalize(vmin=min(weights)-0.1, vmax=max(weights)+0.1)
        edge_colors = [(0, 0, 1, norm(w)) for w in weights]
        sm_edges = cm.ScalarMappable(cmap=cm.Blues, norm=norm)
        sm_edges.set_array([])

        fig, ax = plt.subplots(figsize=(4, 4))
        pos = nx.kamada_kawai_layout(MST)
        nx.draw(MST, pos, with_labels=True, node_color=node_colors,
                node_size=node_sizes, font_size=6, font_color="black",
                edge_color=edge_colors, ax=ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm_nodes)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)
        ax.set_title(title, fontsize=8)
        return fig

    # First row: MST 2018-2023
    cols = st.columns(6)
    for idx, year in enumerate(range(2018, 2024)):
        df_year = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year == year].reset_index(drop=True)
        fig = plot_mst(df_year, f"MST {year}")
        cols[idx].pyplot(fig)

    # Second row: MST All Years
    st.subheader("MST 2018–2023 Combined")
    fig_all = plot_mst(cleaned_PBMZI, "MST 2018–2023")
    st.pyplot(fig_all)
