import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =========================
# Load and prepare the data
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_PBMZI.xlsx")  # Change to your Excel file
    # Do your cleaning steps here...
    # For example: convert date
    df['Date'] = pd.to_datetime(df['Date'])
    return df

cleaned_PBMZI = load_data()

# Extract available years
available_years = sorted(cleaned_PBMZI['Date'].dt.year.unique())

# =========================
# Sidebar Navigation & Filters
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["PBMZI (2018-2023)", "MST Overview"])

# Common filters in sidebar
st.sidebar.subheader("Filters")

# Companies filter only for PBMZI page
all_companies = cleaned_PBMZI.columns[1:]

# Page-specific filters
if page == "PBMZI (2018-2023)":
    selected_companies = st.sidebar.multiselect(
        "Select companies:", all_companies, default=list(all_companies)
    )

    selected_years = st.sidebar.multiselect(
        "Select years:", available_years, default=available_years
    )
    if not selected_years:
        selected_years = list(range(2018, 2024))

elif page == "MST Overview":
    selected_years = st.sidebar.multiselect(
        "Select year(s):", available_years
    )
    if not selected_years:
        selected_years = list(range(2018, 2024))

# =========================
# PAGE 1 – PBMZI EDA
# =========================
if page == "PBMZI (2018-2023)":
    st.title("PBMZI (2018-2023)")

    filtered_data = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]
    # 1. Price Movement
    st.subheader("Price Movement Overview")
    fig, ax = plt.subplots(figsize=(14, 8))
    for company in selected_companies:
        ax.plot(filtered_data['Date'], filtered_data[company], label=company)
    ax.legend()
    ax.set_title("Price Movement of PBMZI ({}–{})".format(min(selected_years), max(selected_years)))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (~M$)")
    st.pyplot(fig)

    # 2. Return Movement
    st.subheader("Logarithmic Return Movement")
    log_return = filtered_data[selected_companies].apply(lambda col: np.log(col / col.shift(1)))
    fig, ax = plt.subplots(figsize=(14, 8))
    for company in selected_companies:
        ax.plot(filtered_data['Date'], log_return[company], label=company)
    ax.legend()
    ax.set_title("Logarithmic Return Movement of PLCs ({}–{})".format(min(selected_years), max(selected_years)))
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    st.pyplot(fig)

    # 3. Volatility
    st.subheader("Volatility (60-Day Rolling STD)")
    
    volatility_return=filtered_data[selected_companies].apply(lambda col: np.log(col / col.shift(1))).rolling(window=60).std()
    fig, ax = plt.subplots(figsize=(14, 8))
    for company inselected_companies:
        ax.plot(filtered_data['Date'], volatility_return[company], label=company)
    ax.legend()
    ax.set_title('60-Day Rolling Volatility Based on PBMZI Logarithmic Return')
    ax.set_xlabel('Year')
    ax.set_ylabel('60-Day Rolling Volatility')
    
    st.pyplot(fig)

    # 4. Correlation Matrix
    st.subheader("Correlation Matrix of Log Return")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(log_return.corr(method='pearson'),
                cmap='coolwarm_r',
                square=True,
                xticklabels=selected_companies,
                yticklabels=selected_companies,
                vmin=-1, vmax=1,
                ax=ax)
    ax.set_title("Correlation Matrix of PBMZI Companies' Log Return", size=15, pad=20)
    st.pyplot(fig)
    
    # 5. Correlation Distribution (PBMZI Log Returns)

    # Kira log return ikut syarikat dan tahun yang dipilih
    log_return_df = filtered_data[selected_companies].apply(lambda col: np.log(col / col.shift(1)))

    # Buang baris NaN (hasil dari shift)
    log_return_df = log_return_df.dropna()

    # Kira Pearson correlation matrix untuk syarikat terpilih
    log_corr_matrix = log_return_df[selected_companies].corr(method='pearson')

    # Ambil hanya lower triangle tanpa diagonal
    mask = np.tril(np.ones(log_corr_matrix.shape), k=-1).astype(bool)
    log_corr_values = log_corr_matrix.where(mask).stack().values

    # Kategori correlation
    bins = [-1.0, -0.7, -0.4, 0, 0.4, 0.7, 1.0]
    labels = [
        'Highly Strong Negative\n[-1.0, -0.7)',
        'Strong Negative\n[-0.7, -0.4)',
        'Weak Negative\n[-0.4, 0)',
        'Weak Positive\n[0, 0.4)',
        'Strong Positive\n[0.4, 0.7)',
        'Highly Strong Positive\n[0.7, 1.0]'
    ]
    log_corr_categories = pd.cut(log_corr_values, bins=bins, labels=labels, right=False)

    # Bilangan setiap kategori
    log_corr_distribution = log_corr_categories.value_counts().sort_index()

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=log_corr_distribution.index, y=log_corr_distribution.values, palette='coolwarm_r', ax=ax)
    plt.xticks(rotation=0, ha='center')
    plt.title("Distribution of Pairwise Correlations (PBMZI Log Returns)", fontsize=16)
    plt.ylabel("Number of Correlation Pairs")
    plt.xlabel("\nCorrelation Category")

    # Label nilai atas bar
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + 1,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    st.pyplot(fig)


# =========================
# PAGE 2 – MST Overview
# =========================
elif page == "MST Overview":
    st.title("MST Overview")

    # Filter data for selected years
    filtered_data = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]

    if filtered_data.shape[0] > 1:
        # Calculate log returns
        row1, column1 = filtered_data.shape
        returns = np.log(filtered_data.iloc[1:row1, 1:column1].values /
                         filtered_data.iloc[0:row1-1, 1:column1].values)
        returns_df = pd.DataFrame(returns, columns=filtered_data.columns[1:])

        # Correlation matrix
        corremat = np.corrcoef(returns_df.T)

        # Distance matrix
        distmat = np.round(np.sqrt(2 * (1 - corremat)), 2)

        # Kruskal MST
        companies = returns_df.columns.tolist()
        G = nx.Graph()
        G.add_nodes_from(companies)
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                G.add_edge(companies[i], companies[j], weight=distmat[i, j])
        MST_PBMZI = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")

        # Node degree colors
        node_values = dict(MST_PBMZI.degree())
        values = np.array(list(node_values.values()))
        norm_nodes = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        cmap_nodes = cm.get_cmap("summer_r")
        node_colors = [cmap_nodes(norm_nodes(val)) for val in values]
        node_sizes = [v * 300 for v in node_values.values()]

        # Edge colors
        weights = nx.get_edge_attributes(MST_PBMZI, "weight").values()
        norm_edges = mcolors.Normalize(vmin=min(weights)-0.1, vmax=max(weights)+0.1)
        edge_colors = [(0, 0, 1, norm_edges(w)) for w in weights]
        sm_edges = cm.ScalarMappable(cmap=cm.Blues, norm=norm_edges)
        sm_edges.set_array([])

        # Plot MST
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.kamada_kawai_layout(MST_PBMZI)
        nx.draw(MST_PBMZI, pos,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                font_size=8.5,
                font_color="black",
                edge_color=edge_colors,
                ax=ax)

        # Colorbars
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        sm_nodes = cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
        sm_nodes.set_array([])
        cbar_nodes = fig.colorbar(sm_nodes, cax=cax)
        cbar_nodes.set_label("Node Degree (Connections)", rotation=270, labelpad=15, fontsize=12)

        cax2 = divider.append_axes("right", size="5%", pad=1.2)
        cbar_edges = fig.colorbar(sm_edges, cax=cax2)
        cbar_edges.set_label("Distance between Nodes", rotation=270, labelpad=15, fontsize=12)
        plt.title(f"Minimum Spanning Tree for PBMZI Log Return\n{', '.join(map(str, selected_years))}", size=16, loc='center', pad=10, x=-12)
        #plt.title(f"Minimum Spanning Tree for PBMZI Log Return\n{selected_years}", size=16, loc='left', pad=0)
        st.pyplot(fig)
    else:
        st.warning("Not enough data for the selected year to build MST.")
