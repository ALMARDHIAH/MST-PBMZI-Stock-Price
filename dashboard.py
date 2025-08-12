import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("cleaned_PBMZI.xlsx")
        df['Date'] = pd.to_datetime(df['Date'])
        # Ensure all columns after 'Date' are numeric, coerce errors to NaN
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except FileNotFoundError:
        st.error("The data file 'cleaned_PBMZI.xlsx' was not found in the working directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

cleaned_PBMZI = load_data()

if cleaned_PBMZI.empty:
    st.stop()

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
        if not selected_years:
            st.warning("Select at least one year.")
            st.stop()
    with col2:
        st.markdown("### Companies")
        st.write(list(cleaned_PBMZI.columns[1:]))

    # Filtered data
    filtered_df = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]
    if filtered_df.empty:
        st.warning("No data for the selected year(s).")
        st.stop()

    with col3:
        st.subheader("Price Movement")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        for company in filtered_df.columns[1:]:
            if pd.api.types.is_numeric_dtype(filtered_df[company]):
                ax1.plot(filtered_df['Date'], filtered_df[company], label=company)
        ax1.legend(fontsize=6)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (~M$)')
        st.pyplot(fig1)

    with col4:
        st.subheader("Return Movement")
        # Log return, handle zeros or negatives to avoid log error
        valid = (filtered_df[filtered_df.columns[1:]] > 0).all()
        if not valid.any():
            st.warning("No valid positive price data for log return.")
        else:
            log_return = filtered_df[filtered_df.columns[1:]].apply(
                lambda col: np.log(col / col.shift(1)) if (col > 0).all() else np.nan
            )
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
        log_return_filtered = filtered_df[filtered_df.columns[1:]].apply(
            lambda col: np.log(col / col.shift(1)) if (col > 0).all() else np.nan
        )
        if log_return_filtered.dropna(axis=1, how="all").shape[1] < 2:
            st.warning("Not enough valid data for correlation matrix.")
        else:
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
        if df.shape[0] < 2 or df.shape[1] < 3:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            return fig
        # Only use positive values for log
        df_valid = df[(df.iloc[:, 1:] > 0).all(axis=1)].reset_index(drop=True)
        if df_valid.shape[0] < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough valid data", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            return fig
        row1, col1 = df_valid.shape
        returns = np.log(df_valid.iloc[1:row1, 1:col1].values / df_valid.iloc[0:row1-1, 1:col1].values)
        returns_df = pd.DataFrame(returns, columns=df_valid.columns[1:])
        if returns_df.isnull().all().all():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough valid returns", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            return fig
        corremat = np.corrcoef(returns_df.T)
        distmat = np.round
