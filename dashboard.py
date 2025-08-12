# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE SETUP ---
st.set_page_config(page_title="PBMZI Dashboard", layout="wide")

# --- TITLE ---
st.title("PBMZI (2018-2023)")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Replace with your file path
    df = pd.read_excel("your_data.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# --- CLEANING (YOUR EXISTING CLEANING LOGIC) ---
# Assume your existing cleaning is already done here, ending with cleaned_PBMZI
cleaned_PBMZI = df.copy()  # Replace with your cleaned dataframe

# --- SIDEBAR FOR YEAR SELECTION ---
st.sidebar.header("Filter by Year")
years = list(range(2018, 2024))
selected_years = []
for year in years:
    if st.sidebar.checkbox(str(year), value=True):
        selected_years.append(year)

# --- FILTER DATA BASED ON SELECTION ---
if selected_years:
    filtered_df = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]
else:
    filtered_df = cleaned_PBMZI.copy()

# --- PRICE MOVEMENT ---
st.subheader("Price Movement Overview")
fig1, ax1 = plt.subplots(figsize=(14, 8))
for company in filtered_df.columns[1:]:
    ax1.plot(filtered_df['Date'], filtered_df[company], label=company)
ax1.legend()
ax1.set_title('Price Movement of PBMZI (Selected Years)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (~M$)')
plt.tight_layout()
st.pyplot(fig1)

# --- RETURN MOVEMENT ---
st.subheader("Logarithmic Return Movement Overview")
log_return_PBMZI = filtered_df[filtered_df.columns[1:]].apply(lambda col: np.log(col / col.shift(1)))
fig2, ax2 = plt.subplots(figsize=(14, 8))
for company in filtered_df.columns[1:]:
    ax2.plot(filtered_df['Date'], log_return_PBMZI[company], label=company)
ax2.legend()
ax2.set_title('Logarithmic Return Movement of PLCs (Selected Years)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Log Return (~M$)')
plt.tight_layout()
st.pyplot(fig2)

# --- VOLATILITY ---
st.subheader("Volatility (60-Day Rolling STD) Based on Raw Price")
volatility_price_raw = filtered_df[filtered_df.columns[1:]].rolling(window=60).std()
fig3, ax3 = plt.subplots(figsize=(14, 8))
for company in filtered_df.columns[1:]:
    ax3.plot(filtered_df['Date'], volatility_price_raw[company], label=company)
ax3.legend()
ax3.set_title('Volatility (60-Day Rolling STD) - Selected Years')
ax3.set_xlabel('Date')
ax3.set_ylabel('Rolling Std of Price (~M$)')
plt.tight_layout()
st.pyplot(fig3)

# --- CORRELATION ---
st.subheader("Correlation Matrix")
fig4, ax4 = plt.subplots(figsize=(10, 10))
sns.heatmap(
    filtered_df[filtered_df.columns[1:]].corr(method='pearson'),
    cmap='coolwarm_r',
    square=True,
    xticklabels=filtered_df.columns[1:], 
    yticklabels=filtered_df.columns[1:], 
    vmin=-1, vmax=1,
    ax=ax4
)
ax4.set_title("Correlation Matrix of PBMZI Companies' Stock Prices (Selected Years)", size=15, pad=20)
st.pyplot(fig4)
