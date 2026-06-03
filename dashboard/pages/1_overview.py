import streamlit as st
import pandas as pd

df = pd.read_csv("data/cleaned_bank_calls.csv")

st.title("Overview")
st.write("Dataset summary and KPIs")

st.write(f"Rows: {len(df)}")
#st.write(f"Fraud cases: {df['is_fraud'].sum()}")