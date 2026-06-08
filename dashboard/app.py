import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="📞",
    layout="wide"
)

st.title("Fraud Detection Dashboard")

st.write("Welcome to the Fraud Detection Dashboard. Use the sidebar to navigate between different sections of the analysis.")

st.markdown("""
### Available Sections:
- **Overview**: Dataset summary and KPIs.
- **Live Prediction**: Estimate fraud risk for a specific call.
- **Number Lookup**: Check history and risk for a specific phone number.
- **Trends**: Temporal analysis of fraud calls.
""")
