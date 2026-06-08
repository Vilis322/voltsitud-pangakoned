import streamlit as st

from core.data import load_call_data, lookup_number_history
from core.model import lookup_risk_info

st.title("Number Lookup")

calls = load_call_data()
phone_query = st.text_input("Phone number", placeholder="Enter caller number or prefix")

if not phone_query:
    st.info("Enter a phone number to see matching history and risk score.")
else:
    history = lookup_number_history(calls, phone_query)
    risk = lookup_risk_info(history)

    if history.empty:
        st.warning("No matching call history found for that number.")
    else:
        st.metric(
            "Aggregate risk score",
            f"{risk['confidence_score']:.2%}" if risk["confidence_score"] is not None else "N/A",
            risk["risk_level"],
        )

        st.markdown("### Caller history")
        st.dataframe(
            history[
                [
                    "timestamp",
                    "call_id",
                    "caller_number",
                    "called_number",
                    "channel",
                    "label_5way",
                    "confidence_score",
                ]
            ]
            .sort_values("timestamp", ascending=False)
            .reset_index(drop=True)
        )

        st.markdown("### Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Calls found**")
            st.write(len(history))
            st.write("**Most common label**")
            st.write(history["label_5way"].mode().iloc[0])
        with col2:
            st.write("**Most common channel**")
            st.write(history["channel"].mode().iloc[0])
            st.write("**Average duration (sec)**")
            st.write(round(history["duration_sec"].mean(), 1))