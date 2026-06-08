from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_model.joblib"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
FEATURE_COLUMNS_PATH = ROOT / "models" / "feature_columns.joblib"

st.set_page_config(
    page_title="Fraud Detection Dashboard", 
    page_icon="📞", 
    layout="wide"
)

st.title("Fraud Detection Dashboard")
st.write("Enter call details to estimate fraud risk.")

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
    st.error("Model files are missing.")
    st.code("python models/save_best_model.py", language="bash")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

with st.form("prediction_form"):
    st.subheader("Call details")

    duration_sec = st.number_input("Call duration, seconds", min_value=0, value=120)
    hour = st.slider("Hour of call", 0, 23, 12)
    weekday = st.slider("Weekday", 0, 6, 0, help="0 = Monday, 6 = Sunday")
    manual_severity = st.slider("Manual severity", 0, 5, 0)

    caller_number_prefix_freq = st.number_input(
        "Caller number prefix frequency",
        min_value=0.0,
        value=1.0,
    )

    channel = st.selectbox("Channel", ["Mobile", "VoIP", "Other"])

    st.subheader("Call and action flags")

    was_answered = st.checkbox("Call was answered", value=True)
    was_hangup_by_client = st.checkbox("Client hung up")
    login_after_call = st.checkbox("Login after call")
    twofa_confirmed_after_call = st.checkbox("2FA confirmed after call")
    transfer_after_call = st.checkbox("Transfer after call")
    new_payee_after_call = st.checkbox("New payee after call")
    settings_changed_after_call = st.checkbox("Settings changed after call")

    time_to_next_action_min = st.number_input(
        "Time to next action, minutes",
        min_value=0.0,
        value=10.0,
    )

    submitted = st.form_submit_button("Predict fraud risk")


if submitted:
    is_weekend = int(weekday >= 5)
    is_business_hours = int(9 <= hour <= 17)

    input_data = {
        "duration_sec": duration_sec,
        "was_answered": int(was_answered),
        "was_hangup_by_client": int(was_hangup_by_client),
        "login_after_call": int(login_after_call),
        "twofa_confirmed_after_call": int(twofa_confirmed_after_call),
        "transfer_after_call": int(transfer_after_call),
        "new_payee_after_call": int(new_payee_after_call),
        "settings_changed_after_call": int(settings_changed_after_call),
        "time_to_next_action_min": time_to_next_action_min,
        "manual_severity": manual_severity,
        "hour": hour,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "is_business_hours": is_business_hours,
        "caller_number_prefix_freq": caller_number_prefix_freq,
        "channel_Mobile": int(channel == "Mobile"),
        "channel_VoIP": int(channel == "VoIP"),
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    model_input = input_df

    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)

        if scaler_features == feature_columns:
            model_input = scaler.transform(input_df)

    prediction = model.predict(model_input)[0]
    probabilities = model.predict_proba(model_input)[0]
    fraud_probability = probabilities[1]
    confidence = max(probabilities)

    if prediction == 1:
        st.error("Prediction: Fraud risk detected")
    else:
        st.success("Prediction: No fraud detected")

    st.metric("Fraud probability", f"{fraud_probability:.2%}")
    st.metric("Confidence", f"{confidence:.2%}")

    with st.expander("Input features"):
        st.dataframe(input_df)
