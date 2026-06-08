import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("Fraud Detection Dashboard")
st.write("Fraud risk prediction and interactive analytics.")


@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns


@st.cache_data
def load_dashboard_data():
    return pd.read_csv(DATA_PATH)


if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
    st.error("Model files are missing. Run this command first:")
    st.code("python models/save_best_model.py", language="bash")
    st.stop()


model, scaler, feature_columns = load_model_files()

st.header("Fraud Risk Prediction")

with st.form("prediction_form"):
    st.subheader("Call details")

    col1, col2, col3 = st.columns(3)

    with col1:
        duration_sec = st.number_input("Call duration, seconds", min_value=0, value=120)
        hour = st.slider("Hour of call", 0, 23, 12)
        weekday = st.slider("Weekday", 0, 6, 0)

    with col2:
        manual_severity = st.slider("Manual severity", 0, 5, 0)
        caller_number_prefix_freq = st.number_input(
            "Caller number prefix frequency",
            min_value=0.0,
            value=1.0,
        )
        channel = st.selectbox("Channel", ["Mobile", "VoIP", "Other"])

    with col3:
        time_to_next_action_min = st.number_input(
            "Time to next action, minutes",
            min_value=0.0,
            value=10.0,
        )

    st.subheader("Action flags")

    flag_col1, flag_col2, flag_col3 = st.columns(3)

    with flag_col1:
        was_answered = st.checkbox("Call was answered", value=True)
        was_hangup_by_client = st.checkbox("Client hung up")

    with flag_col2:
        login_after_call = st.checkbox("Login after call")
        twofa_confirmed_after_call = st.checkbox("2FA confirmed after call")

    with flag_col3:
        transfer_after_call = st.checkbox("Transfer after call")
        new_payee_after_call = st.checkbox("New payee after call")
        settings_changed_after_call = st.checkbox("Settings changed after call")

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

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        if prediction == 1:
            st.error("Prediction: Fraud risk detected")
        else:
            st.success("Prediction: No fraud detected")

    with result_col2:
        st.metric("Fraud probability", f"{fraud_probability:.2%}")

    with result_col3:
        st.metric("Confidence", f"{confidence:.2%}")

    st.progress(float(fraud_probability))

    if fraud_probability < 0.3:
        st.success("Risk level: Low")
    elif fraud_probability < 0.7:
        st.warning("Risk level: Medium")
    else:
        st.error("Risk level: High")

    with st.expander("Input features"):
        st.dataframe(input_df)


st.header("Fraud Analytics")

if not DATA_PATH.exists():
    st.warning("Dashboard dataset not found.")
    st.code("data/labeled_bank_calls.csv", language="text")
    st.stop()

df = load_dashboard_data()

if "is_fraud" not in df.columns:
    st.error("Column 'is_fraud' is missing from the dataset.")
    st.stop()


chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Label Distribution")

    label_counts = df["is_fraud"].value_counts().reset_index()
    label_counts.columns = ["is_fraud", "count"]
    label_counts["label"] = label_counts["is_fraud"].map({
        0: "Not fraud",
        1: "Fraud"
    }).fillna(label_counts["is_fraud"].astype(str))

    fig = px.pie(
        label_counts,
        names="label",
        values="count",
        title="Fraud vs Not Fraud"
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Fraud by Hour")

    if "hour" in df.columns:
        fraud_by_hour = (
            df.groupby("hour")["is_fraud"]
            .mean()
            .reset_index()
        )

        fig = px.density_heatmap(
            fraud_by_hour,
            x="hour",
            y="is_fraud",
            z="is_fraud",
            nbinsx=24,
            title="Fraud Rate Heatmap by Hour",
            labels={
                "hour": "Hour",
                "is_fraud": "Fraud rate"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'hour' is missing.")


st.subheader("Top Suspicious Numbers")

number_columns = [
    "caller_number",
    "phone_number",
    "number",
    "caller",
    "from_number"
]

number_column = next((col for col in number_columns if col in df.columns), None)

if number_column:
    suspicious_numbers = (
        df.groupby(number_column)
        .agg(
            total_calls=(number_column, "count"),
            fraud_cases=("is_fraud", "sum"),
            fraud_rate=("is_fraud", "mean"),
        )
        .reset_index()
        .sort_values(["fraud_rate", "fraud_cases", "total_calls"], ascending=False)
        .head(10)
    )

    fig = px.bar(
        suspicious_numbers,
        x=number_column,
        y="fraud_rate",
        hover_data=["total_calls", "fraud_cases"],
        title="Top Suspicious Numbers by Fraud Rate",
        labels={
            number_column: "Number",
            "fraud_rate": "Fraud rate"
        }
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No caller number column found in the dataset.")


st.subheader("Feature Importance")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values("importance", ascending=False).head(15)

    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Feature Importances",
        labels={
            "importance": "Importance",
            "feature": "Feature"
        }
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance is not available for this model.")
