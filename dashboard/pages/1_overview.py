import streamlit as st

from core.data import (
    available_channels,
    available_date_range,
    available_labels,
    filter_calls,
    load_call_data,
)

st.title("Overview")
st.write("Dataset filters and summary statistics")

df = load_call_data()
if df.empty:
    st.warning("No call data available.")
    st.stop()

start_date, end_date = available_date_range(df)

with st.sidebar:
    st.header("Filters")
    date_range = st.date_input(
        "Date range",
        [start_date, end_date],
        min_value=start_date,
        max_value=end_date,
    )

    selected_labels = st.multiselect(
        "Label class",
        available_labels(df),
        default=available_labels(df),
    )

    selected_channels = st.multiselect(
        "Channel",
        available_channels(df),
        default=available_channels(df),
    )

    prefix = st.text_input("Caller number prefix")

if isinstance(date_range, tuple) or isinstance(date_range, list):
    start_date, end_date = date_range[0], date_range[-1]

filtered = filter_calls(
    df,
    start_date=start_date,
    end_date=end_date,
    labels=selected_labels,
    caller_prefix=prefix,
    channels=selected_channels,
)

st.subheader("Filtered view")
st.write(f"Total rows: {len(filtered)}")

if filtered.empty:
    st.info("No rows match the selected filters.")
else:
    st.markdown("**Top labels in filtered sample**")
    st.table(filtered["label_5way"].value_counts().rename_axis("label").reset_index(name="count"))

    st.markdown("**Top channels in filtered sample**")
    st.table(filtered["channel"].value_counts().rename_axis("channel").reset_index(name="count"))

    st.subheader("Top risk calls")
    st.dataframe(
        filtered.sort_values("confidence_score", ascending=False)
        .head(15)
        [["timestamp", "caller_number", "called_number", "label_5way", "channel", "confidence_score"]]
        .reset_index(drop=True)
    )