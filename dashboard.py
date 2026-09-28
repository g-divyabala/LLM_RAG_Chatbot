import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "chat_logs.csv"

st.set_page_config(
    page_title="RAG Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Chatbot Monitoring Dashboard")

if not LOG_FILE.exists():
    st.warning("No monitoring data found yet.")
    st.stop()

df = pd.read_csv(LOG_FILE)

if df.empty:
    st.info("No chatbot interactions have been recorded yet.")
    st.stop()

df["retrieval_time_seconds"] = pd.to_numeric(
    df["retrieval_time_seconds"],
    errors="coerce"
)

df["generation_time_seconds"] = pd.to_numeric(
    df["generation_time_seconds"],
    errors="coerce"
)

df["total_latency_seconds"] = pd.to_numeric(
    df["total_latency_seconds"],
    errors="coerce"
)

df["total_tokens"] = pd.to_numeric(
    df["total_tokens"],
    errors="coerce"
)

total_queries = len(df)

average_latency = df["total_latency_seconds"].mean()

average_retrieval_time = df["retrieval_time_seconds"].mean()

average_generation_time = df["generation_time_seconds"].mean()

total_tokens = df["total_tokens"].sum()

positive_feedback = (
    df["feedback"] == "positive"
).sum()

negative_feedback = (
    df["feedback"] == "negative"
).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Queries",
        total_queries
    )

with col2:
    st.metric(
        "Average Latency",
        f"{average_latency:.2f} sec"
    )

with col3:
    st.metric(
        "Total Tokens",
        int(total_tokens)
    )

with col4:
    st.metric(
        "Positive Feedback",
        positive_feedback
    )

st.divider()

st.subheader("Response Performance")

performance_data = pd.DataFrame({
    "Metric": [
        "Retrieval Time",
        "Generation Time",
        "Total Latency"
    ],
    "Seconds": [
        average_retrieval_time,
        average_generation_time,
        average_latency
    ]
})

fig_latency = px.bar(
    performance_data,
    x="Metric",
    y="Seconds",
    title="Average Response Time"
)

st.plotly_chart(
    fig_latency,
    use_container_width=True
)

st.subheader("Token Usage")

fig_tokens = px.bar(
    df,
    x="timestamp",
    y="total_tokens",
    title="Token Usage Per Query"
)

st.plotly_chart(
    fig_tokens,
    use_container_width=True
)

st.subheader("User Feedback")

feedback_data = pd.DataFrame({
    "Feedback": [
        "Positive",
        "Negative"
    ],
    "Count": [
        positive_feedback,
        negative_feedback
    ]
})

fig_feedback = px.pie(
    feedback_data,
    names="Feedback",
    values="Count",
    title="User Feedback"
)

st.plotly_chart(
    fig_feedback,
    use_container_width=True
)

st.subheader("Interaction Logs")

st.dataframe(
    df,
    use_container_width=True
)