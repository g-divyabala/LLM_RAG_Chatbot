import streamlit as st
import ollama
import csv
import time
from pathlib import Path
from datetime import datetime
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
LOGS_PATH = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_PATH / "chat_logs.csv"

LOGS_PATH.mkdir(exist_ok=True)

st.set_page_config(
    page_title="RAG Knowledge Chatbot",
    page_icon="🤖"
)

st.title("🤖 RAG Knowledge Chatbot")
st.write("Ask questions based on the provided knowledge base.")


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


@st.cache_resource
def load_vector_database():
    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        embedding_model,
        allow_dangerous_deserialization=True
    )


vector_db = load_vector_database()


if not LOG_FILE.exists():
    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "query_id",
            "timestamp",
            "question",
            "retrieved_context",
            "answer",
            "retrieval_time_seconds",
            "generation_time_seconds",
            "total_latency_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "feedback"
        ])


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask a question about Python...")


if question:

    query_id = str(uuid.uuid4())

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    retrieval_start = time.time()

    results = vector_db.similarity_search(
        question,
        k=2
    )

    retrieval_time = time.time() - retrieval_start

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    generation_start = time.time()

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    generation_time = time.time() - generation_start

    answer = response["message"]["content"]

    prompt_tokens = response.get(
        "prompt_eval_count",
        0
    )

    completion_tokens = response.get(
        "eval_count",
        0
    )

    total_tokens = prompt_tokens + completion_tokens

    total_latency = retrieval_time + generation_time

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.caption(
        f"Retrieval time: {retrieval_time:.2f} seconds"
    )

    st.caption(
        f"Generation time: {generation_time:.2f} seconds"
    )

    st.caption(
        f"Total latency: {total_latency:.2f} seconds"
    )

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            query_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            context,
            answer,
            round(retrieval_time, 4),
            round(generation_time, 4),
            round(total_latency, 4),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            "not_provided"
        ])