import os
import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable

from src.router import classify_query
from src.pdf_rag import get_retriever
from src.llm_client import hf_chat
from src.prompts import general_prompt, problem_prompt, hospital_prompt

load_dotenv()

st.set_page_config(page_title="CityMed Hospital Chatbot", page_icon="🏥")
st.title("🏥 CityMed Hospital Chatbot")
st.caption("Welcome To CityMed Hospital")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    try:
        st.session_state.retriever = get_retriever("data")
        st.success("Welcome....💖🙏🏻")
    except Exception as exc:
        st.session_state.retriever = None
        st.warning(f"RAG is not ready yet: {exc}")


@traceable(name="Hospital Chatbot Pipeline", run_type="chain")
def process_query(user_input, history_text, retriever):
    route = classify_query(user_input)

    if route == "GENERAL":
        answer = hf_chat(general_prompt(user_input, history_text))
        sources = []

    elif route == "PROBLEM":
        answer = hf_chat(problem_prompt(user_input, history_text))
        sources = []

    else:
        if retriever is None:
            answer = (
                "I could not load the hospital PDF knowledge base. "
                "Please add text-based PDFs to the data folder and restart the app."
            )
            sources = []
        else:
            docs = retriever.retrieve(user_input, top_k=5)

            context = "\n\n".join([
                f"[Source: {d['source']}, page {d['page']}]\n{d['text']}"
                for d in docs
            ])

            answer = hf_chat(hospital_prompt(user_input, history_text, context))
            sources = docs

    return route, answer, sources


with st.sidebar:
    st.header("Configuration")
    st.write("Provider:", os.getenv("HF_PROVIDER", "hf-inference"))
    st.write("Chat model:", os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    st.write("Embedding model:", os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask about symptoms, doctors, timings, or hospital location...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    history_text = "\n".join([
        f'{m["role"]}: {m["content"]}'
        for m in st.session_state.messages[-8:]
    ])

    try:
        route, answer, sources = process_query(
            user_input=user_input,
            history_text=history_text,
            retriever=st.session_state.retriever
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

            if sources:
                with st.expander("Sources used"):
                    for i, doc in enumerate(sources, start=1):
                        st.markdown(f"**{i}. {doc['source']} — page {doc['page']}**")
                        st.write(
                            doc["text"][:500] +
                            ("..." if len(doc["text"]) > 500 else "")
                        )

    except Exception as exc:
        err = f"Error: {exc}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": err
        })

        with st.chat_message("assistant"):
            st.error(err)