# AI-Powered-ChatBot-Langchain

# 🏥 Hospital AI Chatbot

An intelligent hospital assistant powered by **Hugging Face Inference API** and **Retrieval-Augmented Generation (RAG)**. The chatbot answers general health queries, provides symptom guidance, and retrieves hospital-specific information directly from your PDF documents.

---

## ✨ Features

- **General query handling** — answers common medical knowledge questions
- **Symptom guidance** — provides safe, informative responses to symptom-based queries
- **Hospital information retrieval** — fetches doctor details, timings, and hospital info from PDFs using RAG
- **Conversation memory** — maintains context across the session via Streamlit state
- **Responsible AI** — does not diagnose or prescribe; gives general and hospital information only

---

## 🗂️ Project Structure

```
<img width="604" height="764" alt="image" src="https://github.com/user-attachments/assets/5e818627-9892-4289-8ceb-61a01ebcb563" />

```

---

## ⚙️ How It Works

```
User Query
    │
    ▼
 Router (router.py)
    │
    ├── GENERAL ──────────────────────► Chat model → Answer
    │
    ├── PROBLEM ──────────────────────► Chat model (safe guidance) → Answer
    │
    └── HOSPITAL_INFO ────► PDF Chunks + FAISS (pdf_rag.py)
                                 │
                            Embeddings (hf_embeddings.py)
                                 │
                            Chat model (llm_client.py) → Answer
```

---

## 🚀 Getting Started

### 1. Create a Hugging Face Access Token

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **User Access Token** with permission to call Inference Providers.

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
HF_CHAT_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_EMBEDDING_MODEL=thenlper/gte-large
HF_PROVIDER=hf-inference
```

> **Notes:**
> - `HF_CHAT_MODEL` must support chat completion on Hugging Face Inference Providers.
> - `HF_EMBEDDING_MODEL` must support feature extraction / embeddings.
> - If a model is unavailable for your provider, choose an alternative from its Hugging Face model page.

### 3. Add your hospital PDFs

Place your hospital documents inside the `data/` folder. Recommended files:

| File | Contents |
|------|----------|
| `doctors.pdf` | Doctor names, specialties, contact info |
| `timings.pdf` | OPD and department schedules |
| `hospital_info.pdf` | Location, facilities, departments |

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 💬 Example Queries

**General questions**
```
What is fever?
What are common causes of a cough?
```

**Symptom guidance**
```
I have a headache and fever.
I feel dizzy and weak.
```

**Hospital information**
```
Who is the cardiologist?
What is Dr. Ravi's timing?
Where is the hospital located?
Which doctor should I consult for a skin allergy?
```

---

## ⚠️ Safety Disclaimer

This chatbot is designed to provide **general health information** and **hospital-specific details only**.

- It does **not** diagnose medical conditions.
- It does **not** prescribe medication or treatment.
- Always consult a qualified medical professional for personal health concerns.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | [Streamlit](https://streamlit.io) |
| LLM | Hugging Face Inference API |
| Embeddings | HuggingFace `thenlper/gte-large` |
| Vector Store | FAISS |
| PDF Parsing | LangChain / PyMuPDF |
| RAG Framework | Custom pipeline |

---

## 📄 License

This project is for educational and informational purposes only.
