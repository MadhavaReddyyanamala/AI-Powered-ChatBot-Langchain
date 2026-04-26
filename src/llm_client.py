import os
from openai import OpenAI
from langsmith import traceable

def _client() -> OpenAI:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is missing. Add it to your .env file.")
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

@traceable(name="HuggingFace LLM Call", run_type="llm")
def hf_chat(prompt: str) -> str:
    model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful and helpful hospital assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()