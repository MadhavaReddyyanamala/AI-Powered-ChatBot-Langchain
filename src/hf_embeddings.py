import os
from typing import List
import numpy as np
from huggingface_hub import InferenceClient
from langsmith import traceable

class HFEmbeddings:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is missing. Add it to your .env file.")
        self.model = os.getenv("HF_EMBEDDING_MODEL", "thenlper/gte-large")
        self.provider = os.getenv("HF_PROVIDER", "hf-inference")
        self.client = InferenceClient(provider=self.provider, api_key=token)

    @traceable(name="Embedding Documents", run_type="embedding")
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vectors = self.client.feature_extraction(texts, model=self.model)
        arr = np.array(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @traceable(name="Embedding Query", run_type="embedding")
    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.client.feature_extraction(text, model=self.model)
        arr = np.array(vectors, dtype="float32")
        if arr.ndim > 1:
            # some providers may return nested arrays even for one input
            arr = arr[0]
        return arr.reshape(1, -1)