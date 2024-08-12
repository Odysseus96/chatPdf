from typing import Dict, List, Any

import ollama
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel

class OllamaLocalEmbeddings(BaseModel, Embeddings):
    model: str
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = ollama.embed(self.model, texts)["embeddings"]
        return embedded_docs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def get_embeddings(self, texts):
        if isinstance(texts, str):
            return self.embed_query(texts)
        elif isinstance(texts, list):
            return self.embed_documents(texts)
        else:
            raise ValueError("只能为str或list格式")

