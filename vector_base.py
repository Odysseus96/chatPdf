import os
from typing import Dict, List, Optional, Tuple, Union
import json
from embedding import BaseEmbeddings, OllamaEmbedding
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document):
        self.document = document

    def get_vector(self, embeddingModel):
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(embeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1, vector2):
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query, embeddingModel, k=1):
        query_vector = embeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector) for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
