import numpy as np
import ollama


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, model):
        self.model = model

    def get_embedding(self, text, model, prompt):
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1, vector2):
        """
        calculate cosine similarity between two vectors
        :param vector1:
        :param vector2:
        :return:
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OllamaEmbedding(BaseEmbeddings):
    """
    class for ollama embeddings
    """
    def __init__(self, model):
        super().__init__(model)
        import ollama
        self.model = model

    def get_embedding(self, text, prompt, model='mxbai-embed-large'):
        response = ollama.embeddings(model=model, prompt=prompt)
        embedding = response["embedding"]
        return embedding
