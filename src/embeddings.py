"""
Embedding generation using a local Sentence-Transformers model
(no API key required for this step).
"""

from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print("loading embedding model:", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("embedding dimensions:", self.model.get_sentence_embedding_dimension())

    def generate_embeddings(self, texts):
        """texts: a list of strings (or a single-item list for a query)."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("embeddings shape:", embeddings.shape)
        return embeddings
