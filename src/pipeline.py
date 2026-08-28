"""
Top-level orchestration: build the index once, then answer any number
of queries against it.
"""

from src.ingestion import load_all_pdfs, split_chunks
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retrieval import RAGRetriever
from src.llm import get_llm, generate_output


class RAGPipeline:
    def __init__(self, pdf_dir: str = "data/pdfs", persist_directory: str = "vector_store"):
        self.pdf_dir = pdf_dir
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager(persist_directory=persist_directory)
        self.retriever = RAGRetriever(self.embedding_manager, self.vector_store)
        self.llm = get_llm()

    def build_index(self):
        """Ingest PDFs, chunk them, embed them, and load them into the vector store.
        Skips re-embedding if the collection is already populated."""
        if self.vector_store.collection.count() > 0:
            print("vector store already populated, skipping ingestion")
            return

        docs = load_all_pdfs(self.pdf_dir)
        chunks = split_chunks(docs)
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vector_store.add_documents(chunks, embeddings)

    def answer(self, query: str, top_k: int = 3) -> str:
        return generate_output(query, self.retriever, self.llm, top_k=top_k)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.build_index()
    print(pipeline.answer("What is Retrieval-Augmented Generation?"))
