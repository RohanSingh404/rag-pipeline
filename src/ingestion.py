"""
Ingestion pipeline: loads PDFs from a directory and splits them into
overlapping text chunks ready for embedding.
"""

import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_all_pdfs(path_dir: str = "data/pdfs"):
    """Load every PDF in a directory into LangChain Document objects."""
    all_docs = []
    num_docs = 0

    for file_name in os.listdir(path_dir):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(path_dir, file_name)
            loader = PyPDFLoader(file_path)
            document = loader.load()
            all_docs.extend(document)
            num_docs += 1

    print(f"number of PDF files loaded: {num_docs}")
    print(f"total pages loaded: {len(all_docs)}")
    return all_docs


def split_chunks(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into overlapping chunks using recursive character splitting."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"created {len(chunked_docs)} chunks from {len(documents)} pages")
    return chunked_docs


if __name__ == "__main__":
    docs = load_all_pdfs()
    chunks = split_chunks(docs)
