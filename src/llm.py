"""
LLM setup (Google Gemini via LangChain) and the final RAG generation step.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(model: str = "gemini-2.5-flash", temperature: float = 0.1, max_output_tokens: int = 1024):
    api_key = os.environ.get("API_KEY_GEMINI")
    if not api_key:
        raise ValueError("API_KEY_GEMINI not set. Copy .env.example to .env and add your key.")

    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def generate_output(query: str, retriever, llm, top_k: int = 3):
    """Retrieve relevant chunks, build a context-grounded prompt, and query the LLM."""
    results = retriever.retrieve(query, top_k)

    context = "\n".join([doc["document"] for doc in results]) if results else ""

    if not context:
        print("no relevant context found for the given query")

    prompt = f"""Use the given context to answer the query. If the context does not contain
the answer, say you don't have enough information rather than guessing.

Context: {context}
Query: {query}"""

    response = llm.invoke(prompt)
    return response.content
