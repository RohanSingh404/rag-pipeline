"""
Streamlit demo for the RAG pipeline.
Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAG PDF Q&A", page_icon="ð")
st.title("ð RAG Pipeline — PDF Q&A")
st.caption("Ingestion → Chunking → Chroma retrieval → Gemini generation")


@st.cache_resource(show_spinner="Loading pipeline and building index...")
def load_pipeline():
    pipeline = RAGPipeline()
    pipeline.build_index()
    return pipeline


pipeline = load_pipeline()

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Retrieving context and generating answer..."):
        answer = pipeline.answer(query)
    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Show retrieved context"):
        results = pipeline.retriever.retrieve(query, top_k=3)
        for r in results:
            st.markdown(f"**Rank {r['rank']} · similarity {r['similarity_score']:.2f}**")
            st.text(r["document"][:500])
            st.divider()
