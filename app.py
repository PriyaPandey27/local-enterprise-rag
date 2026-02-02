import os
os.environ["USER_AGENT"] = "EnterpriseRAG/1.0"

import streamlit as st

import tempfile
import time
from typing import List

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# ----------------------------
# Simple Ensemble Retriever
# ----------------------------
class SimpleEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        seen = set()
        results = []

        for retriever in self.retrievers:
            docs = retriever.invoke(query)  # ✅ FIX HERE
            for doc in docs:
                doc_id = (doc.page_content, str(doc.metadata))
                if doc_id not in seen:
                    seen.add(doc_id)
                    results.append(doc)

        return results


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Enterprise Knowledge Base",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session State
# ----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("Knowledge Center")

    uploaded_files = st.file_uploader(
        "Upload Documents", type=["pdf", "csv"], accept_multiple_files=True
    )
    url_input = st.text_input("Add Web URL")
    process_btn = st.button("Process & Index Sources")

# ----------------------------
# Ingestion Functions
# ----------------------------
def process_documents(files, url):
    docs = []

    if files:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            try:
                if file.name.endswith(".pdf"):
                    docs.extend(PyPDFLoader(tmp_path).load())
                elif file.name.endswith(".csv"):
                    docs.extend(CSVLoader(tmp_path).load())
            finally:
                os.remove(tmp_path)

    if url:
        docs.extend(WebBaseLoader(url).load())

    return docs


def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store, splits


# ----------------------------
# Processing Trigger
# ----------------------------
if process_btn:
    if not uploaded_files and not url_input:
        st.warning("Please upload files or enter a URL.")
    else:
        with st.spinner("Ingesting and indexing..."):
            raw_docs = process_documents(uploaded_files, url_input)

            if not raw_docs:
                st.error("No documents processed.")
            else:
                vector_store, splits = build_vector_store(raw_docs)

                # Retrievers
                faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                bm25_retriever = BM25Retriever.from_documents(splits)
                bm25_retriever.k = 5

                # Manual Ensemble
                ensemble_retriever = SimpleEnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever]
                )

                # LLM
                llm = ChatOllama(model="llama3.2")

                prompt = ChatPromptTemplate.from_template(
                    """Answer the question using ONLY the context below:
                    {context}

                    Question: {question}
                    """
                )

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain_from_docs = (
                    RunnablePassthrough.assign(
                        context=lambda x: format_docs(x["context"])
                    )
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                rag_chain = RunnableParallel(
                    {
                        "context": ensemble_retriever,
                        "question": RunnablePassthrough(),
                    }
                ).assign(answer=rag_chain_from_docs)

                st.session_state.rag_chain = rag_chain
                st.session_state.documents_loaded = True
                st.success("Knowledge base built successfully!")

# ----------------------------
# Main UI
# ----------------------------
st.markdown('<h1 class="main-header">Enterprise Intelligence Hub</h1>', unsafe_allow_html=True)

query = st.text_input("Ask a question")

if query and st.session_state.documents_loaded:
    with st.spinner("Thinking..."):
        result = st.session_state.rag_chain.invoke(query)

        st.markdown("### 💡 Answer")
        st.markdown(
            f"<div class='card'>{result['answer']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 📚 Sources")
        for doc in result["context"]:
            st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")
