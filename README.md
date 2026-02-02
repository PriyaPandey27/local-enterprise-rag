# Local Enterprise Multi-Source RAG

A 100% local, privacy-focused RAG system using Streamlit, LangChain, Ollama, and FAISS.

## Prerequisites

1.  **Ollama**: Ensure you have [Ollama](https://ollama.com/) installed.
2.  **Pull Models**: Run the following commands in your terminal:
    ```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ```
    *(Note: The app uses HuggingFace embeddings by default for better local CPU performance, but having a local embedding model in Ollama is good practice).*

## Setup

1.  Create a virtual environment:
    ```bash
    python -m venv rag_env
    ```

2.  Activate the environment:
    *   **Windows**: `rag_env\Scripts\activate`
    *   **Mac/Linux**: `source rag_env/bin/activate`

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2.  The application will open in your browser at `http://localhost:8501`.

## Usage

1.  **Ingestion**: Use the sidebar to upload PDF documents or CSV files, or enter a URL.
2.  **Process**: Click "Process & Index Sources" to build the local knowledge base. Wait for the "System Ready" indicator.
3.  **Query**: Type your question in the main search bar.
4.  **Results**: View the "Final Answer," "Source Citations," and "Retriever Insights" (showing exact text chunks retrieved).
