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
    
## Demo
<img width="500" height="400" alt="Screenshot 2026-02-02 210525" src="https://github.com/user-attachments/assets/8020c483-06aa-4ba7-9051-e2b2a58057f8" />
<img width="500" height="400" alt="Screenshot 2026-02-02 210802" src="https://github.com/user-attachments/assets/dfa2a4d3-612b-4f66-9acf-84380a74c686" />


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

2.  The application will open in your browser.

## Usage

1.  **Ingestion**: Use the sidebar to upload PDF documents or CSV files, or enter a URL.
2.  **Process**: Click "Process & Index Sources" to build the local knowledge base. Wait for the "System Ready" indicator.
3.  **Query**: Type your question in the main search bar.
4.  **Results**: View the "Final Answer," "Source Citations," and "Retriever Insights" (showing exact text chunks retrieved).





