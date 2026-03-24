# Construction Marketplace Mini RAG

This project implements a Retrieval-Augmented Generation (RAG) assistant for a construction marketplace. It allows users to upload internal documents ( policies, FAQs, specifications ), builds a local vector index, and grounds LLM-generated answers strictly within the retrieved context to ensure correctness and prevent hallucinations.

---

## 🏗️ Architecture & Implementation

### 1. Document Chunking & Retrieval
- **Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter`. Text is chunked with `chunk_size=1000` and `chunk_overlap=200`. This preserves necessary semantic context across sentence boundaries so the embedding representations are accurate.
- **Retrieval Engine**: Uses **FAISS** (Facebook AI Similarity Search) as the local vector store over standard cosine-similarity options to demonstrate scale-ready retrieval logic. For each query, the top-k most relevant chunks are fetched using $L2$ vector similarity rules.

### 2. Embedding Model
- **Model**: `all-MiniLM-L6-v2` (`sentence-transformers`)
- **Reasoning**: This open-source model operates seamlessly and entirely locally, offering an incredible tradeoff between speed and embedding accuracy, which makes it perfect for local CPU search loops while ensuring data privacy for proprietary marketplace data.

### 3. LLM-Based Generation
- **Model**: `llama-3.1-8b-instant` (via Groq API)
- **Reasoning**: This acts as a reliable OpenRouter facsimile, giving access to the extraordinarily fast LLaMA 3.1 8B parameter model for free. This checks the mandatory constraint of utilizing free open LLM structures while ensuring low latency generation.

### 4. Grounding & Transparency
- **Enforcing Grounded Context**: Hallucination explicitly avoided structurally. The system prompts the LLM with:
  > *Answer the user's question explicitly and strictly using ONLY the provided context. If the context does not contain the answer, say "I cannot answer this based on the provided documents." Do not use your internal general knowledge.*
- **Transparency GUI**: Every single response processed in the Streamlit frontend UI attaches a dropdown expander labeled **🔍 View Retrieved Context**. It displays the raw `[Chunk N]` outputs ingested by the LLM, making answers entirely explainable.

---

## 🚀 How to Run Locally

### Prerequisites
- Python `3.10` or newer
- `uv` Python package manager (or `pip`)

### Setup Instructions
1. Clone the repository and navigate into the `rag` directory.
2. Initialize your API key. Create a `.env` file in the root if you prefer to use one:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. Install all the dependencies automatically. 
   ```bash
   uv add -r requirements.txt
   ```
4. Run the Streamlit User Interface chatbot natively:
   ```bash
   uv run streamlit run frontend.py
   ```
5. Navigate to `http://localhost:8501`. Upload any set of PDFs, markdown formats, CSVs, or text documents and process the index!
