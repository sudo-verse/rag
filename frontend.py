import os
import shutil
import time
import streamlit as st
from src.search import RAGSearch

# Config
DATA_DIR = "data"
FAISS_DIR = "faiss_store"

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"  # Automatically collapses sidebar on mobile
)

# Custom Styling (Dark/Modern & Responsive)
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    div.stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Customize the file uploader */
    div[data-testid="stFileUploader"] {
        padding: 10px;
        border-radius: 12px;
        border: 1px dotted #3d4b60;
    }
    
    /* Title modern look */
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #f1f2f6;
        font-size: 2.5rem;
    }

    /* Mobile Responsive CSS */
    @media screen and (max-width: 768px) {
        .title-text {
            font-size: 1.8rem;
            text-align: center;
        }
        div.stChatFloatingInputContainer {
            padding-bottom: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title Area
st.markdown("<h1 class='title-text'>🤖 Intelligent Document Q&A</h1>", unsafe_allow_html=True)
st.markdown("Upload your documents securely and ask intelligent queries using state-of-the-art AI.")
st.divider()

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Session State for Conversation and Engine
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG Engine only once
if "searcher" not in st.session_state:
    with st.spinner("Initializing AI Engine..."):
        try:
            st.session_state.searcher = RAGSearch(persist_dir=FAISS_DIR)
        except Exception as e:
            st.error(f"Failed to initialize AI Engine: {e}")

# Sidebar
with st.sidebar:
    st.header("📂 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload new files into the knowledge base", 
        type=["pdf", "txt", "csv", "docx", "xlsx", "json", "md"],
        accept_multiple_files=True
    )
    
    if st.button("Process & Rebuild Index", type="primary", use_container_width=True):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save uploaded files to the data directory
            total_files = len(uploaded_files)
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Saving {uploaded_file.name}...")
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                progress_bar.progress((i + 1) / total_files * 0.3)
                
            status_text.text("Rebuilding Vector Knowledge Base... This may take a while.")
            
            # Remove old faiss index to force rebuild
            if os.path.exists(FAISS_DIR):
                shutil.rmtree(FAISS_DIR)
            
            # Recreate RAGSearch to trigger building the vector database
            start_time = time.time()
            st.session_state.searcher = RAGSearch(persist_dir=FAISS_DIR)
            
            progress_bar.progress(1.0)
            status_text.text(f"Indexing complete in {time.time() - start_time:.1f}s!")
            st.success("Indexing perfectly complete! You can now ask questions about your documents.")
        else:
            st.warning("Please upload at least one file to re-index.")

    st.divider()
    
    st.markdown("### 📚 Supported Files Loaded")
    try:
        files_in_data = os.listdir(DATA_DIR)
        valid_extensions = ('.pdf', '.txt', '.csv', '.docx', '.xlsx', '.json', '.md')
        # Simple rendering of loaded files
        filtered_files = [f for f in files_in_data if f.endswith(valid_extensions)]
        if not filtered_files:
            st.info("No supported documents currently loaded.")
        else:
            for f in filtered_files:
                st.markdown(f"- `{f}`")
    except Exception as e:
        st.error(f"Could not read data directory: {e}")

# Display previously typed chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message and message["context"]:
            with st.expander("🔍 View Retrieved Context (Transparency & Explainability)"):
                st.markdown("**The following document chunks were retrieved and used to generate this answer:**")
                for i, chunk in enumerate(message["context"]):
                    st.info(f"**Chunk {i+1}:**\n\n{chunk}")

# Chat input container
if query := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Supply recent message history (excluding current user query)
                recent_history = [m for m in st.session_state.messages[:-1] if m.get("role") in ["user", "assistant"]]
                response, context_chunks = st.session_state.searcher.search_and_summarize(query, chat_history=recent_history)
                message_placeholder.markdown(response)
                
                # Render transparency context drawer directly
                if context_chunks:
                    with st.expander("🔍 View Retrieved Context (Transparency & Explainability)"):
                        st.markdown("**The following document chunks were retrieved and used to generate this answer:**")
                        for i, chunk in enumerate(context_chunks):
                            st.info(f"**Chunk {i+1}:**\n\n{chunk}")

                # Keep history
                st.session_state.messages.append({"role": "assistant", "content": response, "context": context_chunks})
            except Exception as e:
                error_msg = f"Error during search: {e}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
