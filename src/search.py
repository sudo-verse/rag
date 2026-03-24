import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model) if groq_api_key else ChatGroq(model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5, chat_history: list = None) -> tuple[str, list[str]]:
        results = self.vectorstore.query(query, top_k=top_k)
        if not results:
            return "Knowledge base is empty. Please upload some documents first.", []
            
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found.", []
            
        history_block = ""
        if chat_history:
            # formatting recent history for memory context
            recent = chat_history[-4:] 
            formatted_msgs = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent])
            history_block = f"\nRecent Conversation History:\n{formatted_msgs}\n"
            
        # Explicit instruction to strictly prevent hallucinations and force groundness
        prompt = f"""You are an assistant for a construction marketplace.
Answer the user's question explicitly and strictly using ONLY the provided context. 
If the context does not contain the answer, say "I cannot answer this based on the provided documents." Do not use your internal general knowledge.
{history_block}
Context:
{context}

Question: '{query}'

Answer:"""
        response_text = self.llm.invoke(prompt).content
        
        # If the LLM determines the chunks do not contain the answer, do not return the irrelevant chunks
        if "I cannot answer this" in response_text:
            return response_text, []
            
        return response_text, texts

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)