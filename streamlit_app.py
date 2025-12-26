import streamlit as st
import base64

# Internal Module Imports
from src.rag.hybrid_store import HybridStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.config import settings
from src.rag.logger import get_logger

# Setup logging
logger = get_logger(__name__)

# --- UI Setup & Styling ---
st.set_page_config(page_title="Privacy Advice", layout="centered", page_icon="🤖")

# Custom CSS for a dark mode interface
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    .user-msg { 
        background-color: #2C3E50; color: #FFFFFF; padding: 15px; 
        border-radius: 15px; margin-bottom: 10px; margin-left: auto; width: fit-content; max-width: 85%; 
        border-bottom-right-radius: 2px;
    }
    .ai-msg { 
        background-color: #1E1E1E; color: #F0F0F0; padding: 15px; 
        border-radius: 15px; margin-bottom: 10px; border-bottom-left-radius: 2px;
        border: 1px solid #333; width: fit-content; max-width: 85%;
    }
    .source-card {
        background-color: #181818; border: 1px solid #444; padding: 10px; border-radius: 8px; margin-top: 5px;
    }
    h1, h2, h3, .stExpander { color: #BB86FC !important; }
</style>
""", unsafe_allow_html=True)

# --- Logic: Load Advanced Pipeline ---
@st.cache_resource(show_spinner="Initializing ...")
def load_rag_pipeline():
    """
    Initializes the Hybrid Store and RAG Pipeline.
    Uses st.cache_resource to prevent reloading the model on every interaction.
    """
    store = HybridStore()
    try:
        # Attempt to load existing local indices
        store.load_store()
        logger.info("Successfully loaded FAISS and BM25 indices.")
    except Exception as e:
        st.error(f"Failed to load indices: {e}. Please ensure indices are created.")
        return None

    # Initialize Pipeline with Reranker and gpt-4.1-nano
    pipeline = RAGPipeline(hybrid_store=store)
    return pipeline

rag_engine = load_rag_pipeline()

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faq_query" not in st.session_state:
    st.session_state.faq_query = None

# --- Helper to use Base64 in HTML ---
def get_icon_html(b64_string, size=30):
    return f'<img src="data:image/png;base64,{b64_string}" width="{size}">'

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Settings") 
    st.markdown("Adjust precision/ recall values")
    
    # UI controls for retrieval depth
    top_k_retrieve = st.slider("Recall Depth (Hybrid Search)", 5, 50, 20)
    top_k_rerank = st.slider("Precision Depth (Rerank)", 1, 10, 5)
    
    show_context = st.checkbox("Show retrieved context", value=False)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("💡 Common Inquiries")
    faqs = [
        {"label": "What is NDB?", 
         "query": "Explain the Notifiable Data Breach (NDB) scheme."},
        {"label": "Privacy Principles", 
         "query": "What are the 13 Australian Privacy Principles (APPs)?"},
        {"label": "Application Range", 
         "query": "Does the Privacy Act 1988 apply to small businesses?"}
    ]
    for faq in faqs:
        if st.button(faq["label"], use_container_width=True):
            st.session_state.faq_query = faq["query"]
            st.rerun()

# --- Main Chat UI ---
st.title("My AI Buddy✨")
st.caption("AI-Powered Advisory for the Australian Privacy Act 1988")

# Render historical messages
for msg in st.session_state.messages:
    role_class = "user-msg" if msg["role"] == "user" else "ai-msg"
    st.markdown(f"<div class='{role_class}'>{msg['text']}</div>", unsafe_allow_html=True)
    
    # Render Sources if available and enabled
    if show_context and msg.get("docs") and msg["role"] == "assistant":
        with st.expander("📚 Verified Legal Sources"):
            for d in msg["docs"]:
                # Safe access to metadata
                score = d.metadata.get('relevance_score', 0)
                unit_id = d.metadata.get('unit_id', 'General Provision')
                
                st.markdown(f"""
                <div class='source-card'>
                    <strong>Section:</strong> {unit_id} | <strong>Relevance:</strong> {score:.2f}<br>
                    <small>{d.page_content[:250]}...</small>
                </div>
                """, unsafe_allow_html=True)

# --- Input Handling ---
user_input = st.chat_input("Ask about privacy law")
# Prioritize FAQ button clicks over text input
current_query = st.session_state.faq_query or user_input

if current_query:
    # 1. Add User message to UI
    st.session_state.messages.append({"role": "user", "text": current_query})
    st.markdown(f"<div class='user-msg'>{current_query}</div>", unsafe_allow_html=True)
    
    # Reset FAQ state for next turn
    st.session_state.faq_query = None

    # 2. Generate Response via LCEL Chain
    with st.spinner("Thinking..."):
        try:
            # Build and Invoke the chain
            # Pass the query as a dict to satisfy RunnableParallel structure
            chain = rag_engine.build_chain()
            output = chain.invoke({"question": current_query})
            
            # The chain returns a dict: {"answer": str, "docs": List[Document]}
            ai_answer = output["answer"]
            retrieved_docs = output["docs"]

            # Display AI Response
            st.markdown(f"<div class='ai-msg'>{ai_answer}</div>", unsafe_allow_html=True)
            
            # 3. Persist to session state
            st.session_state.messages.append({
                "role": "assistant",
                "text": ai_answer,
                "docs": retrieved_docs
            })
            
            # Force rerun to update scroll position and show context
            st.rerun()

        except Exception as e:
            logger.exception("Error in Streamlit execution loop")
            st.error(f"Pipeline Error: {e}")