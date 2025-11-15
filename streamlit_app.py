import streamlit as st

from src.rag.rag_pipeline import RAGPipeline
from src.rag.faiss_store import FaissStore
from src.rag.config import settings
from src.rag.logger import get_logger
from src.rag.exception import DocumentLoadError

logger = get_logger(__name__)

# ---------------------------------------------------------
# Load RAG Pipeline (cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    store = FaissStore()
    try:
        store.load_store()
        logger.info("FAISS store loaded successfully.")
    except DocumentLoadError:
        logger.warning("FAISS index not found. Initializing empty FAISS store.")
    except Exception as e:
        logger.error(f"Unexpected FAISS load error: {e}")
        store.vector_store = None

    pipeline = RAGPipeline(
        vector_store=store,
        temperature=settings.llm_temperature,
        streaming=True,
    )

    return pipeline

rag = load_pipeline()

# ---------------------------------------------------------
# UI Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Buddy",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Light theme styling
st.markdown("""
<style>
body, .block-container {
    background-color: #ffffff;
    color: #000000;
}
.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
}
.ai-msg {
    background-color: #F1F1F1;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.subheader("My AI Buddy")
st.markdown("###### Let AI speak on Australian privacy law")
st.caption("RAG • HuggingFace + LangChain + FAISS • Mobile Friendly")

# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_context" not in st.session_state:
    st.session_state.show_context = False

# ---------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top-K documents", 1, 10, settings.top_k)
    st.session_state.show_context = st.checkbox("Show Retrieved Context", value=False)
    if st.button("Clear Chat"):
        st.session_state.messages = []
    st.markdown("---")
    st.caption("Streamlit Cloud • CPU-only • HF Inference API")

# ---------------------------------------------------------
# Display Chat History (top)
# ---------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-msg'>{msg['text']}</div>", unsafe_allow_html=True)
        if st.session_state.show_context and msg.get("docs"):
            with st.expander("Retrieved Context"):
                for d in msg["docs"]:
                    st.markdown(d["content"])
                    st.caption(d["metadata"])

# ---------------------------------------------------------
# Chat Input (bottom)
# ---------------------------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "text": user_input,
        "docs": []
    })
    st.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Streaming RAG LLM Output
    # ---------------------------------------------------------
    with st.spinner("Thinking..."):
        try:
            stream = rag.generate_streaming(
                query=user_input,
                top_k=top_k,
                include_docs=True
            )

            ai_box = st.empty()
            final_text = ""
            final_docs = []

            for chunk in stream:
                if "token" in chunk:
                    final_text += chunk["token"]
                    ai_box.markdown(
                        f"<div class='ai-msg'>{final_text}</div>",
                        unsafe_allow_html=True
                    )

                if "docs" in chunk:
                    final_docs = [
                        {"content": d.page_content, "metadata": d.metadata or {}}
                        for d in chunk["docs"]
                    ]

            st.session_state.messages.append({
                "role": "assistant",
                "text": final_text,
                "docs": final_docs
            })

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------------------------------------
# Footer (always under input)
# ---------------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:12px; color:gray;'>
        &copy; 2025 Abdul Bari. All rights reserved.
    </p>
    """,
    unsafe_allow_html=True
)
