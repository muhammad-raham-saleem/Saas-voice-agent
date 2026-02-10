import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from src.rag_engine import RAGEngine

st.set_page_config(
    page_title="TechFlow AI Support",
    page_icon="🛠️",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .main-header h1 {
        color: #1E88E5;
        margin-bottom: 0;
    }
    .main-header p {
        color: #666;
        font-size: 1.1rem;
    }
    .source-chip {
        display: inline-block;
        background: #E3F2FD;
        color: #1565C0;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load RAG engine (cached) ───────────────────────────────
@st.cache_resource(show_spinner="Loading TechFlow knowledge base...")
def load_rag_engine():
    return RAGEngine()


engine = load_rag_engine()


# ── Voice helpers ──────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using Deepgram REST API."""
    import httpx

    response = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-3", "smart_format": "true", "language": "en"},
        headers={
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "audio/wav",
        },
        content=audio_bytes,
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


def generate_speech(text: str) -> bytes | None:
    """Generate speech audio bytes using ElevenLabs. Returns MP3 bytes."""
    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        audio = client.text_to_speech.convert(
            text=text,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )
        # audio is an iterator — collect into bytes
        return b"".join(audio)
    except Exception:
        return None


def has_voice_keys() -> bool:
    """Check if voice API keys are configured."""
    dg = os.getenv("DEEPGRAM_API_KEY", "")
    el = os.getenv("ELEVENLABS_API_KEY", "")
    return bool(dg and dg != "your-key-here" and el and el != "your-key-here")


voice_enabled = has_voice_keys()


# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛠️ TechFlow AI Support</h1>
    <p>Ask anything about features, billing, troubleshooting, API, or policies</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 System Metrics")
    stats = engine.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Documents", stats["doc_count"])
    col2.metric("Chunks", stats["chunk_count"])
    col1.metric("Dimensions", stats["vector_dimensions"])
    col2.metric("Vectors", stats["total_vectors"])

    st.divider()

    # Voice toggle
    if voice_enabled:
        st.header("🎙️ Voice Mode")
        voice_on = st.toggle("Enable voice", value=True)
    else:
        voice_on = False

    st.divider()
    st.header("💡 Suggested Questions")
    suggestions = [
        "How do I reset my password?",
        "What's included in the Pro plan?",
        "What are the API rate limits?",
        "My tasks aren't syncing",
        "What's your refund policy?",
        "How do Kanban boards work?",
        "What integrations are available?",
        "How is my data protected?",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, key=f"sug_{suggestion}", use_container_width=True):
            st.session_state["pending_question"] = suggestion

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        engine.memory.clear()
        st.rerun()

# ── Chat state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.caption(src["content"][:300] + "..." if len(src["content"]) > 300 else src["content"])
                    st.divider()
        if msg.get("response_time"):
            st.caption(f"⏱️ {msg['response_time']}s")
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

# ── Voice input ─────────────────────────────────────────────
question = None

if voice_on:
    st.markdown("**🎙️ Click the mic to record your question:**")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#1E88E5",
        icon_size="2x",
        pause_threshold=2.0,
    )
    if audio_bytes and audio_bytes != st.session_state.get("_last_audio"):
        st.session_state["_last_audio"] = audio_bytes
        with st.spinner("Transcribing..."):
            try:
                question = transcribe_audio(audio_bytes)
                if question:
                    st.success(f'🗣️ "{question}"')
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# ── Text input + pending questions ──────────────────────────
pending = st.session_state.pop("pending_question", None)
if pending:
    question = pending

text_input = st.chat_input("Ask a question about TechFlow...")
if text_input:
    question = text_input

# ── Process question ────────────────────────────────────────
if question:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = engine.query(question)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📎 Sources"):
                for src in result["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.caption(src["content"][:300] + "..." if len(src["content"]) > 300 else src["content"])
                    st.divider()

        st.caption(f"⏱️ {result['response_time']}s")

        # Generate and play voice response
        audio_data = None
        if voice_on:
            with st.spinner("Generating voice response..."):
                audio_data = generate_speech(result["answer"])
            if audio_data:
                st.audio(audio_data, format="audio/mp3", autoplay=True)

    # Save assistant message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "response_time": result["response_time"],
        "audio": audio_data,
    })
