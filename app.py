import streamlit as st
import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from model import GPT, Config

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="10M Nandha SLM",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 58px;
    font-weight: bold;
    margin-top: 30px;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}

.card {
    background-color: #111;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #333;
    margin-top: 15px;
}

.center {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="NandhaKrishnan/10M_NOVA",
        filename="model.pt"
    )

    config = Config()
    model = GPT(config)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    return model


@st.cache_resource
def load_tokenizer():
    tokenizer_path = hf_hub_download(
        repo_id="NandhaKrishnan/10M_NOVA",
        filename="tokenizer.json"
    )
    return Tokenizer.from_file(tokenizer_path)


# =========================================================
# 1️ HOME PAGE
# =========================================================
if st.session_state.page == "home":

    st.markdown('<div class="main-title">10M Nandha SLM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Experimental Small Language Model Built from Scratch</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3> Model Information</h3>
    <ul>
        <li><b>Training Time:</b> 2 Hours</li>
        <li><b>GPU:</b> NVIDIA T4 GPU</li>
        <li><b>Architecture:</b> GPT-style Transformer</li>
        <li><b>Parameters:</b> ~10 Million</li>
        <li><b>Framework:</b> PyTorch</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.warning(
        " Experimental model. "
        "Accuracy may be limited due to only ~2 hours of training on NVIDIA T4 GPU."
    )
    st.markdown("""
    <div class="card">
    <h3> Model Configuration</h3>
    <ul>
        <li><b>Vocabulary Size:</b> 16,384</li>
        <li><b>Context Length:</b> 256</li>
        <li><b>Embedding Dimension:</b> 384</li>
        <li><b>Transformer Layers:</b> 4</li>
        <li><b>Attention Heads:</b> 6</li>
        <li><b>Dropout:</b> 0.1</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3> Model Variants</h3>
    <ul>
        <li><b>Base Model:</b> 10M parameters trained from scratch</li>
        <li><b>Fine-Tuned Model:</b> Alpaca 52K instruction tuning</li>
        <li><b>Internet-Connected Small Language Model (Experimental):</b> Web search and real-time knowledge integration (in progress)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(" =>   Continue to Model Selection", use_container_width=True):
        st.session_state.page = "select"
        st.rerun()

    st.markdown("---")

    st.warning(
        " Experimental model. "
        "Accuracy may be limited due to only ~2 hours of training on NVIDIA T4 GPU."
    )


# =========================================================
# 2️⃣ MODEL SELECTION PAGE
# =========================================================
elif st.session_state.page == "select":

    st.title(" Select Model")

    col1, col2, col3 = st.columns(3)

    # ---------------- BASE MODEL ----------------
    with col1:
        st.subheader("Base Model")
        st.write("Trained from scratch (10M parameters)")

        if st.button("Use Base Model", use_container_width=True):
            st.session_state.page = "base_chat"
            st.rerun()

    # ---------------- FINE TUNED ----------------
    with col2:
        st.subheader("Fine-Tuned Model")
        st.write("Alpaca 52K Instruction Tuned")

        if st.button("Use Fine-Tuned Model", use_container_width=True):
            st.markdown(" Redirecting to Fine-Tuned Model...")

            st.markdown(
                """
                <meta http-equiv="refresh" content="0; url=finetune-10mslm.streamlit.app" />
                """,
                unsafe_allow_html=True
            )
            st.stop()

    # ---------------- INTERNET MODEL ----------------
    with col3:
        st.subheader("Internet-Connected Small Language Model (Experimental)")
        st.write("Web search and real-time knowledge integration (in progress)")

        if st.button("Use Internet-Connected Small Language Model (Experimental)", use_container_width=True):
            st.session_state.page = "internet"
            st.rerun()

    st.markdown("---")

    if st.button("⬅ Back to Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()


# =========================================================
# 3️⃣ BASE MODEL CHAT
# =========================================================
elif st.session_state.page == "base_chat":

    st.title(" Base Model (10 Million Parameters) Chat")

    model = load_model()
    tokenizer = load_tokenizer()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask something...")

    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            output = model.generate(
                x,
                max_new_tokens=60,
                temperature=0.8,
                top_k=40
            )

        gen_ids = output[0][len(ids):].tolist()

        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")

    if st.button("⬅ Back"):
        st.session_state.page = "select"
        st.rerun()


# =========================================================
# 4️⃣ INTERNET MODEL PAGE
# =========================================================
elif st.session_state.page == "internet":

    st.title(" Internet-Connected SLM (Experimental)")

    st.warning("DuckDuckGo integration is under development.")

    st.info("""
We are currently working on enabling real-time web search capability
for this model using DuckDuckGo API.

I am currently facing deployment errors in this implementation, 
so I will fix the issues and integrate this feature as soon as possible.

This feature will allow the model to access real-time information from the internet once completed.
""")

    if st.button("⬅ Back"):
        st.session_state.page = "select"
        st.rerun()
