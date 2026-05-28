# Updated `app.py` with Landing Page + Chat Interface


import streamlit as st
import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from model import GPT, Config


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="10M Nandha - Custom GPT-Style LLM",
    page_icon="",
    layout="wide"
)


# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        margin-top: 30px;
    }

    .sub-title {
        text-align: center;
        font-size: 22px;
        color: gray;
        margin-bottom: 40px;
    }

    .info-box {
        background-color: #111111;
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        border: 1px solid #333333;
    }

    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

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

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    return model


# -----------------------------
# LOAD TOKENIZER
# -----------------------------
@st.cache_resource

def load_tokenizer():

    tokenizer_path = hf_hub_download(
        repo_id="NandhaKrishnan/10M_NOVA",
        filename="tokenizer.json"
    )

    return Tokenizer.from_file(tokenizer_path)


# -----------------------------
# FRONT PAGE
# -----------------------------
if not st.session_state.chat_started:

    st.markdown(
        '<div class="main-title">10M Nandha MODEL (Next-gen Adaptive Neural Dense Hyper-optimized Architecture)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Custom GPT-Style Language Model Built From Scratch</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # TRAINING INFO
    st.markdown(
        """
        <div class="info-box">
        <h2> Model Information</h2>

        <ul>
            <li><b>Training Time:</b> 2 Hours</li>
            <li><b>GPU:</b> NVIDIA T4 GPU</li>
            <li><b>Architecture:</b> GPT-style Transformer</li>
            <li><b>Parameters:</b> ~10 Million</li>
            <li><b>Framework:</b> PyTorch</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CONFIG INFO
    st.markdown(
        """
        <div class="info-box">
        <h2> Model Configuration</h2>

        <ul>
            <li><b>Vocabulary Size:</b> 16,384</li>
            <li><b>Context Length:</b> 256</li>
            <li><b>Embedding Dimension:</b> 384</li>
            <li><b>Transformer Layers:</b> 4</li>
            <li><b>Attention Heads:</b> 6</li>
            <li><b>Dropout:</b> 0.1</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button(" Chat Now", use_container_width=True):
            st.session_state.chat_started = True
            st.rerun()


# -----------------------------
# CHAT INTERFACE
# -----------------------------
else:

    st.title(" Chat with 10M Nandha MODEL")

    model = load_model()
    tokenizer = load_tokenizer()

    # Show old messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Type your message...")

    if prompt:

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # TOKENIZE
        ids = tokenizer.encode(prompt).ids

        x = torch.tensor([ids], dtype=torch.long)

        # GENERATE
        with torch.no_grad():

            output = model.generate(
                x,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )

        # DECODE
        generated_ids = output[0][len(ids):].tolist()

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        # SHOW RESPONSE
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("⬅ Back to Home", use_container_width=True):
            st.session_state.chat_started = False
            st.rerun()
