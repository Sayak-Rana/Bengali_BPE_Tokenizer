# streamlit_bengali_bpe.py
import streamlit as st
import json
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
# Model filenames expected in the repository root (same folder as this script)
MERGES_FILE = "bengali_bpe_demo.merges.txt"
VOCAB_FILE  = "bengali_bpe_demo.vocab.json"

# ----------------------------
# Utilities: load model
# ----------------------------
def load_model(merges_path=MERGES_FILE, vocab_path=VOCAB_FILE):
    merges = []
    token_to_id = {}
    mp = Path(merges_path)
    vp = Path(vocab_path)
    if mp.exists():
        for ln in mp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            # Expect merges written as two tokens per line (e.g., "ক া")
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
            else:
                # fallback — skip malformed lines
                continue
            merges.append((a, b))
    if vp.exists():
        token_to_id = json.loads(vp.read_text(encoding="utf-8"))
    return merges, token_to_id

# ----------------------------
# Encoding logic (greedy using merges)
# ----------------------------
def encode_word_with_merges(word, merges):
    symbols = list(word) + ["</w>"]
    rank = {pair: idx for idx, pair in enumerate(merges)}
    while True:
        if len(symbols) < 2:
            break
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
        candidate = None
        best_r = None
        best_pos = None
        for pos, p in enumerate(pairs):
            if p in rank:
                r = rank[p]
                if best_r is None or r < best_r:
                    best_r = r
                    candidate = p
                    best_pos = pos
        if candidate is None:
            break
        a, b = candidate
        merged = a + b
        symbols = symbols[:best_pos] + [merged] + symbols[best_pos+2:]
    return symbols

def encode_text(text, merges, token_to_id):
    tokens = []
    for w in text.strip().split():
        if not w:
            continue
        tokens.extend(encode_word_with_merges(w, merges))
    ids = [token_to_id.get(t, -1) for t in tokens]  # -1 => token not in saved vocab
    return tokens, ids

def pretty_tokens(tokens, hide_end=True):
    if hide_end:
        return [t[:-4] if t.endswith("</w>") else t for t in tokens]
    return tokens

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Bengali Language Tokenizer", layout="centered")
st.title("Bengali Language Tokenizer")

# clear dataset attribution and language
st.markdown(
    """
**Tokenizer language:** **Bengali (Bangla)**  
**Training corpus (demo):** `mHossain/bengali_sentiment_v2` (Hugging Face dataset)
"""
)

# Load model
MERGES, TOKEN_TO_ID = load_model()
if not MERGES or not TOKEN_TO_ID:
    st.warning(
        "Model files not found in the app folder. Ensure 'bengali_bpe_demo.merges.txt' and 'bengali_bpe_demo.vocab.json' are present in the repository."
    )
else:
    st.success(f"Loaded model: {len(MERGES)} merges, vocab size = {len(TOKEN_TO_ID)}")

# Controls: mutually exclusive display mode using radio
display_mode = st.radio(
    "Token display mode:",
    ("Hide </w> marker (clean display)", "Show raw tokens (with </w>)"),
    index=0
)

text_in = st.text_area("Enter Bengali text here:", value="আমি মেশিন লার্নিং শিখছি", height=140)

if st.button("Tokenize"):
    if not MERGES:
        st.error("Model not loaded. Add model files to the repo.")
    else:
        tokens, ids = encode_text(text_in, MERGES, TOKEN_TO_ID)

        if display_mode == "Hide </w> marker (clean display)":
            st.write("#### Tokens (display)")
            st.write(pretty_tokens(tokens, hide_end=True))
        else:
            st.write("#### Raw tokens")
            st.write(tokens)

        st.write("#### Token IDs (per token)")
        st.write(ids)

# Small about / metadata
st.markdown("---")
st.markdown(
    "- **Tokenizer language:** Bengali (Bangla)\n"
    "- **Training corpus used for demo:** https://huggingface.co/datasets/mHossain/bengali_sentiment_v2"
    # "- If you update the model files in the repository, redeploy on Streamlit Cloud to pick up changes."
)
