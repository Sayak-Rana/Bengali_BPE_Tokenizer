# streamlit_bengali_bpe.py
import streamlit as st
import json
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
# The app expects these files to be in the same folder as this script in the repo:
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
            # Accept either "a b" or "ab" style; we expect two columns
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
            else:
                # fallback — attempt to split single string into two tokens
                # but usually merges are written as "a b"
                tokens = ln.split()
                if len(tokens) == 2:
                    a, b = tokens
                else:
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
st.set_page_config(page_title="Bengali BPE Tokenizer", layout="centered")
st.title("Bengali BPE Tokenizer — Demo")
st.markdown(
    """
**Language:** Bengali (Bangla)  
This app uses a BPE tokenizer trained offline.  
Place the following files in the same repository folder before deploying:
- `bengali_bpe_demo.merges.txt`
- `bengali_bpe_demo.vocab.json`
"""
)

# Load model
MERGES, TOKEN_TO_ID = load_model()
if not MERGES or not TOKEN_TO_ID:
    st.warning("Model files not found. Make sure 'bengali_bpe_demo.merges.txt' and 'bengali_bpe_demo.vocab.json' are present in the repo folder.")
else:
    st.success(f"Loaded model: {len(MERGES)} merges, vocab size = {len(TOKEN_TO_ID)}")

# Controls
hide_end = st.checkbox("Hide </w> marker in display", value=True)
show_raw = st.checkbox("Show raw tokens (with </w>)", value=False)

text_in = st.text_area("Enter Bengali text here:", value="আমি মেশিন লার্নিং শিখছি", height=140)

if st.button("Tokenize"):
    if not MERGES:
        st.error("Model not loaded. Add model files to the repo.")
    else:
        tokens, ids = encode_text(text_in, MERGES, TOKEN_TO_ID)
        st.write("#### Tokens (display)")
        st.write(pretty_tokens(tokens, hide_end=hide_end))
        if show_raw:
            st.write("#### Raw tokens (debug)")
            st.write(tokens)
        st.write("#### Token IDs (per token)")
        st.write(ids)

# Small about / metadata
st.markdown("---")
st.markdown(
    "Model metadata:\n\n"
    "- Tokenizer language: **Bengali (Bangla)**\n"
    "- Training corpus: e.g. `mHossain/bengali_sentiment_v2` (used for demo)\n"
    "- Model files: `bengali_bpe_demo.merges.txt`, `bengali_bpe_demo.vocab.json`\n\n"
    "If you update the model files in the repo, redeploy on Streamlit Cloud to pick up changes."
)
