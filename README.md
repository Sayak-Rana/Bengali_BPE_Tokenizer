# Bengali BPE Tokenizer

This project implements a **Byte Pair Encoding (BPE) tokenizer** for the **Bengali (Bangla)** language.  

---

## Files in this repository

- **`Bengali_BPE_Assign1.ipynb`**  
  - Used to load the Bengali corpus.  
  - Implement the BPE algorithm from scratch.  
  - Train the tokenizer.  
  - Save the trained model files (`.merges.txt` and `.vocab.json`).  

- **`streamlit_bengali_bpe.py`**  
  - Streamlit application file.  
  - Loads the trained model files.  
  - Provides a text box interface to test tokenization of Bengali sentences.  

- **`requirements.txt`**  
  - Contains the dependency for running the Streamlit app.  

---

## Dataset

- The tokenizer was trained using the Bengali corpus from:  
  [mHossain/bengali_sentiment_v2](https://huggingface.co/datasets/mHossain/bengali_sentiment_v2)  

---

## Website Link

The Streamlit app is hosted here:  
👉 [https://beng-bpe-tokenizer-sayak.streamlit.app/](https://beng-bpe-tokenizer-sayak.streamlit.app/)

---

## How to use

- Open the website link.  
- Paste any Bengali sentence into the text box.  
- Click **Tokenize** to see the tokenized output.  

---

## Developed by

Sayak Rana 
