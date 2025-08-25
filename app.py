import torch
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Batch Translator", page_icon="🌐")

MODEL = "Helsinki-NLP/opus-mt-en-id"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    mdl.to(device)
    return tok, mdl, device

tok, mdl, device = load_model()

st.title("EN → ID Batch Translator 🌐")

uploaded_file = st.file_uploader("Upload CSV (harus ada kolom 'text' atau 'teks')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # pastikan kolom teks ada
    if "text" in df.columns:
        col_name = "text"
    elif "teks" in df.columns:
        col_name = "teks"
    else:
        st.error("CSV harus punya kolom 'text' atau 'teks'")
        st.stop()

    st.write("Contoh data:")
    st.dataframe(df.head())

    if st.button("Translate All"):
        results = []
        with st.spinner("Translating..."):
            for t in df[col_name].astype(str).tolist():
                inputs = tok(t, return_tensors="pt", truncation=True).to(device)
                outs = mdl.generate(**inputs, max_new_tokens=128)
                translation = tok.decode(outs[0], skip_special_tokens=True)
                results.append(translation)

        df["translation"] = results
        st.success("Selesai 🚀")
        st.dataframe(df.head())

        # convert to CSV untuk di-download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download hasil CSV",
            data=csv,
            file_name="translations.csv",
            mime="text/csv"
        )

