import streamlit as st
import os
import sys

# Import core classes from semantic_search.py
from semantic_search import (
    DataLoader, 
    TFIDFEngine, 
    SemanticSearchEngine, 
    HybridRetrievalEngine,
    get_top_k,
    CSV_FILE
)

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Cache the data loading and engine initialization so it doesn't reload on every interaction
@st.cache_resource
def load_engines():
    if not os.path.exists(CSV_FILE):
        st.error(f"File dataset tidak ditemukan: {CSV_FILE}")
        st.stop()
        
    data = DataLoader()
    data.load_documents_from_csv(CSV_FILE)
    
    tfidf = TFIDFEngine(data)
    tfidf.build_index()
    tfidf.compute_tfidf()
    
    semantic = SemanticSearchEngine(data)
    semantic.encode_documents()
    
    # We will set alpha dynamically later
    hybrid = HybridRetrievalEngine(tfidf, semantic, alpha=0.6)
    
    return data, tfidf, semantic, hybrid

# Sidebar configuration
st.sidebar.title("Pengaturan")
st.sidebar.markdown("Atur parameter pencarian di sini.")
alpha = st.sidebar.slider(
    "Bobot Alpha (Hybrid)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.6, 
    step=0.1,
    help="1.0 = 100% SBERT (Semantik), 0.0 = 100% TF-IDF (Leksikal)"
)

# Header
st.title("🔍 Semantic & Hybrid Search Engine")
st.markdown("Mesin pencari ini menggunakan dataset **Komentar Imunisasi Balita**.")

# Initialize engines
with st.spinner("Memuat model dan dataset... (Ini mungkin memakan waktu beberapa detik pada proses pertama)"):
    data, tfidf_engine, semantic_engine, hybrid_engine = load_engines()
    
# Update alpha on hybrid engine if changed
hybrid_engine.alpha = alpha

st.markdown("---")
# Search UI
query = st.text_input("Ketikkan kata kunci pencarian Anda (misal: 'demam setelah imunisasi'):")

if query:
    with st.spinner("Mencari..."):
        # Perform searches
        t_res = get_top_k(tfidf_engine.search(query), data, k=10)
        s_res = get_top_k(semantic_engine.search(query), data, k=10)
        h_res = get_top_k(hybrid_engine.search(query), data, k=10)
        
        st.subheader("Hasil Pencarian (Top 10)")
        
        # Display results in 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 TF-IDF (Leksikal)")
            if not t_res:
                st.info("Tidak ada hasil.")
            for i, r in enumerate(t_res):
                st.markdown(f"**{i+1}. Doc {r['doc_id']}** (Skor: `{r['score']:.4f}`)")
                st.caption(f"{r['text']}")
                st.divider()
                
        with col2:
            st.markdown("### 🧠 SBERT (Semantik)")
            if not s_res:
                st.info("Tidak ada hasil.")
            for i, r in enumerate(s_res):
                st.markdown(f"**{i+1}. Doc {r['doc_id']}** (Skor: `{r['score']:.4f}`)")
                st.caption(f"{r['text']}")
                st.divider()
                
        with col3:
            st.markdown("### 🔗 Hybrid (Gabungan)")
            if not h_res:
                st.info("Tidak ada hasil.")
            for i, r in enumerate(h_res):
                st.markdown(f"**{i+1}. Doc {r['doc_id']}** (Skor: `{r['score']:.4f}`)")
                st.caption(f"{r['text']}")
                st.divider()
