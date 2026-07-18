import csv
import re
import math
import os
import sys
import io
from collections import defaultdict

# Fix encoding untuk Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stdin.encoding != 'utf-8':
    try:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

FALLBACK_STOP_WORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
    'untuk', 'pada', 'adalah', 'dalam', 'tidak', 'akan', 'sudah',
    'juga', 'saya', 'aku', 'kamu', 'dia', 'kami', 'kita', 'mereka',
    'ada', 'bisa', 'atau', 'ya', 'nya', 'se', 'tapi', 'karena',
    'kalau', 'lagi', 'mau', 'apa', 'sama', 'kan', 'aja', 'sih',
    'dong', 'deh', 'loh', 'kok', 'jadi', 'udah', 'gak', 'ga',
    'gk', 'yg', 'tp', 'tdk', 'sm', 'krn', 'krna', 'klo', 'dg',
    'the', 'a', 'an', 'of', 'to', 'in', 'is', 'it', 'for'
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "tokenisasi dan stopwatch removal.csv")

class DataLoader:
    def __init__(self):
        self.documents = {}
        self.doc_sources = {}
        self.num_docs = 0

    def load_documents_from_csv(self, filepath):
        print(f"Memuat dokumen dari: {os.path.basename(filepath)}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                reader = csv.reader(f)
                rows = list(reader)

        header_row = None
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if cell.strip().lower() == 'komentar':
                    header_row = i
                    break
            if header_row is not None:
                break

        if header_row is None:
            print("[ERROR] Kolom 'Komentar' tidak ditemukan di CSV!")
            return 0

        header = rows[header_row]
        komentar_idx = None
        no_idx = None
        sumber_idx = None

        for j, cell in enumerate(header):
            cell_lower = cell.strip().lower()
            if cell_lower == 'komentar':
                komentar_idx = j
            elif cell_lower == 'no':
                no_idx = j
            elif cell_lower == 'sumber':
                sumber_idx = j

        if komentar_idx is None:
            return 0

        doc_count = 0
        current_doc_id = None
        current_text = None
        current_source = None

        for i in range(header_row + 1, len(rows)):
            row = rows[i]
            if len(row) <= komentar_idx:
                if current_doc_id is not None and len(row) > 0:
                    extra_text = ' '.join([cell for cell in row if cell.strip()])
                    if extra_text:
                        current_text += ' ' + extra_text
                continue

            no_val = row[no_idx].strip() if no_idx is not None and no_idx < len(row) else ''
            komentar_val = row[komentar_idx].strip() if komentar_idx < len(row) else ''

            if no_val and komentar_val:
                if current_doc_id is not None and current_text:
                    self.documents[current_doc_id] = current_text
                    self.doc_sources[current_doc_id] = current_source
                    doc_count += 1

                try:
                    current_doc_id = int(no_val)
                except ValueError:
                    current_doc_id = doc_count + 1
                current_text = komentar_val
                current_source = row[sumber_idx].strip() if sumber_idx is not None and sumber_idx < len(row) else '-'
            elif current_doc_id is not None and komentar_val:
                current_text += ' ' + komentar_val

        if current_doc_id is not None and current_text:
            self.documents[current_doc_id] = current_text
            self.doc_sources[current_doc_id] = current_source
            doc_count += 1

        self.num_docs = len(self.documents)
        print(f"[OK] Berhasil memuat {self.num_docs} dokumen")
        return self.num_docs

class TFIDFEngine:
    def __init__(self, data_loader):
        if SASTRAWI_AVAILABLE:
            stemmer_factory = StemmerFactory()
            self.stemmer = stemmer_factory.create_stemmer()
            stopword_factory = StopWordRemoverFactory()
            self.stop_words = set(stopword_factory.get_stop_words())
        else:
            self.stemmer = None
            self.stop_words = FALLBACK_STOP_WORDS.copy()

        self.data = data_loader
        self.processed_docs = {}
        self.inverted_index = {}
        self.tfidf_weights = {}
        self.doc_lengths = {}
        self.vocabulary = set()

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def build_index(self):
        print("Membangun Inverted Index untuk TF-IDF...")
        for doc_id, text in self.data.documents.items():
            tokens = self.preprocess(text)
            self.processed_docs[doc_id] = tokens
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = freq
                self.vocabulary.add(term)

    def compute_tfidf(self):
        print("Menghitung bobot TF-IDF...")
        N = self.data.num_docs
        for term in self.vocabulary:
            self.tfidf_weights[term] = {}
            df = len(self.inverted_index[term])
            idf = math.log10(N / df) if df > 0 else 0
            for doc_id, tf in self.inverted_index[term].items():
                tf_weight = 1 + math.log10(tf) if tf > 0 else 0
                tfidf = tf_weight * idf
                self.tfidf_weights[term][doc_id] = tfidf

        for doc_id in self.data.documents:
            length_sq = 0
            for term in self.vocabulary:
                if term in self.tfidf_weights and doc_id in self.tfidf_weights[term]:
                    weight = self.tfidf_weights[term][doc_id]
                    length_sq += weight ** 2
            self.doc_lengths[doc_id] = math.sqrt(length_sq) if length_sq > 0 else 0

    def search(self, query):
        query_tokens = self.preprocess(query)
        if not query_tokens:
            return {}

        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1

        query_weights = {}
        query_length_sq = 0
        N = self.data.num_docs

        for term, tf in query_tf.items():
            if term in self.inverted_index:
                df = len(self.inverted_index[term])
                idf = math.log10(N / df) if df > 0 else 0
                tf_weight = 1 + math.log10(tf) if tf > 0 else 0
                tfidf = tf_weight * idf
                query_weights[term] = tfidf
                query_length_sq += tfidf ** 2

        query_length = math.sqrt(query_length_sq) if query_length_sq > 0 else 0
        if query_length == 0:
            return {}

        scores = {}
        for term, q_weight in query_weights.items():
            if term in self.tfidf_weights:
                for doc_id, d_weight in self.tfidf_weights[term].items():
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += q_weight * d_weight

        cosine_scores = {}
        for doc_id, dot_product in scores.items():
            doc_length = self.doc_lengths[doc_id]
            if doc_length > 0 and query_length > 0:
                cosine_sim = dot_product / (query_length * doc_length)
            else:
                cosine_sim = 0
            cosine_scores[doc_id] = cosine_sim

        return cosine_scores

class SemanticSearchEngine:
    def __init__(self, data_loader, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers tidak tersedia. Jalankan 'pip install sentence-transformers torch'")
        self.data = data_loader
        print(f"Memuat model Sentence-BERT: {model_name}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.doc_embeddings = {}

    def encode_documents(self):
        print("Meng-encode dokumen dengan Sentence-BERT...")
        doc_ids = list(self.data.documents.keys())
        texts = [self.data.documents[did] for did in doc_ids]
        
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        
        for did, emb in zip(doc_ids, embeddings):
            self.doc_embeddings[did] = emb

    def search(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = {}
        for did, doc_emb in self.doc_embeddings.items():
            sim = util.cos_sim(query_embedding, doc_emb).item()
            scores[did] = sim
        return scores

class HybridRetrievalEngine:
    def __init__(self, tfidf_engine, semantic_engine, alpha=0.6):
        self.tfidf = tfidf_engine
        self.semantic = semantic_engine
        self.alpha = alpha

    def _min_max_norm(self, scores_dict):
        if not scores_dict:
            return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 0.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def search(self, query):
        tfidf_scores = self.tfidf.search(query)
        semantic_scores = self.semantic.search(query)

        norm_tfidf = self._min_max_norm(tfidf_scores)
        norm_semantic = self._min_max_norm(semantic_scores)

        hybrid_scores = {}
        all_docs = set(self.tfidf.data.documents.keys())
        for did in all_docs:
            t_score = norm_tfidf.get(did, 0.0)
            s_score = norm_semantic.get(did, 0.0)
            h_score = (self.alpha * s_score) + ((1 - self.alpha) * t_score)
            if h_score > 0:
                hybrid_scores[did] = h_score

        return hybrid_scores



def calc_metrics(retrieved_docs, ground_truth, k=10):
    retrieved = set(r['doc_id'] for r in retrieved_docs[:k])
    relevant_retrieved = retrieved & ground_truth
    precision = len(relevant_retrieved) / len(retrieved) if len(retrieved) > 0 else 0
    recall = len(relevant_retrieved) / len(ground_truth) if len(ground_truth) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f_measure': f_measure
    }

def get_top_k(scores_dict, data_loader, k=10):
    results = []
    for did, score in scores_dict.items():
        results.append({
            'doc_id': did,
            'score': score,
            'text': data_loader.documents[did]
        })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:k]

if __name__ == "__main__":
    print("Inisialisasi Sistem Pencarian...")
    
    data = DataLoader()
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] File CSV tidak ditemukan di: {CSV_FILE}")
        sys.exit(1)
    
    data.load_documents_from_csv(CSV_FILE)
    
    print("\n[1] Menyiapkan TF-IDF Engine...")
    tfidf_engine = TFIDFEngine(data)
    tfidf_engine.build_index()
    tfidf_engine.compute_tfidf()
    
    print("\n[2] Menyiapkan Semantic Search Engine (SBERT)...")
    semantic_engine = SemanticSearchEngine(data)
    semantic_engine.encode_documents()
    
    alpha = 0.6
    print(f"\n[3] Menyiapkan Hybrid Retrieval Engine (alpha={alpha})...")
    hybrid_engine = HybridRetrievalEngine(tfidf_engine, semantic_engine, alpha=alpha)
    
    # Ground truth
    query1 = "demam setelah imunisasi"
    ground_truth_q1 = {8, 10, 13, 15, 22, 27, 31, 34, 40, 42, 45, 46}
    
    query2 = "vaksin anak balita"
    ground_truth_q2 = {1, 2, 4, 5, 7, 28, 29, 32, 38, 43, 44, 47}
    
    print("\nMelakukan Pencarian dan Evaluasi...")
    
    # Eval Q1
    q1_tfidf = get_top_k(tfidf_engine.search(query1), data)
    q1_sbert = get_top_k(semantic_engine.search(query1), data)
    q1_hybrid = get_top_k(hybrid_engine.search(query1), data)
    
    metrics_q1 = {
        "TF-IDF": calc_metrics(q1_tfidf, ground_truth_q1),
        "SBERT": calc_metrics(q1_sbert, ground_truth_q1),
        "Hybrid": calc_metrics(q1_hybrid, ground_truth_q1)
    }
    
    examples_q1 = {
        "TF-IDF": q1_tfidf[:3],
        "SBERT": q1_sbert[:3],
        "Hybrid": q1_hybrid[:3]
    }
    
    # Eval Q2
    q2_tfidf = get_top_k(tfidf_engine.search(query2), data)
    q2_sbert = get_top_k(semantic_engine.search(query2), data)
    q2_hybrid = get_top_k(hybrid_engine.search(query2), data)
    
    metrics_q2 = {
        "TF-IDF": calc_metrics(q2_tfidf, ground_truth_q2),
        "SBERT": calc_metrics(q2_sbert, ground_truth_q2),
        "Hybrid": calc_metrics(q2_hybrid, ground_truth_q2)
    }
    
    print("\n[OK] Evaluasi Selesai.")

    print("\n" + "="*60)
    print("  PENCARIAN INTERAKTIF (SEMANTIC & HYBRID)")
    print("="*60)
    print("Ketik pertanyaan atau kata kunci untuk melihat hasil pencarian.")
    print("Ketik 'quit' atau 'exit' untuk keluar.")
    print("="*60)
    
    while True:
        try:
            print()
            user_query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari pencarian interaktif.")
            break
            
        if not user_query:
            continue
            
        if user_query.lower() in ('quit', 'exit', 'keluar', 'q'):
            print("\nTerima kasih!")
            break
            
        print("\nMencari dokumen...")
        
        # Search all 3 methods
        t_res = get_top_k(tfidf_engine.search(user_query), data, k=3)
        s_res = get_top_k(semantic_engine.search(user_query), data, k=3)
        h_res = get_top_k(hybrid_engine.search(user_query), data, k=3)
        
        print("\n--- Top 3: TF-IDF (Leksikal) ---")
        for i, r in enumerate(t_res):
            print(f"{i+1}. [Doc {r['doc_id']}] Skor: {r['score']:.4f}")
            print(f"   Komentar: {r['text'][:100]}...")
            
        print("\n--- Top 3: SBERT (Semantik) ---")
        for i, r in enumerate(s_res):
            print(f"{i+1}. [Doc {r['doc_id']}] Skor: {r['score']:.4f}")
            print(f"   Komentar: {r['text'][:100]}...")
            
        print("\n--- Top 3: HYBRID (Gabungan) ---")
        for i, r in enumerate(h_res):
            print(f"{i+1}. [Doc {r['doc_id']}] Skor: {r['score']:.4f}")
            print(f"   Komentar: {r['text'][:100]}...")
