# src/tfidf_bm25_index.py
import pickle
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_FILE = "data/docs.pkl"
OUTPUT_PKL = "models/tfidf_bm25_index.pkl"
VECTOR_PICKLE = "models/vectorizer.pkl"
VOCAB_PICKLE = "models/vocab.pkl"
OUTPUT_JSON = "models_json/tfidf_bm25_index.json"

#bm25 parameters recommended  
k1 = 1.5
b = 0.75

def build_tfidf(docs):
    # docs: dict {doc_id: [tokens]}
    
    corpus = [" ".join(tokens) for tokens in docs.values()]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return vectorizer, tfidf_matrix

def build_bm25(tfidf_matrix, vectorizer, docs):
    #bm25 matrix built
    # following all formulas/methoid in slides

    N = len(docs)
    avg_doc_len = np.mean([len(tokens) for tokens in docs.values()])
    
    tf = tfidf_matrix.toarray() * len(docs)
    doc_lengths = np.array([len(tokens) for tokens in docs.values()])

    idf = vectorizer.idf_

    bm25_matrix = np.zeros_like(tf)
    
    for i in range(tf.shape[0]):  
        for j in range(tf.shape[1]): 
            tf_ij = tf[i, j]
            idf_j = idf[j]
            len_d = doc_lengths[i]
            bm25_matrix[i, j] = idf_j * ((tf_ij * (k1 + 1)) / (tf_ij + k1 * (1 - b + b * (len_d / avg_doc_len))))
    
    return bm25_matrix

def save_models(tfidf_matrix, bm25_matrix, vectorizer, output_pkl=OUTPUT_PKL, vector_pickle=VECTOR_PICKLE, vocab_pickle=VOCAB_PICKLE, json_file=OUTPUT_JSON):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    with open(output_pkl, "wb") as f:
        pickle.dump({ "tfidf": tfidf_matrix,"bm25": bm25_matrix}, f)
    
    with open(vector_pickle, "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(vocab_pickle, "wb") as f:
        pickle.dump(vectorizer.vocabulary_, f)
    
    # json for my-proofreadintg only t
    tfidf_json = {}
    for term, idx in vectorizer.vocabulary_.items():
        tfidf_json[term] = {
            "idf": float(vectorizer.idf_[idx]),
            "bm25": float(np.mean(bm25_matrix[:, idx]))
        }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(tfidf_json, f, ensure_ascii=False, indent=2)
    
    print(f"tf-idf +bm25 is saved as pickle: {output_pkl}")
    print(f"Vectorizer is  saved as pickle: {vector_pickle}")
    print(f"Vocabulary is saved as pickle: {vocab_pickle}")
    print(f"TF-IDF + BM25 JSON saved: {json_file}")

if __name__ == "__main__":
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
    
    vectorizer, tfidf_matrix = build_tfidf(docs)
    bm25_matrix = build_bm25(tfidf_matrix, vectorizer, docs)
    save_models(tfidf_matrix, bm25_matrix, vectorizer)



