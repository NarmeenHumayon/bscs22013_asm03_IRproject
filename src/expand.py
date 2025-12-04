"""
expand.py
Implement simple pseudo-relevance feedback (PRF) using Rocchio-like approach:
 - take top-k retrieved docs, compute centroid of TF-IDF vectors, pick top terms to add to query
Course mapping: query expansion, relevance feedback (Rocchio algorithm).
"""

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def expand_query(query, models_dir, top_k=5, add_terms=5):
    # models_dir should have vectorizer.pkl and tfidf_matrix.npz and doc_order.txt
    vectorizer = joblib.load(models_dir + '/vectorizer.pkl')
    X = __import__('scipy').sparse.load_npz(models_dir + '/tfidf_matrix.npz')
    with open(models_dir + '/doc_order.txt','r',encoding='utf8') as fh:
        doc_ids = [l.strip() for l in fh]

    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    # centroid
    centroid = X[top_idx].mean(axis=0)  # sparse mean
    centroid = np.asarray(centroid).ravel()
    # map feature->score
    feat_names = vectorizer.get_feature_names_out()
    top_terms_idx = np.argsort(centroid)[-add_terms:][::-1]
    top_terms = [feat_names[i] for i in top_terms_idx]
    return top_terms

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='models')
    parser.add_argument('--q', default='oil price pakistan')
    args = parser.parse_args()
    print(expand_query(args.q, args.models))
