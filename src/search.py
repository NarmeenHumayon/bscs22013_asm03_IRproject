# src/search.py
import pickle
import json
import os
import numpy as np
import re
from preprocess import preprocess_text

# --- Paths ---
BOOLEAN_INDEX_PKL = "models/boolean_index.pkl"
POSITIONAL_INDEX_PKL = "models/positional_index.pkl"
TFIDF_BM25_PKL = "models/tfidf_bm25_index.pkl"
VECTOR_PICKLE = "models/vectorizer.pkl"
DOCS_FILE = "data/docs.pkl"
OUTPUT_FILE = "search_results.txt"

# --- Load functions ---
def load_boolean_index():
    with open(BOOLEAN_INDEX_PKL, "rb") as f:
        return pickle.load(f)

def load_positional_index():
    with open(POSITIONAL_INDEX_PKL, "rb") as f:
        return pickle.load(f)

def load_tfidf_bm25():
    with open(TFIDF_BM25_PKL, "rb") as f:
        return pickle.load(f)

def load_vectorizer():
    with open(VECTOR_PICKLE, "rb") as f:
        return pickle.load(f)

def load_docs():
    with open(DOCS_FILE, "rb") as f:
        return pickle.load(f)

# --- BOOLEAN SEARCH (supports AND / OR) ---
def boolean_search(query, boolean_index):
    """
    Query examples:
        'karachi AND sindh'   -> intersection
        'karachi OR sindh'    -> union
        'karachi'             -> single term
    """
    # Convert to lowercase and split by spaces
    tokens = query.lower().split()
    if not tokens:
        return set()

    result_docs = set()
    current_op = "AND"  # default operator
    temp_docs = None

    for token in tokens:
        if token in ["and", "or"]:
            current_op = token.upper()
        else:
            docs = set(boolean_index.get(token, []))
            if temp_docs is None:
                temp_docs = docs
            else:
                if current_op == "AND":
                    temp_docs &= docs
                elif current_op == "OR":
                    temp_docs |= docs

    if temp_docs is not None:
        result_docs = temp_docs
    return result_docs


# --- POSITIONAL / PHRASE SEARCH ---
def positional_search(query, positional_index):
    terms = preprocess_text(query)
    if not terms:
        return set()
    first_term_docs = positional_index.get(terms[0], {})
    result_docs = set()
    for doc_id in first_term_docs:
        positions = first_term_docs[doc_id]
        for pos in positions:
            match = True
            for i, term in enumerate(terms[1:], 1):
                term_positions = positional_index.get(term, {}).get(doc_id, [])
                if pos + i not in term_positions:
                    match = False
                    break
            if match:
                result_docs.add(doc_id)
                break
    return result_docs

# --- PROXIMITY SEARCH / EXTENDED POS SEARCH ---
def positional_proximity_search(query, positional_index):
    """
    Supports:
      'term1 term2'      -> exact phrase
      'term1 /k term2'   -> term1 within k words of term2
    """
    query = query.lower()
    prox_match = re.match(r"(\w+)\s*/(\d+)\s*(\w+)", query)
    if prox_match:
        term1, k, term2 = prox_match.groups()
        k = int(k)
        result_docs = set()
        docs1 = positional_index.get(term1, {})
        docs2 = positional_index.get(term2, {})
        common_docs = set(docs1.keys()) & set(docs2.keys())
        for doc_id in common_docs:
            pos1_list = docs1[doc_id]
            pos2_list = docs2[doc_id]
            for p1 in pos1_list:
                for p2 in pos2_list:
                    if abs(p1 - p2) <= k:
                        result_docs.add(doc_id)
                        break
        return result_docs
    else:
        return positional_search(query, positional_index)

# --- RANKED RETRIEVAL (BM25 / TF-IDF) ---
def ranked_retrieval(query, tfidf_bm25_index, vectorizer, top_k=10):
    terms = preprocess_text(query)
    if not terms:
        return []

    tfidf_matrix = tfidf_bm25_index['tfidf']
    bm25_matrix = tfidf_bm25_index['bm25']
    vocab = vectorizer.vocabulary_

    scores = np.zeros(tfidf_matrix.shape[0])
    for term in terms:
        if term in vocab:
            idx = vocab[term]
            scores += bm25_matrix[:, idx]
    ranked_doc_ids = np.argsort(scores)[::-1]

    return [(int(doc_id), float(scores[doc_id])) for doc_id in ranked_doc_ids[:top_k] if scores[doc_id] > 0]

# --- Main Loop ---
def main():
    docs = load_docs()
    boolean_index = positional_index = tfidf_bm25_index = vectorizer = None

    while True:
        print("\n=== IR Search System ===")
        print("Choose search type (-1 to exit):")
        print("1. Boolean Search")
        print("2. Positional / Phrase / Proximity Search")
        print("3. Ranked Retrieval (BM25/TF-IDF)")

        choice = input("Enter choice: ").strip()
        if choice == "-1":
            print("Exiting search system. Goodbye!")
            break

        query = input("Enter your search query: ").strip()
        if not query:
            print("Empty query, please try again.")
            continue

        results_output = []

        if choice == "1":
            if boolean_index is None:
                boolean_index = load_boolean_index()
            results = boolean_search(query, boolean_index)
            results_output.append("[Boolean Search Results]")
            if results:
                for doc_id in sorted(results):
                    snippet = " ".join(docs[doc_id][:30])
                    results_output.append(f"Doc {doc_id}: {snippet}...")
            else:
                results_output.append("No matching documents found.")

        elif choice == "2":
            if positional_index is None:
                positional_index = load_positional_index()
            results = positional_proximity_search(query, positional_index)
            results_output.append("[Positional / Phrase / Proximity Search Results]")
            if results:
                for doc_id in sorted(results):
                    snippet = " ".join(docs[doc_id][:30])
                    results_output.append(f"Doc {doc_id}: {snippet}...")
            else:
                results_output.append("No matching documents found.")

        elif choice == "3":
            if tfidf_bm25_index is None:
                tfidf_bm25_index = load_tfidf_bm25()
            if vectorizer is None:
                vectorizer = load_vectorizer()
            results = ranked_retrieval(query, tfidf_bm25_index, vectorizer)
            results_output.append("[Ranked Retrieval Results (Top 10)]")
            if results:
                for doc_id, score in results:
                    snippet = " ".join(docs[doc_id][:30])
                    results_output.append(f"Doc {doc_id} (score: {score:.4f}): {snippet}...")
            else:
                results_output.append("No relevant documents found.")

        else:
            print("Invalid choice. Please enter 1, 2, 3 or -1 to exit.")
            continue

        # Print and save results
        print("\n".join(results_output))
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(results_output))
        print(f"\n[Results saved to {OUTPUT_FILE}]")

if __name__ == "__main__":
    main()
