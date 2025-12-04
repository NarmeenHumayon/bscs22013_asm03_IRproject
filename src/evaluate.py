# src/evaluate.py
import pickle
from search import (
    boolean_search,
    positional_proximity_search,
    ranked_retrieval,
    load_boolean_index,
    load_positional_index,
    load_tfidf_bm25,
    load_vectorizer,
    load_docs
)

RESULTS_FILE = "evaluation_results.txt"

def save_results(text):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[Results saved to {RESULTS_FILE}]\n")

def run_search(mode, query, boolean_idx, positional_idx, tfidf_bm25, vectorizer):
    """Run actual search functions and return a list of doc IDs (not scores)."""
    if mode == "1":
        return sorted(list(boolean_search(query, boolean_idx)))
    elif mode == "2":
        return sorted(list(positional_proximity_search(query, positional_idx)))
    elif mode == "3":
        ranked = ranked_retrieval(query, tfidf_bm25, vectorizer)
        return [doc_id for doc_id, _ in ranked]
    else:
        return []

def compute_metrics(retrieved, relevant):
    """
    retrieved: iterable of doc IDs returned by system
    relevant: iterable/set of doc IDs judged relevant
    returns: precision, recall, f1, TP, FP, FN
    """
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    tp = len(retrieved_set & relevant_set)
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp, fp, fn

def manual_evaluation():
    print("\n=== Manual Evaluation Mode ===")

    # Load indexes
    print("Loading indexes...")
    boolean_idx = load_boolean_index()
    positional_idx = load_positional_index()
    tfidf_bm25 = load_tfidf_bm25()
    vectorizer = load_vectorizer()
    docs = load_docs()
    print("Indexes loaded.\n")

    print("Choose search type:")
    print("1. Boolean Search")
    print("2. Phrase / Proximity Search")
    print("3. Ranked Retrieval (BM25/TF-IDF)")
    search_mode = input("Enter choice: ").strip()

    query = input("Enter query: ").strip()

    retrieved = run_search(search_mode, query, boolean_idx, positional_idx, tfidf_bm25, vectorizer)

    print("\n[System Retrieved Documents]")
    if not retrieved:
        print("No documents retrieved by search engine.")
    else:
        for doc in retrieved:
            print(f"Doc {doc}")

    rel_input = input("\nEnter relevant doc IDs (space-separated): ").strip()
    relevant_docs = list(map(int, rel_input.split())) if rel_input else []

    # Compute metrics properly
    precision, recall, f1, tp, fp, fn = compute_metrics(retrieved, relevant_docs)

    output = []
    output.append("=== Evaluation Report ===")
    output.append(f"Query: {query}")
    output.append(f"Search Mode: {search_mode}")
    output.append(f"Retrieved (count={len(retrieved)}): {retrieved}")
    output.append(f"Relevant (count={len(relevant_docs)}): {sorted(relevant_docs)}")
    output.append(f"TP = {tp}, FP = {fp}, FN = {fn}")
    output.append(f"Precision = {precision:.4f}")
    output.append(f"Recall    = {recall:.4f}")
    output.append(f"F1 Score  = {f1:.4f}\n")

    final_text = "\n".join(output)
    print("\n" + final_text)
    save_results(final_text)

if __name__ == "__main__":
    print("\n=== IR System Evaluation ===")
    print("1 - Manual Relevance Judgments (recommended)")
    mode = input("Enter choice: ").strip()
    if mode == "1":
        manual_evaluation()
    else:
        print("Invalid choice.")
