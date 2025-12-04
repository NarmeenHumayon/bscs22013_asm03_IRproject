# src/positional_index.py
import pickle
import json
import os

# Paths
DOCS_FILE = "data/docs.pkl"
OUTPUT_PKL = "models/positional_index.pkl"
OUTPUT_JSON = "models_json/positional_index.json"


def build_positional_index(docs):
    """
    Build a positional inverted index:
    term -> { doc_id: [positions] }
    """
    index = {}
    for doc_id, tokens in docs.items():
        for pos, term in enumerate(tokens):
            if term not in index:
                index[term] = {}
            if doc_id not in index[term]:
                index[term][doc_id] = []
            index[term][doc_id].append(pos)
    return index

def save_index(index, pkl_file=OUTPUT_PKL, json_file=OUTPUT_JSON):
    # Ensure JSON folder exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    # Save as pickle
    with open(pkl_file, "wb") as f:
        pickle.dump(index, f)
    
    # Save as JSON (for human readability)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Positional index saved as pickle: {pkl_file}")
    print(f"Positional index saved as JSON: {json_file}")

if __name__ == "__main__":
    # Load preprocessed docs
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)

    # Build index
    positional_index = build_positional_index(docs)

    # Save index
    save_index(positional_index)
