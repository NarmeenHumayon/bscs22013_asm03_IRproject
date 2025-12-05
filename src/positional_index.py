# src/positional_index.py

import pickle
import json
import os

DOCS_FILE = "data/docs.pkl"
OUTPUT_PKL = "models/positional_index.pkl"
OUTPUT_JSON = "models_json/positional_index.json"


def build_positional_index(docs):
    # concept: term -> { doc_id: [positions] }
    #this is how appears in json, pkl format we cant read

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
    # json directory: models_json/positional_index.py
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    # pkl
    with open(pkl_file, "wb") as f:
        pickle.dump(index, f)
    
    #json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Positional_index is saved as pickle: {pkl_file}")
    print(f"Positional_index  is saved as JSON: {json_file}")

if __name__ == "__main__":
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)

    positional_index = build_positional_index(docs)

    save_index(positional_index)

