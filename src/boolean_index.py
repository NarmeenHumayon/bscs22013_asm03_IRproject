# src/boolean_index.py
"""
Boolean Index Construction
- Builds a Boolean inverted index from preprocessed documents
- Saves the index in both pickle (fast loading) and JSON (human-readable) formats
"""

import pickle
import json
import os

# Paths
DOCS_FILE = "data/docs.pkl"
PICKLE_OUTPUT = "models/boolean_index.pkl"
JSON_OUTPUT_FOLDER = "models_json"
JSON_OUTPUT = os.path.join(JSON_OUTPUT_FOLDER, "boolean_index.json")

# Create JSON folder if it doesn't exist
os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)

# Load preprocessed documents
with open(DOCS_FILE, "rb") as f:
    docs = pickle.load(f)

# Build Boolean inverted index
# Structure: {term1: [doc_id1, doc_id2, ...], term2: [...]}
boolean_index = {}
for doc_id, tokens in docs.items():
    for token in tokens:
        if token not in boolean_index:
            boolean_index[token] = set()
        boolean_index[token].add(doc_id)

# Convert sets to lists for JSON serialization
boolean_index_json = {term: list(doc_ids) for term, doc_ids in boolean_index.items()}

# Save as pickle
with open(PICKLE_OUTPUT, "wb") as f:
    pickle.dump(boolean_index, f)
print(f"Boolean index saved to {PICKLE_OUTPUT} (pickle format)")

# Save as JSON
with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(boolean_index_json, f, indent=2)
print(f"Boolean index saved to {JSON_OUTPUT} (JSON format)")

