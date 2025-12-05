# path:
# src/boolean_index.py

# preprocessed docs.pkl input here. Ourtputs: models/boolean_index.pkl and models_json/boolean_index.json (not used in  project but i made for human readable output)

import pickle
import json
import os

DOCS_FILE = "data/docs.pkl"
PICKLE_OUTPUT = "models/boolean_index.pkl"
JSON_OUTPUT_FOLDER = "models_json"
JSON_OUTPUT = os.path.join(JSON_OUTPUT_FOLDER, "boolean_index.json")

os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)

with open(DOCS_FILE, "rb") as f:
    docs = pickle.load(f)

boolean_index = {}
for doc_id, tokens in docs.items():
    for token in tokens:
        if token not in boolean_index:
            boolean_index[token] = set()
        boolean_index[token].add(doc_id)

# .json convert using for loop 
boolean_index_json = {term: list(doc_ids) for term, doc_ids in boolean_index.items()}

# .pkl outpur
with open(PICKLE_OUTPUT, "wb") as f:
    pickle.dump(boolean_index, f)
print(f"Boolean_index is saved to {PICKLE_OUTPUT} (pickle format)")

with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(boolean_index_json, f, indent=2)
print(f"Boolean_index is saved to {JSON_OUTPUT} (JSON format)")


