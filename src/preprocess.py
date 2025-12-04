# src/preprocess.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import json
import os

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Folder to save JSON outputs
JSON_OUTPUT_FOLDER = "models_json"
os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)

def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, stem."""
    text = str(text).lower()  # ensure string
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS]
    return tokens

def preprocess_articles(input_file="data/articles.csv",
                        pickle_output="data/docs.pkl",
                        json_output=os.path.join(JSON_OUTPUT_FOLDER, "docs.json")):
    """Preprocess articles and save in both pickle and JSON formats."""
    try:
        # Use latin1 encoding and skip bad lines
        df = pd.read_csv(input_file, encoding='latin1', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    docs = {}
    for idx, row in df.iterrows():
        doc_id = idx
        docs[doc_id] = preprocess_text(row['Article'])
    
    # Save as pickle
    with open(pickle_output, "wb") as f:
        pickle.dump(docs, f)
    print(f"Preprocessed {len(docs)} articles saved to {pickle_output} (pickle format)")

    # Save as JSON (convert sets/lists as needed)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    print(f"Preprocessed {len(docs)} articles saved to {json_output} (JSON format)")

if __name__ == "__main__":
    preprocess_articles()
