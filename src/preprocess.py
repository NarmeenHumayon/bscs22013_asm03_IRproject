# src/preprocess.py
# first step in IR pipeline is preprocessing 
# normalisation, lowercase, punctuation removal, tokenisation, stopword remival, stemming, result format

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import json
import os

# dirst dowanload nltk by running requiremnts.txt then stopwords from nltk
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# models_json/docs.json fr human readable output
JSON_OUTPUT_FOLDER = "models_json"
os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)

def preprocess_text(text):
    # lowercase, punctuation removal, tokenizeation, stopwords removal & stemming
    text = str(text).lower() 
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS]
    return tokens

def preprocess_articles(input_file="data/articles.csv",
                        pickle_output="data/docs.pkl",
                        json_output=os.path.join(JSON_OUTPUT_FOLDER, "docs.json")):

    try:
        # GPT suggested correction: use latin1 encoding and skip bad lines
        df = pd.read_csv(input_file, encoding='latin1', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    docs = {}
    for idx, row in df.iterrows():
        doc_id = idx
        docs[doc_id] = preprocess_text(row['Article'])
    
    with open(pickle_output, "wb") as f:
        pickle.dump(docs, f)
    print(f"Preprocessed {len(docs)} articles saved to {pickle_output} (pickle format)")

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    print(f"Preprocessed {len(docs)} articles saved to {json_output} (JSON format)")

if __name__ == "__main__":
    preprocess_articles()
