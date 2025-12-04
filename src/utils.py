# src/utils.py
# utility helpers: normalization, tokenization, file I/O, and mapping to course topics.

import re
import json
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[\u2018\u2019\u201c\u201d—–]', ' ', text)  # unicode quotes/dashes
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str, do_lemmatize=True):
    toks = word_tokenize(text)
    toks = [t for t in toks if t not in STOP and len(t) > 1]
    if do_lemmatize:
        toks = [LEMMATIZER.lemmatize(t) for t in toks]
    return toks

def save_jsonl(items, out_path):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf8') as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + '\n')

def load_jsonl(path):
    lst = []
    with open(path,'r',encoding='utf8') as fh:
        for line in fh:
            lst.append(json.loads(line))
    return lst

def precision_recall_f1(retrieved_docs, relevant_docs):
    retrieved_docs = set(retrieved_docs)
    relevant_docs = set(relevant_docs)
    
    tp = len(retrieved_docs & relevant_docs)
    fp = len(retrieved_docs - relevant_docs)
    fn = len(relevant_docs - retrieved_docs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1
