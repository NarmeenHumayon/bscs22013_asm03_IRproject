# bscs22013_asm03_ir — CS516 assignment 03 solution

## Overview of project:
This repository contains a local IR system combining:
- Boolean retrieval (inverted boolean index)
- Positional index (phrase/proximity)
- BM25 candidate retrieval (rank-bm25)
- TF–IDF vector-space reranking
- Pseudo-Relevance Feedback (Rocchio-style expansion)

It uses course topics taught: boolean retrieval, positional indices, TF–IDF, BM25, query expansion, and evaluation. search.py file when run, prompts user to ask which search method to use: boolean, positional /k near search and tf-idf ranking top 10 results.

## Settup for Windows (my laptop is windows)
in terminal:

cd bscs22013_asm03_ir
python -m venv venv  // i prefer using virtual environment
venv\Scripts\activate
pip install -r requirements.txt // automatically downloads required dependicies to run my project 


## Step by step files to run:
We follow IR pipeline to compile
first preprocess by:
python src/preprocess.py
then:
python src/boolean_index.py
python src/positional_index.py
python src/tfidf_bm25_index.py


// these create both .pkl in models folder and .json in models_json folder. .pkl will be fast for model to use when running search query but .json return huma readable file so i can cross-check outputs :)

then to search run:
python src/search.py

this is run a while loop. It will keep prompting you to chose seaching option unless to enter -1 to exit. 
1. Boolean Search (AND, OR, NOT) handled
2. Positional / Phrase / Proximity Search (karachi /3 sindh) all docs with sindh within 3 words distance from karachi
3. Ranked Retrieval (bm25/tf-idf ranking)

these 3 are implmented 


## Data file used
data of Articles.csv downloaded from provided link in assignment document:
https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles/data  
You can find csv in data folder data/articles.csv
