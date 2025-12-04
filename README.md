# bscs22013_asm03_ir — CS516 HW3 solution

## Overview
This repository contains a local IR system combining:
- Boolean retrieval (inverted boolean index)
- Positional index (phrase/proximity)
- BM25 candidate retrieval (rank-bm25)
- TF–IDF vector-space reranking
- Pseudo-Relevance Feedback (Rocchio-style expansion)
- Optional NER-based filtering (if spaCy installed)

It maps directly to CS516 topics: boolean retrieval, positional indices, TF–IDF, BM25, query expansion, and evaluation.

## Quick setup (Windows)
```cmd
cd bscs22013_asm03_ir
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
