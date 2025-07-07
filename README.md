## 📌 Overview

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot that helps financial analysts explore insights from thousands of **real consumer complaints** submitted to the Consumer Financial Protection Bureau (CFPB).

The chatbot answers internal business questions like:
> “Why are customers unhappy with BNPL?”  
> “What is the most frequent issue with personal loans?”

It retrieves semantically relevant complaint narratives and uses a Large Language Model (LLM) to generate context-aware summaries — enabling smarter decision-making grounded in real customer voice.

---

## 🎯 Objectives

- Explore and clean large-scale customer complaint data.
- Generate high-quality embeddings of complaint narratives.
- Index them into a vector database for fast retrieval.
- Build a QA system that generates natural language answers based on real complaints.

---

## 📂 Project Structure

├── data/  
│ ├── consumer_complaints.csv  
│ └── filtered_complaints.csv  
├── notebooks/  
│ ├── 1.0-eda-cleaning.ipynb  
│ ├── 2.0-chunk-embed-index.ipynb  
│ └── 3.0-retriever-qa.ipynb   
├── vector_store/  
│ ├── complaint_index.faiss  
│ ├── chunk_texts.pkl  
│ └── chunk_metadata.pkl  
├──  src  
├── app/  
│ └── streamlit_app.py   
├── requirements.txt  
└── README.md  