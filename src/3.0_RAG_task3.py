import faiss
import pickle
import torch
import numpy as np
from torch import no_grad
from huggingface_hub import login
from google.colab import drive
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


index = faiss.read_index(".../vector_store/complaint_index.faiss")

# Load chunks and metadata
with open(".../vector_store/chunk_texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open(".../vector_store/chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")


def ask(query):
    query_vec = model.encode([query])

    top_k = 5
    D, I = index.search(np.array(query_vec), top_k)

    retrieved_chunks = [texts[i] for i in I[0]]
    retrieved_metadata = [metadata[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are an AI assistant that summarizes customer complaints.

    User question:
    {query}

    Based on these complaints:
    {context}

    Answer the question with specific insights from the text.
    """
    login(new_session=False)
    pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
    messages = [
        {"role": "user", "content": prompt},
    ]

    with no_grad():
        return(pipe(messages['content']))