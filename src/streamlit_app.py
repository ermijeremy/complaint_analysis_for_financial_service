import streamlit as st
import pickle
import faiss
import os
import getpass
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from langchain_mistralai import ChatMistralAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# ------------------------
# CONFIG
# ------------------------
HF_TOKEN = st.secrets["MISTRAL_API_KEY"]  # Set this in .streamlit/secrets.toml
HF_MODEL = "bigscience/bloom-560m"   # Can be changed to another HF-hosted model


# ------------------------
# Load Vector Index + Chunks
# ------------------------
@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index("vector_store/complaint_index.faiss")
    with open("vector_store/chunk_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    with open("vector_store/chunk_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, texts, metadata

index, texts, metadata = load_index_and_chunks()

# ------------------------
# Embedding Model (local, lightweight)
# ------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# ------------------------
# Hugging Face Inference Client
# ------------------------
@st.cache_resource
def load_hf_client():
    return ChatMistralAI(
    api_key=HF_TOKEN,  
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

hf_client = load_hf_client()
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def query_hf_model(prompt):
    return hf_client.invoke(prompt)

# ------------------------
# Streamlit App Interface
# ------------------------
st.title("💬 RAG Chatbot: Complaint Insight Assistant")
st.markdown("Ask any question based on customer complaints (e.g. *Why are customers unhappy with Buy Now Pay Later?*)")

query = st.text_input("🔍 Enter your question")

if st.button("Generate Answer") and query:
    with st.spinner("Retrieving complaints and generating answer..."):
        # Step 1: Embed and Search
        query_vec = embedder.encode([query])
        D, I = index.search(np.array(query_vec), k=5)
        retrieved_chunks = [texts[i] for i in I[0]]

        # Step 2: Prepare context and prompt
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""You are a helpful assistant. Based on these complaint narratives:\n\n{context}\n\nAnswer the question: {query}"""

        # Step 3: Use HF API to generate answer
        answer = query_hf_model(prompt)

        # Step 4: Show result
        st.subheader("🧠 Generated Answer")
        st.write(answer.content)

        with st.expander("🔎 Retrieved Complaint Chunks"):
            for chunk in retrieved_chunks:
                st.markdown(f"> {chunk}")
