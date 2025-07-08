import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import gc

# --- Load Vector Store ---
@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index("vector_store/complaint_index.faiss")
    with open("vector_store/chunk_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    with open("vector_store/chunk_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, texts, metadata

index, texts, metadata = load_index_and_chunks()

# --- Load Embedding Model ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# --- Load LLM Pipeline (Flan-T5 or Mistral) ---
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-base"  # or mistralai/Mistral-7B-Instruct-v0.1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        # device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

llm_pipeline = load_llm()

# --- Streamlit UI ---
st.title("💬 RAG Chatbot: Consumer Complaint Insights")
st.markdown("Ask a question like _'Why are people unhappy with Buy Now Pay Later?'_")

query = st.text_input("🔍 Enter your question")

if st.button("Generate Answer") and query:
    with st.spinner("Embedding query and retrieving..."):
        # Step 1: Embed and Search
        query_vec = embedder.encode([query])
        D, I = index.search(np.array(query_vec), k=5)
        retrieved_chunks = [texts[i] for i in I[0]]

        # Step 2: Compose context
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""Based on the following customer complaints:\n\n{context}\n\nAnswer this: {query}"""

        # Step 3: Generate Answer
        with torch.no_grad():
            output = llm_pipeline(prompt, max_new_tokens=300, do_sample=False)[0]['generated_text']

        # Step 4: Display
        st.subheader("🧠 Generated Answer:")
        st.write(output)

        with st.expander("🔎 Retrieved Complaint Chunks"):
            for chunk in retrieved_chunks:
                st.markdown(f"> {chunk}")

        # Cleanup (important for GPU memory)
        gc.collect()
        torch.cuda.empty_cache()