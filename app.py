import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
import os
from PyPDF2 import PdfReader

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:latest"
EMBED_MODEL = "nomic-embed-text:latest"

COLLECTION_NAME = "custom_knowledge"
CHROMA_PATH = ".praison" 

client = chromadb.PersistentClient(path=CHROMA_PATH)

def ollama_embed(texts):
    embeddings = []
    for text in texts:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text}
        )
        embeddings.append(response.json()["embedding"])
    return embeddings

def load_pdf_chunks(pdf_path, chunk_size=300):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

def init_knowledgebase():
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
    )
    if len(collection.get()["ids"]) == 0:  # Avoid re-uploading
        chunks = load_pdf_chunks("nature.pdf")
        embeds = ollama_embed(chunks)
        ids = [f"doc_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeds, ids=ids)
    return collection

# -- RAG: Retrieve + Generate --
def retrieve_relevant_chunks(query, collection, top_k=4):
    query_embed = ollama_embed([query])[0]
    results = collection.query(query_embeddings=[query_embed], n_results=top_k)
    return results["documents"][0]

def call_ollama(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt}
    )
    return response.json()["response"]

def ask_agent(user_query, collection):
    chunks = retrieve_relevant_chunks(user_query, collection)
    context = "\n\n".join(chunks)
    full_prompt = f"""You are an AI assistant. Answer based on the following context:

{context}

Question: {user_query}
Answer:"""
    return call_ollama(full_prompt)

st.title("Knowledge Agent (RAG + Ollama)")

if "collection" not in st.session_state:
    with st.spinner("Initializing knowledgebase..."):
        st.session_state.collection = init_knowledgebase()
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_agent(prompt, st.session_state.collection)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
