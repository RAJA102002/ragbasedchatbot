# RAG-based Chatbot with Streamlit

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and open-source models.

## Features
- Retrieval-Augmented Generation (RAG) for context-aware responses.
- DeepSeek R1 as the LLM for generating answers.
- ChromaDB as the vector store for efficient document retrieval.
- Nomic Embed Text for embedding knowledge into a searchable format.
- Streamlit UI for easy interaction with the AI chatbot.

## Project Structure
```
.
├── app.py
├── agents.py            # Main Streamlit App
├── nature.pdf           # Sample knowledge document
└── README.md            # Project Documentation
```

## Installation & Setup

## Clone the Repository
```bash
   git clone https://github.com/RAJA102002/ragbasedchatbot.git
```
##  Install Dependencies
Could you make sure you have Python 3.8+ installed?
- Install Python
- Install Ollama

## Start Ollama (Locally) and other Dependencies
Ensure you have Ollama installed and running:
```bash
#Install Ollama
```

Download DeepSeek R1:
```bash
ollama pull deepseek-llm:latest

```

Download Nomic Embed models:
```bash
ollama pull nomic-embed-text:latest

```

Install required packages:
```bash
pip install chromadb

```

Now, set your OpenAI API key, to the requirement for the SDK:
```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=fake-key
```

## Run the Application**
```bash
streamlit run agents.py
```

---

How It Works
1. Loads Knowledge – Uses `sample.pdf`, For Example `nature.pdf` for retrieval-based answering.
2. Embeds Data – Utilizes Nomic Embed Text for vectorized search.
3. Retrieves Relevant Info – Searches ChromaDB for the most relevant content.
4. Generates Responses – Feeds retrieved data into DeepSeek R1 for contextual answers.

---
Example Usage
1. Run the app and open the Streamlit UI.
2. Ask a question related to the uploaded document.
3. Get AI-generated responses based on retrieved knowledge!
