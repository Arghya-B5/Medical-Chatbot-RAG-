🩺 Medical Chatbot — RAG Based AI Application

A Retrieval-Augmented Generation (RAG) based Medical Chatbot built with Python, Flask, LangChain, Pinecone, Hugging Face Embeddings, Groq, MLflow, and DagsHub.

The chatbot retrieves relevant information from a medical knowledge base stored in Pinecone and uses the Groq LLM to generate an answer based on the retrieved context.

🚀 Features
🤖 AI-powered medical question answering
🔎 Retrieval-Augmented Generation (RAG)
🧠 Hugging Face sentence embeddings
🗄️ Pinecone vector database
⚡ Groq LLM inference
🔗 LangChain RAG pipeline
🌐 Flask web application
📄 Displays retrieved document sources
📊 MLflow experiment tracking
☁️ DagsHub MLflow integration
🔐 Environment-variable based API key management

🏗️ Architecture

                    ┌─────────────────┐
                    │      User       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Flask Web App   │
                    │     app.py      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  User Question  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Retriever    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Pinecone     │
                    │ Vector Database │
                    └────────┬────────┘
                             │
                         Top 3 Docs
                             │
                             ▼
                    ┌─────────────────┐
                    │    LangChain    │
                    │    RAG Chain    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Groq LLM     │
                    │ gpt-oss-20b     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Answer      │
                    │   + Sources     │
                    └─────────────────┘

                             │
                             ▼
                    ┌─────────────────┐
                    │     MLflow      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     DagsHub     │
                    └─────────────────┘


🔄 RAG Pipeline
User Question
      ↓
Hugging Face Embedding
      ↓
Query Vector
      ↓
Pinecone Similarity Search
      ↓
Top 3 Relevant Documents
      ↓
Context + Question
      ↓
Groq LLM
      ↓
Generated Answer
      ↓
Source Documents
      ↓
User

