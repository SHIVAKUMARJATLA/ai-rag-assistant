# ai-rag-assistant


## Dataset Source

Academic regulation documents were collected from the official
BVRIT academic regulations webpage, which links to Google Drive–
hosted PDF documents. These PDFs were downloaded, converted to text,
and cleaned to build a domain-specific knowledge base for the RAG system.


## Data Handling

Raw academic regulation PDFs are sourced from the official
college website and Google Drive links. Due to file size and
licensing, raw PDFs are not committed to the repository.
Preprocessing scripts are provided to reproduce the dataset.


## Chunking & Embeddings

- Implemented semantic chunking using a recursive text splitter to preserve contextual boundaries in academic regulation documents
- Generated dense vector embeddings using a transformer-based sentence embedding model
- Created and persisted a vector database using ChromaDB for efficient semantic retrieval
- Added a retrieval validation script to test similarity search over the embedded knowledge base

### Validation

A lightweight retrieval test script is included to verify that the vector
database returns relevant document chunks for domain-specific queries
before integrating the full RAG pipeline.

## Retrieval-Augmented Generation (RAG)

- Built an end-to-end Retrieval-Augmented Generation (RAG) pipeline for answering domain-specific questions over academic regulation documents
- Integrated semantic retrieval from a ChromaDB vector store with a strict prompt to reduce hallucinations
- Implemented an interactive question–answering interface
- Used a local LLM (Llama 3 via Ollama) to avoid API quota limitations while keeping the architecture LLM-agnostic

### Architecture Overview

The system follows a modular RAG architecture:
- Document preprocessing and chunking
- Vector embedding and storage using ChromaDB
- Semantic retrieval of relevant context
- Grounded answer generation using a local LLM

The LLM layer can be easily swapped with cloud-based models (e.g., Gemini or OpenAI) without changing the retrieval pipeline.

## Agentic RAG with LangGraph

- Implemented an agentic RAG workflow using LangGraph
- Modeled retrieval, answer generation, and verification as explicit agent states
- Reduced hallucinations through structured context verification
- Designed the system to be easily extensible for future tools and memory