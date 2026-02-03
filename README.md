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