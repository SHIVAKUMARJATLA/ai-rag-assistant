import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHUNK_DIR = "data/chunks"
VECTOR_DB_DIR = "vector_store/chroma"

os.makedirs(VECTOR_DB_DIR, exist_ok=True)

documents = []

for file in os.listdir(CHUNK_DIR):
    if file.endswith(".txt"):
        path = os.path.join(CHUNK_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("\n---\n")
        
        for chunk in raw_chunks:
            if chunk.strip():
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file}
                    )
                )

print(f"Total chunks: {len(documents)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_DIR
)

vector_db.persist()
print("Vector database created and persisted")