from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

VECTOR_DB_DIR = "vector_store/chroma"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)

query = "What is the minimum attendance required to appear for exams?"

docs = vectordb.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content[:500])
