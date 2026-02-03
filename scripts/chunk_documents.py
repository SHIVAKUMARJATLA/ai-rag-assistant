import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

CLEAN_DIR = "data/cleaned"
OUTPUT_DIR = "data/chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

for file in os.listdir(CLEAN_DIR):
    if file.endswith(".txt"):
        path = os.path.join(CLEAN_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = text_splitter.split_text(text)

        out_file = file.replace(".txt", "_chunks.txt")
        with open(os.path.join(OUTPUT_DIR, out_file), "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n---\n")
        
        print(f"Created {len(chunks)} chunks from {file}")
