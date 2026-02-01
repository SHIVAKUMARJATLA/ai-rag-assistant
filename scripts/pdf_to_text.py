import os
from PyPDF2 import PdfReader

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

for file in os.listdir(RAW_DIR):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(RAW_DIR, file)
        text = extract_text_from_pdf(pdf_path)

        output_file = file.replace(".pdf", ".txt")
        with open(os.path.join(CLEAN_DIR, output_file), "w", encoding="utf-8") as f:
            f.write(text)

print("PDF text extraction completed")