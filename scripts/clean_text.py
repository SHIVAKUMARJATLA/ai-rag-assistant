import os
import re

CLEAN_DIR = "data/cleaned"

def clean_text(text):
    text = text.replace('\r', '\n')
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

for file in os.listdir(CLEAN_DIR):
    if file.endswith(".txt"):
        path = os.path.join(CLEAN_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = clean_text(text)

        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)

print("Text cleaning completed")