import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

VECTOR_DB_DIR = "vector_store/chroma"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

PROMPT_TEMPLATE = """
You are an academic regulation assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"I could not find this information in the provided regulations."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-flash-latest",
#     temperature=0
# )

llm = ChatOllama(
    model="llama3",
    temperature=0
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question(query: str):
    docs = retriever.invoke(query)

    formatted_prompt = prompt.format(
        context=format_docs(docs),
        question=query
    )

    response = llm.invoke(formatted_prompt)
    return response.content

if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (or type 'exit): ")
        if user_query.lower() == "exit":
            break
        answer = ask_question(user_query)
        print("\nAnswer:\n", answer)