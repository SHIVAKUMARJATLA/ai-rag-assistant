from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str


VECTOR_DB_DIR = "vector_store/chroma"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(
    model="llama3",
    temperature=0
)

def retrieve_docs(state: AgentState):
    docs = retriever.invoke(state["question"])
    contents = [doc.page_content for doc in docs]
    return {"documents": contents}

ANSWER_PROMPT = PromptTemplate(
    template="""
You are an academic regulation assistant.

Answer the question using ONLY the context below.
If the answer is not present, say:
"I could not find this information in the provided regulations."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

def generate_answer(state: AgentState):
    context = "\n\n".join(state["documents"])
    prompt = ANSWER_PROMPT.format(
        context=context,
        question=state["question"]
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def verify_answer(state: AgentState):
    if not state["documents"]:
        return {"answer": "I could not find this information in the provided regulations."}
    return state

graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("answer", generate_answer)
graph.add_node("verify", verify_answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "verify")
graph.add_edge("verify", END)

agent = graph.compile()


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        result = agent.invoke({"question": question})
        print("\nAnswer:\n", result["answer"])
