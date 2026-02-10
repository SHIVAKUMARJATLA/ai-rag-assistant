from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import func

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from scripts.database import SessionLocal, QueryLog, init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Aacademic Regulations AI Assistant",
    version="1.0"
)



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

ANSWER_PROMPT = PromptTemplate(
    template="""
You are an Academic Regulations AI Assistant.

Answer strictly using the provided context.
If the information is missing, reply exactly:
"I could not find this information in the provided regulations."

Context:
{context}

Question:
{question}

Answer (clear and concise):
""",
    input_variables=["context", "question"]
)

def retrieve_docs(state: AgentState):
    docs = retriever.invoke(state["question"])
    contents = [doc.page_content for doc in docs]
    return {"documents": contents}

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

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question_api(request: QuestionRequest):
    result = agent.invoke({"question": request.question})
    answer = result["answer"]

    db = SessionLocal()
    log = QueryLog(
        question=request.question,
        answer=answer
    )
    db.add(log)
    db.commit()
    db.close()
    return {"answer": answer}

@app.get("/stats")
def get_status():
    db = SessionLocal()
    total_queries = db.query(func.count(QueryLog.id)).scalar()
    latest_queries = (
        db.query(QueryLog.question, QueryLog.created_at)
        .order_by(QueryLog.created_at.desc())
        .limit(5)
        .all()
    )

    db.close()

    return {
        "total_queries": total_queries,
        "recent_queries": [
            {"question": q, "timestamp": str(t)} for q, t in latest_queries
        ],
    }