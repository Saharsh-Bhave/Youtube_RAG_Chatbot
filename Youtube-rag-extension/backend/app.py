from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import get_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "chrome-extension://bjgjbpklachbpjbpmjojdkepjmjfaelp",
    "http://localhost:3000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

class Query(BaseModel):
    question: str
    video_id: str

@app.post("/ask")
def ask_youtube(data: Query):
    answer = get_answer(data.video_id, data.question)
    return{"answer": answer}