from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import get_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = "chrome-extension://bjgjbpklachbpjbpmjojdkepjmjfaelp",
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