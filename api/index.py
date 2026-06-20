from fastapi import FastAPI, Form
from app.main import ask_question

app = FastAPI()

@app.get("/response")
def init():
    return "Hello!"

@app.post("/response")
def get_response(text: str = Form(...)):
    answer = ask_question(text)
    return {
        "answer": answer
    }

