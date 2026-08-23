from fastapi import FastAPI
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        pattern=r".*\S.*",
    )

app = FastAPI(title="RAVIN Framework PoC")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ravin",
    }

@app.post("/questions/validate")
def validate_question(request: QuestionRequest):
    return {
        "valid": True,
        "question": request.question,
    }