from fastapi import FastAPI

app = FastAPI(title="RAVIN Framework PoC")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ravin",
    }