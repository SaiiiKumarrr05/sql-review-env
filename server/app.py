from fastapi import FastAPI

app = FastAPI()

# ✅ ADD THIS
@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/reset")
def reset():
    return {"status": "reset successful"}

