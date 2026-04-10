from fastapi import FastAPI

app = FastAPI()

@app.get("/reset")
def reset_get():
    return {"status": "reset successful"}

@app.post("/reset")
def reset_post():
    return {"status": "reset successful"}
