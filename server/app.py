"""
FastAPI server exposing OpenEnv-compatible HTTP endpoints for the SQL Review Environment.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import SQLReviewEnv, SQLAction

app = FastAPI(
    title="SQL Review & Optimization Environment",
    description="OpenEnv-compatible environment for training agents on SQL tasks.",
    version="1.0.0",
)

# Single shared env instance (stateful per session)
_env = SQLReviewEnv()


# ── Request Models ──

class ResetRequest(BaseModel):
    task_index: int = 0  # 0=easy, 1=medium, 2=hard


class StepRequest(BaseModel):
    query: str


# ── Endpoints ──

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment and return initial observation."""
    if req is None:
        req = ResetRequest()
    try:
        task_index = max(0, min(2, req.task_index))
        obs, info = _env.reset(task_index=task_index)
        return {
            "observation": obs.model_dump(),
            "info": info,
            "done": False,
            "reward": 0.0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Submit a SQL query and receive observation, reward, done."""
    try:
        action = SQLAction(query=req.query)
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Return current episode state."""
    try:
        s = _env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "index": i,
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_steps": t["max_steps"],
            }
            for i, t in enumerate(_env.tasks)
        ]
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
