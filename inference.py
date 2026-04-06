"""
inference.py — SQL Review & Optimization Environment
Baseline inference script using OpenAI client against the SQL Review OpenEnv environment.

Environment variables required:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import os
import textwrap
import requests
from typing import Optional
from openai import OpenAI

# ── Config ──
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "sql-review-env"
MAX_STEPS = 7
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5


# ── Logger (exact required format) ──

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ").strip()
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── Env HTTP client ──

def env_reset(task_index: int) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_index": task_index}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(query: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"query": query}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── System prompt ──

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL engineer. You will be given a SQL task to solve.
    You have access to a SQLite database with this schema:

    - employees(id, name, department, salary, hire_date, manager_id)
    - departments(id, name, budget, location)
    - projects(id, name, department_id, start_date, end_date, status)
    - project_assignments(employee_id, project_id, role, hours_allocated)

    Rules:
    - Reply with ONLY the SQL query. No explanation, no markdown, no backticks.
    - Use standard SQLite syntax.
    - Ensure column aliases match exactly what is required.
    - If you see an error from a previous attempt, fix it in your next reply.
""").strip()


def build_user_prompt(obs: dict, step: int) -> str:
    parts = [
        f"Task: {obs['task_description']}",
        f"Step: {step} / {obs['max_steps']}",
    ]
    if obs.get("original_query"):
        parts.append(f"Original query (may be broken/slow):\n{obs['original_query']}")
    if obs.get("expected_output_hint"):
        parts.append(f"Hint: {obs['expected_output_hint']}")
    if obs.get("last_error"):
        parts.append(f"Last error: {obs['last_error']}")
    if obs.get("last_result_preview"):
        parts.append(f"Last result preview: {obs['last_result_preview']}")
    parts.append("Write the correct SQL query now:")
    return "\n\n".join(parts)


def get_sql_from_model(client: OpenAI, obs: dict, step: int) -> str:
    user_prompt = build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # strip markdown code fences if model adds them
        text = text.replace("```sql", "").replace("```", "").strip()
        return text if text else "SELECT 1"
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return "SELECT 1"


def run_task(client: OpenAI, task_index: int, task_id: str) -> float:
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_index)
        obs = result["observation"]
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            query = get_sql_from_model(client, obs, step)
            result = env_step(query)

            obs = result["observation"]
            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)
            error = obs.get("last_error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=query, reward=reward, done=done, error=error)

            if done:
                break

        score = max(rewards) if rewards else 0.0  # best score across steps
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        {"index": 0, "id": "fix_syntax_error"},
        {"index": 1, "id": "optimize_n_plus_one"},
        {"index": 2, "id": "complex_aggregation"},
    ]

    print(f"[DEBUG] Running {len(tasks)} tasks against {ENV_BASE_URL}", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)

    all_scores = []
    for t in tasks:
        score = run_task(client, t["index"], t["id"])
        all_scores.append(score)
        print(f"[DEBUG] Task {t['id']} final score: {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores)
    print(f"[DEBUG] Average score across all tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
