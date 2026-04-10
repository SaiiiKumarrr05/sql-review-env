---
title: SQL Review Env
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SQL Review & Optimization Environment

An **OpenEnv-compatible** real-world environment where AI agents review, fix, and optimize SQL queries against a live SQLite database.

---

## Motivation

SQL code review and optimization is a task every data engineer and backend developer does daily. This environment simulates that workflow — giving an agent a broken or inefficient query and asking it to fix or improve it. Tasks are graded programmatically with deterministic scoring.

---

## Tasks

| # | ID | Difficulty | Description |
|---|-----|-----------|-------------|
| 0 | `fix_syntax_error` | Easy | Fix a broken SQL query with typos (SELCT, FORM, WHER) |
| 1 | `optimize_n_plus_one` | Medium | Rewrite a correlated subquery using a JOIN |
| 2 | `complex_aggregation` | Hard | Write a multi-table aggregation report from scratch |

---

## Database Schema

```sql
employees(id, name, department, salary, hire_date, manager_id)
departments(id, name, budget, location)
projects(id, name, department_id, start_date, end_date, status)
project_assignments(employee_id, project_id, role, hours_allocated)
```

---

## Action & Observation Spaces

**Action:** A SQL query string submitted as `{"query": "SELECT ..."}` to `POST /step`.

**Observation fields:**
- `task_id` — current task name
- `task_description` — what the agent must do
- `schema_info` — full DDL schema
- `original_query` — broken/slow query to fix (if applicable)
- `expected_output_hint` — required columns
- `last_error` — error from previous attempt
- `last_result_preview` — first 3 rows of last result
- `step_number` / `max_steps` — progress tracking

---

## Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Syntax OK | 0.1 | Query runs without error |
| Correctness | 0.8 | Column match + row count match |
| Efficiency bonus | 0.2 | No subquery in medium task |
| Step penalty | -0.05/step | Penalty after step 2 |

All rewards normalized to **[0.0, 1.0]**.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — returns `{"status":"ok"}` |
| POST | `/reset` | Reset env: `{"task_index": 0}` (0=easy,1=medium,2=hard) |
| POST | `/step` | Submit query: `{"query": "SELECT ..."}` |
| GET | `/state` | Current episode state |
| GET | `/tasks` | List all tasks |

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t sql-review-env .
docker run -p 7860:7860 sql-review-env
```

### Run Inference

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

> **Note:** `HF_TOKEN` is read from environment only — never hardcoded. Set it as a HuggingFace Space secret.

---

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| fix_syntax_error | Qwen2.5-72B | ~0.90 |
| optimize_n_plus_one | Qwen2.5-72B | ~0.75 |
| complex_aggregation | Qwen2.5-72B | ~0.55 |

---

## Project Structure

```
sql-review-env/
├── env.py           # Core environment (models, graders, logic)
├── server.py        # FastAPI HTTP server (OpenEnv endpoints)
├── inference.py     # Baseline inference script (structured logs)
├── openenv.yaml     # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Structured Log Format

`inference.py` emits JSON logs in the mandatory format:

```
[START] {"task_id": "fix_syntax_error", "task_index": 0, "max_steps": 5, "model": "..."}
[STEP]  {"task_id": "...", "step": 1, "action": "SELECT ...", "reward": 0.9, "done": false, ...}
[END]   {"task_id": "...", "task_index": 0, "total_steps": 1, "final_reward": 0.9, "success": true}
```
