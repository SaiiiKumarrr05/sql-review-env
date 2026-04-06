"""
SQL Query Review & Optimization Environment
An OpenEnv-compatible environment where an AI agent reviews, fixes, and optimizes SQL queries.
"""

import sqlite3
import re
import time
import uuid
from typing import Optional, Any
from pydantic import BaseModel, Field


# ─────────────────────────── Typed Models ───────────────────────────

class SQLAction(BaseModel):
    """Action the agent takes: submit a SQL query string."""
    query: str = Field(..., description="The SQL query string submitted by the agent")


class SQLObservation(BaseModel):
    """Observation returned after each step."""
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Natural language description of what the agent must do")
    schema_info: str = Field(..., description="Database schema available to the agent")
    original_query: Optional[str] = Field(None, description="The original (possibly broken/slow) query to fix")
    expected_output_hint: Optional[str] = Field(None, description="Hint about expected result structure")
    last_error: Optional[str] = Field(None, description="Error from last submitted query, if any")
    last_result_preview: Optional[str] = Field(None, description="Preview of last query result (first 3 rows)")
    step_number: int = Field(0, description="Current step number in episode")
    max_steps: int = Field(5, description="Max steps allowed for this task")


class SQLReward(BaseModel):
    """Reward breakdown."""
    total: float = Field(..., ge=0.0, le=1.0, description="Total normalized reward [0,1]")
    correctness: float = Field(0.0, description="Did query produce correct results?")
    syntax_ok: float = Field(0.0, description="Did query run without syntax errors?")
    efficiency_bonus: float = Field(0.0, description="Bonus for optimized query")
    step_penalty: float = Field(0.0, description="Penalty for excessive steps")


class EpisodeState(BaseModel):
    """Full internal state of the environment."""
    session_id: str
    task_id: str
    task_index: int
    step: int
    done: bool
    last_action: Optional[str]
    last_reward: float
    cumulative_reward: float


# ─────────────────────────── Task Definitions ───────────────────────────

SCHEMA_SQL = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL NOT NULL,
    location TEXT NOT NULL
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    start_date TEXT NOT NULL,
    end_date TEXT,
    status TEXT NOT NULL
);

CREATE TABLE project_assignments (
    employee_id INTEGER REFERENCES employees(id),
    project_id INTEGER REFERENCES projects(id),
    role TEXT NOT NULL,
    hours_allocated INTEGER NOT NULL,
    PRIMARY KEY (employee_id, project_id)
);
"""

SEED_DATA_SQL = """
INSERT INTO departments VALUES
  (1,'Engineering',500000,'New York'),
  (2,'Marketing',200000,'San Francisco'),
  (3,'Sales',300000,'Chicago'),
  (4,'HR',150000,'New York');

INSERT INTO employees VALUES
  (1,'Alice Johnson','Engineering',95000,'2019-03-15',NULL),
  (2,'Bob Smith','Engineering',85000,'2020-06-01',1),
  (3,'Carol White','Marketing',75000,'2018-11-20',NULL),
  (4,'David Brown','Sales',70000,'2021-02-10',NULL),
  (5,'Eva Martinez','Engineering',90000,'2019-08-05',1),
  (6,'Frank Lee','HR',65000,'2022-01-15',NULL),
  (7,'Grace Kim','Marketing',80000,'2020-03-22',3),
  (8,'Hank Wilson','Sales',72000,'2021-07-30',4),
  (9,'Iris Chen','Engineering',88000,'2020-09-14',1),
  (10,'Jack Davis','Engineering',78000,'2022-05-01',1);

INSERT INTO projects VALUES
  (1,'Apollo',1,'2023-01-01','2023-12-31','completed'),
  (2,'Beacon',2,'2023-03-01',NULL,'active'),
  (3,'Comet',1,'2023-06-01',NULL,'active'),
  (4,'Delta',3,'2022-01-01','2023-06-30','completed'),
  (5,'Echo',1,'2024-01-01',NULL,'active');

INSERT INTO project_assignments VALUES
  (1,1,'Lead',40),(2,1,'Developer',30),(5,1,'Developer',35),
  (9,1,'QA',20),(3,2,'Lead',25),(7,2,'Analyst',20),
  (2,3,'Developer',40),(9,3,'Developer',35),(10,3,'Junior Dev',30),
  (4,4,'Lead',30),(8,4,'Associate',25),(1,5,'Architect',20),
  (5,5,'Lead',40),(10,5,'Developer',35);
"""

TASKS = [
    {
        "id": "fix_syntax_error",
        "name": "Fix Syntax Error",
        "difficulty": "easy",
        "description": (
            "The following SQL query has a syntax error. Fix it so it runs correctly and "
            "returns all employees in the Engineering department with their salary."
        ),
        "original_query": (
            "SELCT name, salary FORM employees "
            "WHER department = 'Engineering' ORDER BY salary DESC"
        ),
        "expected_columns": {"name", "salary"},
        "expected_row_count": 5,
        "max_steps": 5,
    },
    {
        "id": "optimize_n_plus_one",
        "name": "Optimize N+1 Query",
        "difficulty": "medium",
        "description": (
            "The query below uses a correlated subquery (N+1 pattern) to get each employee's "
            "department budget. Rewrite it using a JOIN so it runs efficiently. "
            "Result must have columns: employee_name, department, salary, budget."
        ),
        "original_query": (
            "SELECT e.name AS employee_name, e.department, e.salary, "
            "(SELECT d.budget FROM departments d WHERE d.name = e.department) AS budget "
            "FROM employees e"
        ),
        "expected_columns": {"employee_name", "department", "salary", "budget"},
        "expected_row_count": 10,
        "check_no_subquery": True,
        "max_steps": 5,
    },
    {
        "id": "complex_aggregation",
        "name": "Complex Aggregation Report",
        "difficulty": "hard",
        "description": (
            "Write a query from scratch that returns a department performance report. "
            "For each department, include: department name, total employees, average salary, "
            "number of active projects, and total hours allocated on active projects. "
            "Only include departments that have at least one active project. "
            "Order by total hours allocated descending. "
            "Required columns: dept_name, total_employees, avg_salary, active_projects, total_hours."
        ),
        "original_query": None,
        "expected_columns": {"dept_name", "total_employees", "avg_salary", "active_projects", "total_hours"},
        "expected_row_count": None,  # flexible
        "check_ordering": True,
        "max_steps": 7,
    },
]


# ─────────────────────────── Environment Class ───────────────────────────

class SQLReviewEnv:
    """
    OpenEnv-compatible SQL Review & Optimization Environment.
    Supports 3 tasks: easy (fix syntax), medium (optimize), hard (write from scratch).
    """

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._state: Optional[EpisodeState] = None
        self._current_task: Optional[dict] = None
        self._task_index: int = 0

    # ── DB helpers ──

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SEED_DATA_SQL)
        conn.commit()
        return conn

    def _run_query(self, query: str) -> tuple[list[dict], Optional[str]]:
        """Execute query. Returns (rows, error_or_None)."""
        try:
            cur = self._conn.execute(query)
            rows = [dict(r) for r in cur.fetchall()]
            return rows, None
        except Exception as e:
            return [], str(e)

    def _schema_info(self) -> str:
        return SCHEMA_SQL.strip()

    # ── Graders ──

    def _grade_task(self, task: dict, query: str, rows: list[dict], error: Optional[str]) -> SQLReward:
        tid = task["id"]

        if error:
            return SQLReward(total=0.0, correctness=0.0, syntax_ok=0.0)

        syntax_ok = 1.0
        correctness = 0.0
        efficiency_bonus = 0.0
        step_penalty = max(0.0, (self._state.step - 2) * 0.05)  # penalty after step 2

        # ── Easy: fix syntax ──
        if tid == "fix_syntax_error":
            got_cols = set(rows[0].keys()) if rows else set()
            col_match = task["expected_columns"].issubset(got_cols)
            row_match = len(rows) == task["expected_row_count"]
            correctness = 0.5 * int(col_match) + 0.5 * int(row_match)

        # ── Medium: optimize N+1 ──
        elif tid == "optimize_n_plus_one":
            got_cols = set(rows[0].keys()) if rows else set()
            col_match = task["expected_columns"].issubset(got_cols)
            row_match = len(rows) == task["expected_row_count"]
            no_subquery = "SELECT" not in re.sub(r"^\s*SELECT", "", query, count=1, flags=re.IGNORECASE)
            correctness = 0.4 * int(col_match) + 0.4 * int(row_match)
            efficiency_bonus = 0.2 * int(no_subquery)

        # ── Hard: complex aggregation ──
        elif tid == "complex_aggregation":
            got_cols = set(rows[0].keys()) if rows else set()
            col_match = task["expected_columns"].issubset(got_cols)
            has_rows = len(rows) > 0
            # check ordering by total_hours desc
            hours = [r.get("total_hours", 0) for r in rows]
            ordered = all(hours[i] >= hours[i + 1] for i in range(len(hours) - 1))
            has_join = "JOIN" in query.upper()
            has_group = "GROUP BY" in query.upper()
            has_having = "HAVING" in query.upper()

            correctness = (
                0.35 * int(col_match) +
                0.25 * int(has_rows) +
                0.20 * int(ordered) +
                0.10 * int(has_join) +
                0.10 * int(has_group and has_having)
            )

        total = min(1.0, max(0.0, syntax_ok * 0.1 + correctness * 0.8 + efficiency_bonus - step_penalty))
        return SQLReward(
            total=round(total, 3),
            correctness=round(correctness, 3),
            syntax_ok=syntax_ok,
            efficiency_bonus=round(efficiency_bonus, 3),
            step_penalty=round(step_penalty, 3),
        )

    # ── OpenEnv API ──

    def reset(self, task_index: int = 0) -> tuple[SQLObservation, dict]:
        """Reset environment for a given task index (0=easy,1=medium,2=hard)."""
        if self._conn:
            self._conn.close()
        self._conn = self._init_db()
        self._task_index = task_index
        self._current_task = TASKS[task_index]

        self._state = EpisodeState(
            session_id=str(uuid.uuid4()),
            task_id=self._current_task["id"],
            task_index=task_index,
            step=0,
            done=False,
            last_action=None,
            last_reward=0.0,
            cumulative_reward=0.0,
        )

        obs = SQLObservation(
            task_id=self._current_task["id"],
            task_description=self._current_task["description"],
            schema_info=self._schema_info(),
            original_query=self._current_task.get("original_query"),
            expected_output_hint=f"Expected columns: {self._current_task['expected_columns']}",
            last_error=None,
            last_result_preview=None,
            step_number=0,
            max_steps=self._current_task["max_steps"],
        )
        return obs, {"task": self._current_task["id"], "difficulty": self._current_task["difficulty"]}

    def step(self, action: SQLAction) -> tuple[SQLObservation, float, bool, dict]:
        """Execute one step: run agent's query, grade it, return obs/reward/done/info."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset().")

        self._state.step += 1
        self._state.last_action = action.query

        rows, error = self._run_query(action.query)
        reward_obj = self._grade_task(self._current_task, action.query, rows, error)
        reward = reward_obj.total

        # Episode ends on perfect score or max steps
        max_steps = self._current_task["max_steps"]
        done = reward >= 0.95 or self._state.step >= max_steps

        self._state.done = done
        self._state.last_reward = reward
        self._state.cumulative_reward += reward

        preview = None
        if rows:
            preview = str(rows[:3])

        obs = SQLObservation(
            task_id=self._current_task["id"],
            task_description=self._current_task["description"],
            schema_info=self._schema_info(),
            original_query=self._current_task.get("original_query"),
            expected_output_hint=f"Expected columns: {self._current_task['expected_columns']}",
            last_error=error,
            last_result_preview=preview,
            step_number=self._state.step,
            max_steps=max_steps,
        )

        info = {
            "reward_breakdown": reward_obj.model_dump(),
            "rows_returned": len(rows),
            "session_id": self._state.session_id,
        }

        return obs, reward, done, info

    def state(self) -> EpisodeState:
        """Return current internal state."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def tasks(self) -> list[dict]:
        return TASKS
