---
title: Misinformation Cascade Env
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Misinformation Cascade Env

A real-world OpenEnv benchmark for RL agents and LLMs performing **misinformation containment** under budget, uncertainty, and graph-driven spread dynamics.

Built for the **Meta × Hugging Face × PyTorch OpenEnv Hackathon 2026**.

## Table of Contents

1. Environment Description & Motivation
2. Observation & Action Space
3. Task Descriptions & Difficulty
4. Inference & Results
5. Visual Workflow
6. Setup & Usage Instructions
7. System Architecture
8. Project Structure
9. Pre-Validation Results
10. Team

## 1. Environment Description & Motivation

### Overview

`Misinformation Cascade Env` simulates a trust-and-safety triage workflow over a social graph. At each step, the agent observes high-impact nodes, current infections, and at-risk users, then chooses one intervention action. Hidden latent spread can later become confirmed harm, so delayed or wasteful actions reduce final outcomes.

### Motivation

Real moderation teams face limited intervention capacity and noisy prioritization. This environment was designed to capture those constraints in a form suitable for agent training and evaluation:

- finite budget with action cost trade-offs
- partial observability (`LATENT` progression)
- topology-aware spread dynamics (random graphs vs hub-heavy graphs)
- deterministic task seeds for reproducible benchmarking

## 2. Observation & Action Space

Typed models are implemented with Pydantic in `misinformation_cascade_env/models.py`.

### 2.1 Observation Space (`CascadeObservation`)

Per-step observation includes:

- `top_nodes`: highest influence nodes (risk-priority candidates)
- `confirmed_infected`: visible active spreaders
- `at_risk_nodes`: exposed nodes likely to convert from latent state
- resource counters: `budget_remaining`, `step`, `max_steps`, `steps_remaining`
- spread feedback: `spread_delta_last_step`, `last_action_effect`
- trajectory values: `reward`, `done`

### 2.2 Action Space (`CascadeAction`)

The agent emits one action per step:

| Action | Cost | Purpose |
|---|---:|---|
| `WAIT` | 0 | Skip intervention |
| `FACTCHECK` | 1 | Low-cost targeted correction |
| `BOOST_CORRECTION` | 2 | Raise correction pressure |
| `INOCULATE` | 3 | Preemptive protection |
| `QUARANTINE` | 5 | Hard containment |

## 3. Task Descriptions & Difficulty

Three deterministic tasks are included (easy -> medium -> hard), each with fixed seed and grader threshold.

| Task ID | Difficulty | Seed | Goal | Success Threshold |
|---|---:|---:|---|---:|
| `cascade-easy` | easy | 42 | Early containment on smaller graph | 0.62 |
| `cascade-medium` | medium | 137 | Balance budget vs wider spread surface | 0.40 |
| `cascade-hard` | hard | 512 | Contain hub-heavy + external seeding dynamics | 0.20 |

Graders are deterministic and return scores in `[0.0, 1.0]` (`misinformation_cascade_env/task_grader.py`).

## 4. Inference & Results

The required root inference script is:

- `inference.py`

Latest baseline run (offline advisor fallback, no API key configured):

- `cascade-easy`: success=true, steps=3
- `cascade-medium`: success=true, steps=8
- `cascade-hard`: success=true, steps=9
- aggregate average score: `0.5847`

Structured log format emitted:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... rewards=...`

## 5. Visual Workflow

The OpenEnv web interface is available when the server is running:

- `/web` for interactive step/reset exploration
- `/docs` for OpenAPI endpoint exploration

This allows manual inspection of node states, intervention effects, and reward behavior during an episode.

## 6. Setup & Usage Instructions

### 6.1 Build and Run Docker Container

```bash
# Build (root-level Dockerfile used by validator)
docker build -t misinformation-cascade-openenv .

# Run
docker run --rm -p 8000:8000 misinformation-cascade-openenv
```

### 6.2 Run Baseline Inference

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=<your_token> \
./venv/bin/python inference.py
```

### 6.3 Validate Submission Contract

```bash
./validate-submission.sh https://itsmeamario-misinformation-cascade-openenv.hf.space .
```

## 7. System Architecture

```text
+-------------------------------------------------------------+
|                     OpenEnv Client Loop                     |
|           (LLM / policy -> inference.py -> HTTP)           |
+-------------------------------+-----------------------------+
                                |
                                v
+-------------------------------------------------------------+
|                FastAPI OpenEnv Environment Server           |
|                         server/app.py                       |
+-------------------------------+-----------------------------+
                                |
                                v
+-------------------------------------------------------------+
|             Misinformation Cascade Simulation Core          |
|        env.py + graph_generator.py + task_grader.py        |
|                                                             |
|  Graph State -> Spread Dynamics -> Action Effects -> Reward |
+-------------------------------------------------------------+
```

## 8. Project Structure

```text
openenvHackathon/
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── validate-submission.sh
├── server/
│   ├── app.py
│   └── misinformation_cascade_env_environment.py
└── misinformation_cascade_env/
    ├── env.py
    ├── models.py
    ├── task_grader.py
    ├── offline_advisor.py
    ├── tests/
    └── artifacts/
```

## 9. Pre-Validation Results

Current status: **3/3 checks passed**.

- HF Space `/reset` responded with HTTP 200
- Docker build succeeded
- `openenv validate` passed

Evidence log:

- `misinformation_cascade_env/artifacts/pre_validation.log`

## 10. Team

- Anurup R Krishnan
