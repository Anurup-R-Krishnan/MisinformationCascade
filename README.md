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

An OpenEnv benchmark for **real-world misinformation containment** under budget, uncertainty, and network effects.

This repo is submission-ready at the root (validator-facing files are all here):

- `openenv.yaml`
- `inference.py`
- `Dockerfile`
- `validate-submission.sh`

Core simulator implementation lives in `misinformation_cascade_env/`.

## Table of Contents

1. Environment Description
2. Observation and Action Space
3. Tasks and Graders
4. Reward Design
5. Setup and Usage
6. Submission Validation
7. Project Structure

## 1. Environment Description

The environment models a trust-and-safety workflow: identify and contain harmful information spreading through a social graph with limited intervention budget.

Why this is real-world (not toy):

- Actions map to practical interventions (`FACTCHECK`, `QUARANTINE`, `INOCULATE`, `BOOST_CORRECTION`).
- Consequences are path-dependent (hidden latent infections become confirmed later).
- Resource constraints force trade-offs between immediate suppression and long-horizon prevention.

## 2. Observation and Action Space

Typed models are defined in `misinformation_cascade_env/models.py`.

Observation (`CascadeObservation`) includes:

- ranked influential nodes (`top_nodes`)
- `confirmed_infected` and `at_risk_nodes`
- budget and step counters
- `spread_delta_last_step`, `last_action_effect`
- `reward`, `done`

Action (`CascadeAction`) supports:

- `FACTCHECK` (cost 1)
- `BOOST_CORRECTION` (cost 2)
- `INOCULATE` (cost 3)
- `QUARANTINE` (cost 5)
- `WAIT` (cost 0)

## 3. Tasks and Graders

Three deterministic tasks are shipped (easy -> medium -> hard), each with fixed seed and threshold:

| Task ID | Difficulty | Seed | Success Threshold |
|---|---:|---:|---:|
| `cascade-easy` | easy | 42 | 0.62 |
| `cascade-medium` | medium | 137 | 0.40 |
| `cascade-hard` | hard | 512 | 0.20 |

Grader contract:

- deterministic 0.0-1.0 scoring via `grade_episode(...)`
- task success via `is_task_success(...)`
- task registry and selectors in `misinformation_cascade_env/task_grader.py`

Quality guards:

- deterministic episode tests
- non-constant grader checks across policy quality
- schema and contract tests

## 4. Reward Design

Reward is designed for learning signal across trajectory:

- **dense non-terminal shaping** for incremental containment progress
- **terminal counterfactual score** in `[0.0, 1.0]` vs seeded no-action trajectory
- penalties for inefficient/invalid behavior via world advance + opportunity loss

This prevents purely sparse binary reward and supports policy improvement.

## 5. Setup and Usage

### Local validation

```bash
./venv/bin/openenv validate
```

### Run baseline inference (required entrypoint)

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=<token> \
./venv/bin/python inference.py
```

Expected structured logs:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... rewards=...`

### Build and run Docker image

```bash
docker build -t misinformation-cascade-openenv .
docker run --rm -p 8000:8000 misinformation-cascade-openenv
```

## 6. Submission Validation

Pre-submission helper script (root):

```bash
./validate-submission.sh https://itsmeamario-misinformation-cascade-openenv.hf.space .
```

Latest local run log:

- `misinformation_cascade_env/artifacts/pre_validation.log`

Latest benchmark artifacts:

- `misinformation_cascade_env/artifacts/benchmark_results.json`
- `misinformation_cascade_env/BENCHMARK_REPORT.md`
- `misinformation_cascade_env/artifacts/inference_stdout.log`
- `misinformation_cascade_env/artifacts/inference_stderr.log`
- `misinformation_cascade_env/artifacts/real_world_kpi_results.json`

## 7. Project Structure

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
    ├── server/
    ├── tests/
    └── artifacts/
```
