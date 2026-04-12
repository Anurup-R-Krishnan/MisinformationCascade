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

A real-world OpenEnv benchmark for **misinformation containment** on social networks under budget and uncertainty.

## Submission Snapshot

- Live Space: `https://itsmeamario-misinformation-cascade-openenv.hf.space`
- Root-level submission files are present:
  - `openenv.yaml`
  - `inference.py`
  - `Dockerfile`
  - `validate-submission.sh`
- Latest local pre-validation: `misinformation_cascade_env/artifacts/pre_validation.log` (3/3 checks passed)

Core simulator and implementation details live in `misinformation_cascade_env/`.

## Why This Is Real-World (Not Toy)

The environment models trust-and-safety operations where interventions are limited and delayed decisions are costly.

- Actions map to practical moderation levers (`FACTCHECK`, `QUARANTINE`, `INOCULATE`, `BOOST_CORRECTION`).
- Infection can be hidden (`LATENT`) before becoming visible (`CONFIRMED_INFECTED`).
- Budget and step limits force strategic trade-offs instead of brute-force behavior.

## Action and Observation Space

Typed models are in `misinformation_cascade_env/models.py`.

### Action Space (`CascadeAction`)

| Action | Cost | Purpose |
|---|---:|---|
| `WAIT` | 0 | No intervention |
| `FACTCHECK` | 1 | Low-cost targeted correction |
| `BOOST_CORRECTION` | 2 | Amplify correction effect |
| `INOCULATE` | 3 | Preemptive protection |
| `QUARANTINE` | 5 | Hard containment |

### Observation Space (`CascadeObservation`)

Each step returns structured state including:

- `top_nodes`, `confirmed_infected`, `at_risk_nodes`
- budget and step counters
- `spread_delta_last_step`, `last_action_effect`
- `reward`, `done`

## Tasks and Graders

Three deterministic tasks are shipped (easy -> medium -> hard):

| Task ID | Difficulty | Seed | Success Threshold |
|---|---:|---:|---:|
| `cascade-easy` | easy | 42 | 0.62 |
| `cascade-medium` | medium | 137 | 0.40 |
| `cascade-hard` | hard | 512 | 0.20 |

Grading contract:

- `grade_episode(...)` returns deterministic score in `[0.0, 1.0]`
- `is_task_success(...)` applies task threshold
- implementation: `misinformation_cascade_env/task_grader.py`

Additional quality tests:

- deterministic grading trace
- grader non-constant behavior checks
- difficulty progression checks

## Reward Design

The reward function provides both trajectory and outcome signal:

- dense non-terminal shaping for incremental containment progress
- terminal counterfactual score in `[0.0, 1.0]` vs no-action trajectory
- natural penalty for wasted/invalid actions via consumed steps and missed containment

## Round 1 Requirement Mapping

| Requirement | Status | Evidence |
|---|---|---|
| Real-world utility | Pass | Misinformation containment workflow and budgeted interventions |
| OpenEnv spec compliance | Pass | `openenv validate` and typed models |
| 3+ tasks with graders | Pass | `cascade-easy`, `cascade-medium`, `cascade-hard` |
| Meaningful reward shaping | Pass | dense + terminal counterfactual reward |
| Baseline inference script | Pass | root `inference.py` with OpenAI client |
| HF Space + Docker | Pass | live Space + root Dockerfile |

## Quick Start

### 1) Static validation

```bash
./venv/bin/openenv validate
```

### 2) Run baseline inference (required entrypoint)

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=<token> \
./venv/bin/python inference.py
```

Required structured logs:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... rewards=...`

### 3) Docker build/run

```bash
docker build -t misinformation-cascade-openenv .
docker run --rm -p 8000:8000 misinformation-cascade-openenv
```

## Pre-Submission Validator

```bash
./validate-submission.sh https://itsmeamario-misinformation-cascade-openenv.hf.space .
```

The script checks:

1. Space `/reset` responds with HTTP 200
2. Docker build succeeds
3. `openenv validate` passes

## Repo Layout

```text
openenvHackathon/
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── validate-submission.sh
├── server/
└── misinformation_cascade_env/
```
