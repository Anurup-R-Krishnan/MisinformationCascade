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

OpenEnv environment for strategic containment of misinformation spread on a social graph.

This repository uses root-level submission entrypoints for the hackathon validator:

- `openenv.yaml`
- `inference.py`
- `Dockerfile`

Core environment implementation remains in [`misinformation_cascade_env/`](misinformation_cascade_env).

## Quick Start

```bash
# validate OpenEnv contract from repository root
./venv/bin/openenv validate

# run baseline inference from repository root
./venv/bin/python inference.py

# local docker build from repository root
docker build -t misinformation-cascade-openenv .
```

## Round 1 Coverage

- Real-world task: misinformation containment under finite intervention budget.
- OpenEnv spec: typed `Action` / `Observation` / `State`, plus `step/reset/state` API.
- Tasks with deterministic graders: `cascade-easy`, `cascade-medium`, `cascade-hard`.
- Reward design: shaped trajectory reward + terminal counterfactual score in `[0.0, 1.0]`.
- Baseline script: OpenAI client usage with required environment variables.
- Hugging Face deployment: Docker-based Space with `openenv` tag.

## Required Baseline Environment Variables

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

The baseline script also supports `API_KEY` / `OPENAI_API_KEY` fallback.

## Implementation Details

See the package README for full environment design, grading policy, and reproducibility artifacts:

- [`misinformation_cascade_env/README.md`](misinformation_cascade_env/README.md)
- [`misinformation_cascade_env/task_grader.py`](misinformation_cascade_env/task_grader.py)
- [`misinformation_cascade_env/env.py`](misinformation_cascade_env/env.py)
