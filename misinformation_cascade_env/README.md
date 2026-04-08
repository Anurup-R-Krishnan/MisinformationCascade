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

An OpenEnv environment for strategic containment of misinformation spread on a social graph.

## Round-1 Tasks (Easy -> Medium -> Hard)

This environment ships with three deterministic tasks and programmatic graders:

1. `cascade-easy` (`difficulty=easy`, `seed=42`)
2. `cascade-medium` (`difficulty=medium`, `seed=137`)
3. `cascade-hard` (`difficulty=hard`, `seed=512`)

Task definitions and grader logic live in `task_grader.py`:

- `grade_episode(...)` returns a deterministic normalized score in `[0.0, 1.0]`.
- `is_task_success(...)` maps score to pass/fail with task-specific thresholds.

## Environment Objective

- Goal: maximize final containment score by minimizing infection damage vs a no-action baseline.
- Environment has hidden infection state (`LATENT`) and visible state (`AT_RISK` / `CONFIRMED_INFECTED`).
- Agent operates under finite budget and step limits.

## Action Space

`CascadeAction`

- `action_type`: `FACTCHECK` | `QUARANTINE` | `INOCULATE` | `BOOST_CORRECTION` | `WAIT`
- `target_node_id`: required for all actions except `WAIT`
- `reasoning`: optional string

Action costs:

- `WAIT`: 0
- `FACTCHECK`: 1
- `BOOST_CORRECTION`: 2
- `INOCULATE`: 3
- `QUARANTINE`: 5

## Observation Space

`CascadeObservation`

- `top_nodes`: top influential nodes
- `confirmed_infected`: currently confirmed infected nodes
- `at_risk_nodes`: exposed nodes (deduplicated from `top_nodes`)
- Counters: budget/step/remaining nodes and class counts
- `spread_delta_last_step`, `last_action_effect`
- `done`, `reward`

## Reward and Termination

- Non-terminal reward: shaped step reward from containment progress vs null spread.
- Terminal reward: counterfactual containment score in `[0.0, 1.0]`.
- Episode ends on first of:
  - eradication (`CONFIRMED_INFECTED == 0`)
  - saturation mercy threshold with zero budget
  - max step horizon

## Difficulty Presets

- `easy`: 20 nodes, 3 seed infections, 15 steps, Erdos-Renyi
- `medium`: 35 nodes, 5 seed infections, 20 steps, Erdos-Renyi
- `hard`: 50 nodes, 7 seed infections, 25 steps, Barabasi-Albert + external seeding

## Local Development

```bash
# from this directory
uv sync
uv run server
```

## Validation

```bash
openenv validate
openenv validate --url http://localhost:8000
```

## Required Baseline Inference Script

The hackathon-required inference script is provided as root-level `inference.py`.

Mandatory env vars:

- `API_BASE_URL` (LLM endpoint)
- `MODEL_NAME` (LLM model id)
- `HF_TOKEN` (auth token; `API_KEY`/`OPENAI_API_KEY` also accepted)

Run:

```bash
python inference.py
```

Behavior:

- Uses OpenAI client for model calls.
- Runs all 3 tasks by default (`easy,medium,hard`).
- Emits strict logs:
  - `[START] task=... env=... model=...`
  - `[STEP] step=... action=... reward=... done=... error=...`
  - `[END] success=... steps=... score=... rewards=...`

## Benchmarking

```bash
python -m misinformation_cascade_env.evaluate --episodes 20
cat artifacts/benchmark_results.json
```

Latest benchmark snapshot: `BENCHMARK_REPORT.md`

## Docker

```bash
openenv build
docker run --rm -p 8000:8000 openenv-misinformation_cascade
```

## Deployment

```bash
openenv push
```

## Reproducibility

- Deterministic graph generation and null trajectory from fixed seeds.
- Internal audit state available from `/state` for replay and debugging.
