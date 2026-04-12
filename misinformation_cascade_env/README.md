# Misinformation Cascade Env

An OpenEnv environment for strategic containment of misinformation spread on a social graph.

## Submission Note

The canonical hackathon submission entrypoints are at repository root:

- `README.md`
- `openenv.yaml`
- `inference.py`
- `Dockerfile`
- `validate-submission.sh`

This `misinformation_cascade_env/README.md` documents the implementation internals.

## Why This Environment Matters

This environment models a real operational problem: deciding scarce moderation interventions
under uncertainty while harmful content propagates through influence networks.

- Real-world analogue: trust-and-safety triage, public-health style counter-messaging,
  and rapid-response risk control.
- Non-toy decisions: every action consumes finite budget and has opportunity cost.
- Consequential dynamics: delayed intervention shifts outcomes due to latent-to-confirmed
  infection progression and network topology.

## Round-1 Tasks (Easy -> Medium -> Hard)

This environment ships with three deterministic tasks and programmatic graders:

1. `cascade-easy` (`difficulty=easy`, `seed=42`)
2. `cascade-medium` (`difficulty=medium`, `seed=137`)
3. `cascade-hard` (`difficulty=hard`, `seed=512`)

Task definitions and grader logic live in `task_grader.py`:

- `grade_episode(...)` returns a deterministic normalized score in `[0.0, 1.0]`.
- `is_task_success(...)` maps score to pass/fail with task-specific thresholds.

### Task Quality and Difficulty Progression

- `easy`: small Erdos-Renyi graph where strong early containment can fully suppress spread.
- `medium`: denser graph with tighter budget pressure and lower intervention margin.
- `hard`: hub-heavy Barabasi-Albert topology plus periodic external seeding,
  forcing trade-offs between local suppression and hub-level control.

Each task has:

- fixed seed for reproducibility,
- deterministic grader logic,
- explicit success threshold,
- 0.0-1.0 score output suitable for automated evaluation.

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

### Why the Reward Is Useful for Learning

- Dense trajectory signal: non-terminal reward captures incremental containment progress.
- Outcome signal: terminal reward measures counterfactual containment vs no-action baseline.
- Behavioral pressure: inefficient or delayed actions create negative/low intermediate reward,
  discouraging random or looping behavior.

## Difficulty Presets

- `easy`: 20 nodes, 3 seed infections, 15 steps, Erdos-Renyi
- `medium`: 35 nodes, 5 seed infections, 20 steps, Erdos-Renyi
- `hard`: 50 nodes, 7 seed infections, 25 steps, Barabasi-Albert + external seeding

## Task Success Thresholds

Thresholds are calibrated from deterministic baseline runs (`evaluate --episodes 50`):

- `easy`: `0.62` (greedy_containment avg: `0.7485`)
- `medium`: `0.40` (greedy_containment avg: `0.4610`)
- `hard`: `0.20` (greedy_containment avg: `0.2085`)

This keeps all three tasks non-trivial while aligning success criteria with observed
graph-topology difficulty and budget constraints.

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
  - `[END] success=... steps=... rewards=...`

## Benchmarking

```bash
python -m misinformation_cascade_env.evaluate --episodes 50
cat artifacts/benchmark_results.json
```

Real-world KPI evaluation:

```bash
python -m misinformation_cascade_env.evaluate_realworld --episodes 20 --output artifacts/real_world_kpi_results.json
```

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

## Creativity and Novelty

This benchmark combines mechanics not typically present together in lightweight OpenEnv tasks:

- Hidden progression dynamics (`LATENT` -> `CONFIRMED_INFECTED`) with partial observability.
- Counterfactual scoring against a seeded null trajectory for outcome-grounded reward.
- Graph-topology-aware difficulty escalation (random graph vs hub-dominant graph).
- Intervention economics where each action class has distinct cost and strategic role.

These mechanics make failure modes and policy quality visible in a way that single-step
or fully observable toy tasks do not.

## Human Review Quick Links

- Core simulator logic: `env.py`
- Deterministic graph and null trajectory generation: `graph_generator.py`
- Tasks and grading policy: `task_grader.py`
- Compliance tests: `tests/test_submission_contract.py`, `tests/test_cascade_env.py`, `tests/test_grader_quality.py`
- Reproducibility artifacts: `artifacts/benchmark_results.json`,
  `artifacts/inference_stdout.log`, `artifacts/inference_stderr.log`,
  `artifacts/real_world_kpi_results.json`

## Reproducible Validation Runbook

```bash
# 1) Install dependencies
uv sync

# 2) Run tests
pytest tests/ -v

# 3) Start local server
uv run server --port 8000

# 4) In another terminal, validate OpenEnv contract
openenv validate --url http://localhost:8000

# 5) Regenerate benchmark artifact
python -m misinformation_cascade_env.evaluate --episodes 50 --output artifacts/benchmark_results.json

# 6) Regenerate real-world KPI artifact
python -m misinformation_cascade_env.evaluate_realworld --episodes 50 --output artifacts/real_world_kpi_results.json

# 7) Run inference entrypoint with required env variables
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=<token> \
python inference.py
```
