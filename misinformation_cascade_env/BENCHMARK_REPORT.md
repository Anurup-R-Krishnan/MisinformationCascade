# Benchmark Report

This report is generated from `artifacts/benchmark_results.json` using deterministic
task seeds and fixed policies.

## Generation Command

```bash
python -m misinformation_cascade_env.evaluate \
	--episodes 20 \
	--output artifacts/benchmark_results.json
```

## Metadata

- generated_at_utc: 2026-04-12T05:50:46.608908+00:00
- python_version: 3.14.4
- platform: Linux-6.19.11-1-cachyos-x86_64-with-glibc2.43
- episodes: 20
- policies: wait, random, greedy_containment
- difficulties: easy, medium, hard

## Average Terminal Reward (higher is better)

| Policy | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| wait | 0.0000 | 0.0000 | 0.0000 |
| random | 0.1844 | 0.1482 | 0.1250 |
| greedy_containment | 0.7632 | 0.4468 | 0.1853 |

## Notes

- Greedy containment strongly outperforms passive and random baselines.
- Medium and hard remain structurally difficult due to density/hub dynamics and periodic external seeding.
- Full raw metrics (steps and end-state counts) are in `artifacts/benchmark_results.json`.
