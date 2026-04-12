# Benchmark Report

This report is generated from `artifacts/benchmark_results.json` using deterministic task seeds and fixed policies.

## Generation Command

```bash
python -m misinformation_cascade_env.evaluate \
  --episodes 50 \
  --output artifacts/benchmark_results.json
```

## Metadata

- generated_at_utc: 2026-04-12T07:03:11.836381+00:00
- python_version: 3.14.4
- platform: Linux-6.19.11-1-cachyos-x86_64-with-glibc2.43
- episodes: 50
- policies: wait, random, greedy_containment
- difficulties: easy, medium, hard

## Average Terminal Reward (higher is better)

| Policy | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| wait | 0.0000 | 0.0000 | 0.0000 |
| random | 0.2270 | 0.1391 | 0.1026 |
| greedy_containment | 0.7485 | 0.4610 | 0.2085 |

## Notes

- Greedy containment strongly outperforms passive and random baselines.
- Hard-mode average is now above the configured success threshold (0.2085 > 0.20), while remaining non-trivial.
- Full raw metrics (steps and end-state counts) are in `artifacts/benchmark_results.json`.
