# Benchmark Report

Generated with:

```bash
python -m misinformation_cascade_env.evaluate \
  --episodes 20 \
  --output misinformation_cascade_env/artifacts/benchmark_results.json
```

## Average Terminal Reward (higher is better)

| Policy | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| `wait` | 0.0000 | 0.0000 | 0.0000 |
| `random` | 0.1844 | 0.1482 | 0.1250 |
| `greedy_containment` | 0.7632 | 0.4468 | 0.1853 |

## Notes

- `greedy_containment` clearly outperforms baseline policies on all difficulties.
- Hard mode remains challenging due to external seeding and scale.
- Raw benchmark output is in `artifacts/benchmark_results.json`.
