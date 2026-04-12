# Real-World Use Cases and KPI Mapping

This document maps environment behavior to operational scenarios commonly evaluated in trust and safety systems.

## Domain Scenarios

1. Trust and Safety Operations
- Objective: contain harmful content cascades under finite moderation budget.
- Mapped KPIs: `trust_safety`, `network_resilience`.

2. Election Integrity Monitoring
- Objective: reduce viral misinformation bursts in hub-driven networks.
- Mapped KPIs: `election_integrity`, `crisis_response`.

3. Public Health Messaging Defense
- Objective: prioritize proactive inoculation and maintain low sustained exposure.
- Mapped KPIs: `public_health`, `trust_safety`.

4. Crisis Rumor Response
- Objective: react quickly to emerging misinformation and cap peak spread.
- Mapped KPIs: `crisis_response`, `election_integrity`.

5. Platform Network Resilience Planning
- Objective: maintain containment quality while preserving intervention budget efficiency.
- Mapped KPIs: `network_resilience`, `overall_real_world_utility`.

## KPI Definitions

The evaluator in `evaluate_realworld.py` computes the following normalized domain KPIs in `[0, 1]`:

- `trust_safety`: weighted blend of terminal reward, peak infection control, and sustained control.
- `election_integrity`: weighted blend of reward, spread-event suppression, and peak control.
- `public_health`: weighted blend of sustained control, reward, and response speed.
- `crisis_response`: weighted blend of response speed, peak control, and reward.
- `network_resilience`: weighted blend of reward, peak control, and budget efficiency.
- `overall_real_world_utility`: mean of the five domain KPIs.

Supporting aggregate features per policy and difficulty:

- `avg_reward`
- `avg_peak_ratio`
- `avg_auc_ratio`
- `avg_spread_ratio`
- `avg_first_action_ratio`
- `avg_budget_spent_ratio`

## Reproducible Command

```bash
python -m misinformation_cascade_env.evaluate_realworld --episodes 20 --output artifacts/real_world_kpi_results.json
```

Output artifact:

- `artifacts/real_world_kpi_results.json`
