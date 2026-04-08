"""
Prompt construction and response parsing helpers for inference.
"""

from __future__ import annotations

import json
import re
from typing import Optional

try:
    from .models import VALID_ACTION_TYPES, CascadeObservation
except ImportError:
    from models import VALID_ACTION_TYPES, CascadeObservation


SYSTEM_PROMPT = (
    "You are controlling a misinformation containment simulator. "
    "Return exactly one compact JSON object with keys: "
    "action_type, target_node_id, reasoning. "
    "Allowed action_type values: FACTCHECK, QUARANTINE, INOCULATE, BOOST_CORRECTION, WAIT. "
    "Use null for target_node_id only when action_type is WAIT. "
    "Do not output markdown or extra text."
)


def build_user_prompt(observation: CascadeObservation) -> str:
    top = ", ".join(
        f"{n.node_id}:{n.status}:{n.influence_score:.2f}" for n in observation.top_nodes[:6]
    ) or "none"
    confirmed = ", ".join(n.node_id for n in observation.confirmed_infected[:6]) or "none"
    at_risk = ", ".join(n.node_id for n in observation.at_risk_nodes[:8]) or "none"
    return (
        f"Step={observation.step}/{observation.max_steps}; "
        f"Budget={observation.budget_remaining}; "
        f"Infected={observation.infected_count}; "
        f"Inoculated={observation.inoculated_count}; "
        f"Quarantined={observation.quarantined_count}.\n"
        f"Top nodes: {top}\n"
        f"Confirmed infected: {confirmed}\n"
        f"At risk: {at_risk}\n"
        f"Last effect: {observation.last_action_effect}"
    )


def parse_action_payload(raw_text: str) -> tuple[str, Optional[str], str]:
    payload = _extract_json(raw_text)
    action_type = str(payload.get("action_type", "WAIT")).strip().upper()
    target = payload.get("target_node_id")
    reasoning = str(payload.get("reasoning", "")).strip()

    if action_type not in VALID_ACTION_TYPES:
        return "WAIT", None, "Invalid action from model; defaulted to WAIT."

    if action_type == "WAIT":
        return "WAIT", None, reasoning

    target_node_id = None if target is None else str(target).strip()
    if not target_node_id:
        return "WAIT", None, "Missing target for non-WAIT action; defaulted to WAIT."
    return action_type, target_node_id, reasoning


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"action_type": "WAIT", "target_node_id": None, "reasoning": "No JSON object found."}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action_type": "WAIT", "target_node_id": None, "reasoning": "Unparseable JSON payload."}
