import pytest
from fastapi.testclient import TestClient

from misinformation_cascade_env.inference import sanitize_log_value
from misinformation_cascade_env.prompt_utils import parse_action_payload
from misinformation_cascade_env.server.app import app
from misinformation_cascade_env.server.misinformation_cascade_env_environment import (
    MisinformationCascadeEnvironment,
)
from misinformation_cascade_env.task_grader import resolve_tasks


def test_parse_action_payload_handles_fenced_json():
    raw = "```json\n{\"action_type\":\"QUARANTINE\",\"target_node_id\":\"n_7\",\"reasoning\":\"high impact\"}\n```"
    action_type, target, reasoning = parse_action_payload(raw)
    assert action_type == "QUARANTINE"
    assert target == "n_7"
    assert reasoning == "high impact"


def test_parse_action_payload_falls_back_to_wait_on_invalid_json():
    action_type, target, reasoning = parse_action_payload("not-json")
    assert action_type == "WAIT"
    assert target is None
    assert "No JSON object found" in reasoning


def test_resolve_tasks_deduplicates_selector_values():
    tasks = resolve_tasks("easy,cascade-easy,medium,medium,hard")
    ids = [task.task_id for task in tasks]
    assert ids == ["cascade-easy", "cascade-medium", "cascade-hard"]


def test_sanitize_log_value_compacts_newlines_and_tabs():
    cleaned = sanitize_log_value("line1\nline2\tline3")
    assert cleaned == "line1 line2 line3"


def test_schema_endpoint_is_available():
    client = TestClient(app)
    response = client.get("/schema")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_reset_rejects_invalid_difficulty():
    env = MisinformationCascadeEnvironment()
    with pytest.raises(ValueError, match="difficulty must be one of"):
        env.reset(difficulty="impossible")
