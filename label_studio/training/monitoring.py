"""
File-based monitoring helpers for Triton inference activity and metrics history.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _get_repo_root() -> Path:
    """
    Resolve the repository root from this module path.
    """

    return Path(__file__).resolve().parents[2]


def get_monitoring_root(project_id: int) -> Path:
    """
    Return the monitoring directory for a project.
    """

    root = _get_repo_root() / "data" / "training" / "monitoring" / f"project_{project_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """
    Append a single JSON object to a JSONL file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and ignore malformed lines.
    """

    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _read_last_jsonl_entry(path: Path) -> Dict[str, Any] | None:
    """
    Read the last valid JSON object from a JSONL file.
    """

    items = _read_jsonl(path)
    return items[-1] if items else None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """
    Parse an ISO datetime string into an aware datetime.
    """

    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed


def append_inference_event(project_id: int, payload: Dict[str, Any]) -> None:
    """
    Persist a single inference event for later inspection.
    """

    event = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    _append_jsonl(get_monitoring_root(project_id) / "inference_events.jsonl", event)


def append_metrics_snapshot(
    project_id: int,
    payload: Dict[str, Any],
    *,
    min_interval_seconds: int = 30,
) -> bool:
    """
    Persist a metrics snapshot when enough time passed since the previous one.
    """

    path = get_monitoring_root(project_id) / "metrics_snapshots.jsonl"
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    last_snapshot = _read_last_jsonl_entry(path)
    if last_snapshot:
        last_scope = last_snapshot.get("scope")
        last_user_id = (last_snapshot.get("selected_user") or {}).get("id")
        current_scope = snapshot.get("scope")
        current_user_id = (snapshot.get("selected_user") or {}).get("id")
        last_captured_at = _parse_iso_datetime(last_snapshot.get("captured_at"))
        if (
            last_scope == current_scope
            and last_user_id == current_user_id
            and last_captured_at is not None
            and (snapshot_time := _parse_iso_datetime(snapshot.get("captured_at"))) is not None
            and (snapshot_time - last_captured_at).total_seconds() < min_interval_seconds
        ):
            return False

    _append_jsonl(path, snapshot)
    return True


def read_inference_events(project_id: int, *, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return the latest persisted inference events.
    """

    items = _read_jsonl(get_monitoring_root(project_id) / "inference_events.jsonl")
    if limit <= 0:
        return items
    return items[-limit:]


def read_metrics_snapshots(project_id: int, *, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Return the latest persisted metrics snapshots.
    """

    items = _read_jsonl(get_monitoring_root(project_id) / "metrics_snapshots.jsonl")
    if limit <= 0:
        return items
    return items[-limit:]
