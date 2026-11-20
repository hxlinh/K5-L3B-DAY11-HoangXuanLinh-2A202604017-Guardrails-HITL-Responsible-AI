"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}      # request_id → thời điểm bắt đầu (perf_counter)
        self._pending: dict[str, dict] = {}    # request_id → bản ghi input chờ ghép output

    def record_input(
        self, *, user_id: str, text: str, request_id: str | None = None
    ) -> str:
        """Lưu input + thời điểm bắt đầu. Trả về request_id để ghép với output."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[request_id] = time.perf_counter()
        self._pending[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            # Log cũng là nơi có thể rò rỉ → che secret/PII trước khi ghi
            "input": _mask(text),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Ghép output với input tương ứng, tính latency, append vào self.logs."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        entry = self._pending.pop(key, None) or {
            "request_id": key,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": None,
        }
        entry.update(
            {
                "output": _mask(text),
                "blocked": blocked,
                "layer": layer,
                "latency_ms": (
                    round((time.perf_counter() - started) * 1000, 1)
                    if started is not None
                    else None
                ),
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def _mask(text: str | None) -> str | None:
    if text is None:
        return None
    return content_filter(text)["redacted"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
