"""Shared test fixtures and filesystem builders for `agy-token-stats` behavior tests."""

from __future__ import annotations

from datetime import timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def write_jsonl(path: Path, records: list[dict | str]) -> None:
    """Write a sequence of JSON-serializable dicts or raw strings to `path` as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for rec in records:
        if isinstance(rec, str):
            lines.append(rec)
        else:
            lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_write_jsonl = write_jsonl


def write_subagent_descriptor(
    parent_session_dir: Path,
    *,
    child_id: str,
    type_name: str,
    role: str,
    spawn_step_index: int = 2,
    workspace_uris: list[str] | None = None,
) -> None:
    """Write a `.system_generated/subagents/<child_id>.json` descriptor inside `parent_session_dir`."""
    sub_dir = parent_session_dir / ".system_generated/subagents"
    sub_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "conversationId": child_id,
        "subagentDescriptor": {
            "typeName": type_name,
            "role": role,
        },
        "spawnStepIndex": spawn_step_index,
        "workspaceUris": workspace_uris or ["file:///Users/test/repo"],
    }
    (sub_dir / f"{child_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


_write_subagent_descriptor = write_subagent_descriptor


class TempBrainTestCase(unittest.TestCase):
    """Base test case providing a clean temporary directory per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
