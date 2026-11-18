"""Unit tests for SafeSync-Mesh wire codec, CRC32 verification, and published result integrity."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_quickstart import verify_published_tables, verify_wire_protocol_and_idempotence


def test_wire_codec_and_idempotence() -> None:
    verify_wire_protocol_and_idempotence()


def test_published_result_tables() -> None:
    verify_published_tables()
