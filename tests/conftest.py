"""Gemeinsame Fixtures: das Handbeispiel und kleine Instanzen, an denen jede Einheit geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zrf_model as M  # noqa: E402


@pytest.fixture
def hand():
    """Zwei Typen, vier Platzierungen A (0, 0), B (4, 4), C (10, 0), D (9, 3); Handrechnung in der README."""
    return M.hand_instance()


@pytest.fixture(scope="session")
def small():
    """Sechs Platzierungen, drei Typen: alle 6 Zuordnungen und alle 720 Reihenfolgen sind aufzählbar."""
    return [M.make(k, 6, 3, 100000 + s) for k in ("random", "clustered") for s in range(2)]
