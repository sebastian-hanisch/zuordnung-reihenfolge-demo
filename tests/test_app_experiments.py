"""Pfade der App, die der Rauchtest nicht erreicht: die Experimente im Seiteninhalt, PDF, jede Familie und jeder Start, 12 Platzierungen mit 5 Typen, die Rechenknöpfe (DEMO-PLAYBOOK §1d).

Der Rechenknopf der 240 festen Instanzen wird mit einer verkleinerten Reihe geprüft (vier Instanzen); die echte Reihe rechnet test_claims.py."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import zrf_constants as C
import zrf_evaluation as E

APP = str(Path(__file__).resolve().parent.parent / "app.py")
KIND = {"kind_select": "selectbox", "p_select": "select_slider", "t_select": "select_slider", "seed_input": "number_input", "start_select": "selectbox"}
LABELS = {"Gemeinsames Optimum", "Wechselweise Optimierung", "Schwerpunkt-Zuordnung, dann Reihenfolge", "Untere Schranke", "Startzuordnungen, die zum Optimum führen"}


@pytest.fixture(autouse=True)
def _clear_caches():
    """st.cache_data gilt im ganzen Prozess: Tests, die Konstanten ändern, dürfen keine Ergebnisse an andere weitergeben."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        getattr(at.sidebar, KIND[key])(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown) + list(at.caption))


def _metrics(at):
    return {m.label: m for m in at.metric}


def test_the_default_page_shows_the_five_metrics_and_the_core_verdict():
    at = _app()
    m = _metrics(at)
    assert set(m) == LABELS and m["Gemeinsames Optimum"].value == "74,23" and m["Wechselweise Optimierung"].value == "76,35" and "hängt" in m["Wechselweise Optimierung"].delta
    assert m["Startzuordnungen, die zum Optimum führen"].value == "7 von 24"
    assert "Eine Instanz, kein Befund über die Familie" in _texts(at)


def test_the_five_experiments_and_the_limits_are_on_the_page():
    text = _texts(_app())
    for head in ("Experiment 1: Das Handbeispiel", "Experiment 2: Wechselweise Optimierung", "Experiment 3: Fangbecken", "Experiment 4: Die Trennungen im Vergleich", "Experiment 5: Relaxation und Rundung", "Wo die Annahmen enden", "Mathematische Formulierung"):
        assert head in " ".join(str(s.value) for s in _app().subheader) + text, head
    assert "**34**" in text and "**51**" in text


def test_every_kind_and_every_start_renders():
    for kind in C.KINDS:
        assert not _app(kind_select=kind).exception
    for start in C.STARTS:
        at = _app(start_select=start)
        assert f"Start {C.START_NAMES[start]}" in _texts(at)


def test_the_centroid_start_reaches_the_optimum_on_the_default_instance():
    at = _app(start_select="centroid")
    m = _metrics(at)
    assert "hängt" not in (m["Wechselweise Optimierung"].delta or "") and m["Wechselweise Optimierung"].value == m["Gemeinsames Optimum"].value
    assert "erreicht nach" in _texts(at)


def test_twelve_placements_with_five_types_waits_for_the_knob():
    at = _app(p_select=12, t_select=5)
    assert not at.metric and any("Rechnen" in b.label for b in at.button)


def test_pdf_button_makes_a_download():
    at = _app()
    next(b for b in at.button if "PDF" in b.label).click().run(timeout=300)
    assert not at.exception


def test_fixed_instances_knob_with_a_small_series(monkeypatch):
    small = [("random", 6, 3, 100000), ("clustered", 6, 3, 100001), ("skewed", 6, 3, 100002), ("interleaved", 6, 3, 100003)]
    monkeypatch.setattr(E, "jobs", lambda: small)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    next(b for b in at.button if "Feste Instanzen" in b.label).click().run(timeout=300)
    assert not at.exception and len(at.session_state["rows_cache"]) == 4
    text = _texts(at)
    assert "Familie Zufällig (1 Instanzen)" in text and "Familie Cluster (1 Instanzen)" in text and "Fangbecken über 4 Instanzen" in text and "Über 4 Instanzen" in text
