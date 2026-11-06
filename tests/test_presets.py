"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import zrf_constants as C
import zrf_evaluation as ev
import zrf_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: Maschinenpraxis, Cluster"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_choices():
    for p in C.PRESETS.values():
        assert p["kind"] in C.KINDS and p["p"] in C.P_CHOICES and p["t"] in C.T_CHOICES and p["p"] >= p["t"] and p["start"] in C.STARTS and 0 <= p["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def test_the_presets_differ_from_the_default_where_the_help_says_so():
    P = C.PRESETS
    assert P["Schwerpunkte als Start"]["start"] == "centroid" and P["Zufallsstart"]["start"] == "random" and P["Verschränkte Typen"]["kind"] == "interleaved" and P["Schiefe Typen"]["kind"] == "skewed"
    assert (P["Fünf Typen"]["p"], P["Fünf Typen"]["t"]) == (10, 5) and P["Zwölf Platzierungen"]["p"] == 12 and (P["Wenig Platzierungen"]["p"], P["Wenig Platzierungen"]["t"]) == (6, 3)


def test_the_help_of_the_first_preset_is_true():
    """„Der Standardfall hängt“: die wechselweise Optimierung aus der Maschinenpraxis endet nicht am Optimum, die Schwerpunkt-Zuordnung schon."""
    v = ev.instance_data(ev.Settings(**C.PRESETS[DEFAULT]))
    assert ev.stuck(v["alt"]["practice"]["cost"], v["joint"]) and not ev.stuck(v["centroid"], v["joint"])


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["kind"] = "skewed"
    at.query_params["p"] = "10"
    at.query_params["t"] = "5"
    at.query_params["seed"] = "7"
    at.query_params["start"] = "centroid"
    at.run()
    assert not at.exception
    assert (at.session_state["kind_select"], at.session_state["p_select"], at.session_state["t_select"], at.session_state["start_select"]) == ("skewed", 10, 5, "centroid") and at.sidebar.number_input(key="seed_input").value == 7
    bad = AppTest.from_file(APP, default_timeout=300)
    bad.query_params["kind"] = "gauss"
    bad.query_params["p"] = "7"
    bad.query_params["t"] = "9"
    bad.query_params["seed"] = "abc"
    bad.query_params["start"] = "x"
    bad.run()
    assert not bad.exception
    assert (bad.session_state["kind_select"], bad.session_state["p_select"], bad.session_state["t_select"], bad.session_state["seed_input"], bad.session_state["start_select"]) == (C.DEFAULT_KIND, C.DEFAULT_P, C.DEFAULT_T, C.DEFAULT_SEED, C.DEFAULT_START)


def test_the_seed_is_clamped_to_its_bounds():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["seed"] = "99999999"
    at.run()
    assert at.session_state["seed_input"] == C.SEED_MAX
