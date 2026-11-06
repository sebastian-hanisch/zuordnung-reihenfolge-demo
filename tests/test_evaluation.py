"""Auswertung: Formatierung, Invarianten der Verfahren auf einer Instanz, Zeilen der festen Reihen, Statistiken."""

import math

import pytest

import zrf_constants as C
import zrf_evaluation as E
import zrf_model as M


@pytest.fixture(scope="module")
def v8():
    return E.instance_data(E.Settings())


def test_formatting():
    assert E.de(1.234) == "1,23" and E.de(-0.5, 1) == "−0,5" and E.pct(2.0, 1) == "2,0 %" and E.pct(3.14159, 2) == "3,14 %"
    assert E.num(34.0) == "34" and E.num(74.2314) == "74,23" and E.count(362880) == "362 880" and E.kind_name("clustered") == "Cluster"
    assert E.sigma_text((1, 0)) == "Typ 0 → Platz 1, Typ 1 → Platz 0"


def test_gap_and_stuck_use_the_tolerance():
    assert E.gap(34.0, 34.0) == 0.0 and E.gap(34.0 + 1e-8, 34.0) == 0.0 and E.gap(51.0, 34.0) == pytest.approx(50.0)
    assert not E.stuck(34.0 + 1e-8, 34.0) and E.stuck(34.1, 34.0)


def test_settings_default_to_the_default_constants():
    s = E.Settings()
    assert (s.kind, s.p, s.t, s.seed, s.start) == (C.DEFAULT_KIND, C.DEFAULT_P, C.DEFAULT_T, C.DEFAULT_SEED, C.DEFAULT_START)


def test_the_default_instance_is_stuck_from_the_default_start(v8):
    """Die Standardinstanz ist so gewählt, dass die wechselweise Optimierung aus der Maschinenpraxis hängt (damit die erste Ansicht die Kernaussage zeigt)."""
    assert E.stuck(v8["alt"][C.DEFAULT_START]["cost"], v8["joint"]) and not E.stuck(v8["centroid"], v8["joint"])


def test_every_method_is_at_or_above_the_joint_optimum(v8):
    for k in C.METHODS:
        assert v8[k] >= v8["joint"] - 1e-7, k
    assert v8["lb"] <= v8["joint"] + 1e-7


def test_alternating_never_loses_against_its_start(v8):
    inst = v8["inst"]
    assert v8["alt_practice"] <= v8["practice"] + 1e-7 and v8["alt_centroid"] <= v8["centroid"] + 1e-7
    assert v8["alt_random3"] <= v8["alt_random1"] + 1e-7 and v8["order_first_alt"] <= v8["order_first"] + 1e-7
    assert v8["practice"] == pytest.approx(M.best_order(inst, v8["sigma_practice"])[0]) and v8["centroid"] == pytest.approx(M.best_order(inst, v8["sigma_centroid"])[0])


def test_assignment_at_the_optimal_order_is_the_optimum(v8):
    """Folgt aus der Definition des Optimums (Gegenprobe des linearen Schritts), kein Befund."""
    assert v8["assign_at_opt_order"] == pytest.approx(v8["joint"])


def test_basins_count_every_start_assignment_once(v8):
    assert sum(c for _, c in v8["basins"]) == math.factorial(4) and all(k >= v8["joint"] - 1e-6 for k, _ in v8["basins"])
    assert any(not E.stuck(k, v8["joint"]) for k, _ in v8["basins"])           # von irgendeinem Start aus wird das Optimum erreicht (der Start mit dem optimalen σ ist ein Fixpunkt)


def test_alternating_traces_belong_to_the_start(v8):
    for key, a in v8["alt"].items():
        assert a["trace"][0][0] == 1 and a["rounds"] == len(a["trace"]) and a["cost"] == pytest.approx(M.tour_cost(v8["inst"], a["sigma"], a["order"]))


def test_hand_data():
    h = E.hand_data()
    assert h["joint"] == 34 and h["good"]["cost"] == 34 and h["bad"]["cost"] == 51 and h["bad"]["check"] == 51 and h["lb"] <= 34
    assert sum(d for _, _, d in h["bad"]["legs"]) == 51 and sum(d for _, _, d in h["tour_joint"]) == 34


def test_row_has_exactly_the_row_keys_and_is_deterministic():
    r = E.row("random", 6, 3, 100000)
    assert set(r) == {"kind", "P", "T", "seed", "sigma_joint", *E.ROW_KEYS} and r == E.row("random", 6, 3, 100000)


def test_jobs_are_the_240_fixed_instances():
    jobs = E.jobs()
    assert len(jobs) == 240 and len(set(jobs)) == 240 and all(j[3] >= C.FIXED_SEED0 for j in jobs)
    assert {(j[1], j[2]) for j in jobs} == set(C.CONFIGS) and {j[0] for j in jobs} == set(C.KINDS)


def test_sweep_selects_one_cell():
    rs = E.sweep("random", 8, 4)
    assert len(rs) == C.SEEDS and all(r["kind"] == "random" and (r["P"], r["T"]) == (8, 4) for r in rs)


def _fake(joint, **kw):
    base = {m: joint for m in C.METHODS}
    base.update(kw)
    return {"kind": "random", "P": 8, "T": 4, "seed": 1, "joint": joint, "lb": joint * 0.95, "assign_at_opt_order": joint, "basins": [(joint, 3), (joint * 1.1, 1)], **base}


def test_statistics_by_hand():
    rows = [_fake(100.0, practice=104.0), _fake(100.0, practice=100.0), _fake(50.0, practice=51.0, centroid=50.0)]
    st = E.method_stats(rows, "practice")
    assert st["count"] == 3 and st["stuck"] == 2 and st["max"] == pytest.approx(4.0) and st["mean"] == pytest.approx((4.0 + 0.0 + 2.0) / 3) and st["median"] == pytest.approx(2.0)
    assert E.method_stats(rows, "centroid")["stuck"] == 0
    lb = E.lb_stats(rows)
    assert lb["mean"] == pytest.approx(5.0) and lb["max"] == pytest.approx(5.0)
    b = E.basin_stats(rows)
    assert b["ends_median"] == 2 and b["ends_max"] == 2 and b["share_median"] == pytest.approx(0.75) and b["all_to_optimum"] == 0
    assert E.assign_at_optimum(rows) == 3
    assert len(E.select(rows, kind="random")) == 3 and E.select(rows, kind="skewed") == [] and len(E.select(rows, config=(8, 4))) == 3 and E.select(rows, config=(10, 5)) == []
    assert E.basin_stats([_fake(10.0, basins=[(10.0, 6)])])["all_to_optimum"] == 1
