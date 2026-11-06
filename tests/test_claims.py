"""Die Zahlen der README und der Verdikte, an den 240 festen Instanzen nachgerechnet (Messung, nicht Satz).

Zählungen („über dem Optimum“, Instanzen, Fangbecken) sind **ganzzahlig** und werden genau geprüft; Prozentwerte (Lücke zum gemeinsamen Optimum) mit drei Stellen (Toleranz 0,001 Prozentpunkte). Alle Instanzen und Zufallszahlen
sind SplitMix64 mit festen Seeds (ab 100000): die Zahlen sind auf jedem Rechner gleich. Zeiten stehen hier nie."""

import statistics

import pytest

import rows as RW
import zrf_constants as C
import zrf_evaluation as E

TOL = 1.5e-3

# (Instanzen über dem Optimum, mittlere Lücke in %, größte Lücke in %)
ALL = {
    "practice": (213, 3.928, 15.419),
    "centroid": (1, 0.003, 0.616),
    "order_first": (189, 3.161, 15.212),
    "alt_practice": (176, 2.142, 12.246),
    "alt_centroid": (1, 0.003, 0.616),
    "alt_random1": (151, 1.582, 11.752),
    "alt_random3": (90, 0.571, 8.589),
    "order_first_alt": (102, 0.472, 4.903),
}
CONFIG = {
    (8, 4): {"practice": (66, 3.359, 14.571), "centroid": (1, 0.008, 0.616), "alt_practice": (50, 1.258, 9.632), "alt_random3": (26, 0.407, 4.707), "order_first": (61, 3.015, 12.269), "order_first_alt": (22, 0.459, 4.903)},
    (10, 5): {"practice": (74, 5.234, 12.778), "centroid": (0, 0.0, 0.0), "alt_practice": (63, 3.561, 12.246), "alt_random3": (36, 0.86, 8.589), "order_first": (62, 2.956, 15.212), "order_first_alt": (46, 0.594, 2.824)},
    (12, 4): {"practice": (73, 3.19, 15.419), "centroid": (0, 0.0, 0.0), "alt_practice": (63, 1.607, 8.908), "alt_random3": (28, 0.447, 4.76), "order_first": (66, 3.512, 9.126), "order_first_alt": (34, 0.364, 3.149)},
}
KIND = {
    "random": {"practice": (55, 4.241, 14.571), "centroid": (1, 0.01, 0.616), "alt_practice": (46, 1.471, 12.246), "alt_random1": (44, 1.214, 5.379), "order_first": (55, 2.259, 9.459)},
    "clustered": {"practice": (43, 3.148, 13.604), "centroid": (0, 0.0, 0.0), "alt_practice": (23, 0.462, 4.894), "alt_random1": (27, 0.604, 8.415), "order_first": (22, 0.404, 4.818)},
    "skewed": {"practice": (55, 3.681, 15.419), "centroid": (0, 0.0, 0.0), "alt_practice": (48, 2.027, 9.632), "alt_random1": (31, 1.219, 7.038), "order_first": (52, 2.614, 12.269)},
    "interleaved": {"practice": (60, 4.642, 11.176), "centroid": (0, 0.0, 0.0), "alt_practice": (59, 4.607, 11.176), "alt_random1": (49, 3.292, 11.752), "order_first": (60, 7.367, 15.212)},
}


@pytest.fixture(scope="module")
def rows():
    rs = RW.all_rows()
    for r in rs:
        r["basins"] = [tuple(b) for b in r["basins"]]
    return rs


def _check(rs, key, expect):
    s = E.method_stats(rs, key)
    assert (s["stuck"], s["mean"], s["max"]) == (expect[0], pytest.approx(expect[1], abs=TOL), pytest.approx(expect[2], abs=TOL)), key


def test_the_series_has_240_instances_in_the_order_of_the_jobs(rows):
    assert len(rows) == 240 and [(r["kind"], r["P"], r["T"], r["seed"]) for r in rows] == E.jobs()
    assert all(len(E.select(rows, kind=k, config=c)) == C.SEEDS for k in C.KINDS for c in C.CONFIGS)


@pytest.mark.parametrize("key", list(ALL))
def test_gap_to_the_joint_optimum_over_all_instances(rows, key):
    _check(rows, key, ALL[key])


@pytest.mark.parametrize("config", list(CONFIG))
def test_gap_by_size(rows, config):
    for key, expect in CONFIG[config].items():
        _check(E.select(rows, config=config), key, expect)


@pytest.mark.parametrize("kind", list(KIND))
def test_gap_by_family(rows, kind):
    for key, expect in KIND[kind].items():
        _check(E.select(rows, kind=kind), key, expect)


def test_the_centroid_assignment_is_optimal_in_239_of_240(rows):
    """Die Relaxation (Schwerpunkt-Zuordnung) mit der besten Reihenfolge: 239 von 240 optimal, die eine Ausnahme ist die Familie „Zufällig“, Seed 100006, mit 0,616 %."""
    over = [r for r in rows if E.stuck(r["centroid"], r["joint"])]
    assert len(over) == 1 and (over[0]["kind"], over[0]["seed"]) == ("random", 100006) and E.gap(over[0]["centroid"], over[0]["joint"]) == pytest.approx(0.616, abs=TOL)
    assert all(abs(r["alt_centroid"] - r["centroid"]) < 1e-7 for r in rows)        # von den Schwerpunkten aus ändert die wechselweise Optimierung nichts


def test_the_centroid_start_is_never_worse_than_the_practice_start(rows):
    assert all(r["centroid"] <= r["practice"] + 1e-7 for r in rows)


def test_alternating_from_the_practice_start_is_stuck_where_practice_is_already_a_fixpoint(rows):
    """Bei den verschränkten Typen ändert die wechselweise Optimierung die Maschinenpraxis in 59 von 60 Instanzen nicht."""
    rs = E.select(rows, kind="interleaved")
    assert sum(abs(r["alt_practice"] - r["practice"]) < 1e-7 for r in rs) == 59


def test_order_first_beats_practice_in_130_instances(rows):
    assert sum(r["order_first"] < r["practice"] - 1e-7 for r in rows) == 130


def test_alternating_never_loses_against_its_start_in_any_row(rows):
    assert all(r["alt_practice"] <= r["practice"] + 1e-7 and r["alt_random3"] <= r["alt_random1"] + 1e-7 and r["order_first_alt"] <= r["order_first"] + 1e-7 for r in rows)
    assert all(r[k] >= r["joint"] - 1e-7 for r in rows for k in C.METHODS)


def test_the_lower_bound_is_weak_but_never_above_the_optimum(rows):
    lb = E.lb_stats(rows)
    assert all(r["lb"] <= r["joint"] + 1e-7 for r in rows)
    assert (lb["mean"], lb["median"], lb["max"]) == (pytest.approx(4.624, abs=TOL), pytest.approx(4.509, abs=TOL), pytest.approx(8.743, abs=TOL))


def test_the_assignment_at_the_optimal_order_is_the_optimum_in_all_240(rows):
    """Folgt aus der Definition des Optimums (Gegenprobe des linearen Schritts), kein Befund."""
    assert E.assign_at_optimum(rows) == 240


def test_basins(rows):
    b = E.basin_stats(rows)
    assert (b["count"], b["ends_median"], b["ends_max"], b["all_to_optimum"]) == (240, 7.0, 38, 7)
    assert (b["share_median"], b["share_min"]) == (pytest.approx(0.25, abs=1e-3), pytest.approx(1 / 60, abs=1e-3))
    for r in rows:
        assert sum(c for _, c in r["basins"]) == __import__("math").factorial(r["T"])
    by_size = {c: E.basin_stats(E.select(rows, config=c)) for c in C.CONFIGS}
    assert [by_size[c]["ends_median"] for c in C.CONFIGS] == [5.0, 13.0, 6.0] and [round(by_size[c]["share_median"], 3) for c in C.CONFIGS] == [0.333, 0.179, 0.25]


def test_rounds_of_the_alternating_optimization_from_the_practice_start(rows):
    assert statistics.median(r["rounds_practice"] for r in rows) == 2 and max(r["rounds_practice"] for r in rows) == 4


def test_the_verdict_texts_carry_these_numbers(rows):
    text = E.verdict_compare(rows)[1]
    for piece in ("Maschinenpraxis, dann Reihenfolge 3,93 % (213 von 240 über dem Optimum, größte 15,42 %)", "Schwerpunkt-Zuordnung, dann Reihenfolge 0,00 % (1 von 240 über dem Optimum, größte 0,62 %)",
                  "Wechselweise aus der Maschinenpraxis 2,14 % (176 von 240", "beste von 3 Zufallsstarts 0,57 % (90 von 240", "4,62 % unter dem Optimum (größte 8,74 %)"):
        assert piece in text, piece
    b = E.verdict_basins_fixed(rows)[1]
    assert "im Median zu 7 verschiedenen Endkosten (größte Zahl 38)" in b and "25 %" in b and "in 7 von 240 Instanzen führen alle zum Optimum" in b
