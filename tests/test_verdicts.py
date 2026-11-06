"""Verdikte: Art (success/info), Zahlen im Text gegen die Werte, Wortwahl („hängt“, „Eine Instanz, kein Befund“), keine Zeiten."""

import pytest

import zrf_constants as C
import zrf_evaluation as E


@pytest.fixture(scope="module")
def v8():
    return E.instance_data(E.Settings())


def test_core_verdict_names_the_numbers(v8):
    kind, text = E.verdict_core(E.Settings(), v8)
    assert kind == "info" and "hängt" in text and E.num(v8["joint"]) in text and E.num(v8["alt"]["practice"]["cost"]) in text and E.num(v8["centroid"]) in text
    assert "Eine Instanz, kein Befund über die Familie" in text and "24 Zuordnungen" in text and "führen 3 zum Optimum" in text


def test_core_verdict_is_success_when_the_start_reaches_the_optimum(v8):
    kind, text = E.verdict_core(E.Settings(start="centroid"), v8)
    assert kind == "success" and "erreicht nach" in text and "hängt" not in text


def test_hand_verdict_states_both_values():
    kind, text = E.verdict_hand(E.hand_data())
    assert kind == "info" and "**34**" in text and "**51**" in text and "ABCD" in text and "DACB" in text and "50 %" in text


def test_alternating_verdict_lists_the_rounds(v8):
    kind, text = E.verdict_alternating(E.Settings(), v8)
    assert "83,75 → 83,75" in text and "nicht" in text
    assert "Fixpunkt ist das gemeinsame Optimum" in E.verdict_alternating(E.Settings(start="centroid"), v8)[1]


def test_basin_verdict_counts_the_starts(v8):
    text = E.verdict_basins(E.Settings(), v8)[1]
    assert "24 Startzuordnungen" in text and "**8**" in text and "3 von 24" in text and "−0,00" not in text and "-0,00" not in text


def test_relax_verdict_states_the_check_as_a_consequence(v8):
    text = E.verdict_relax(E.Settings(), v8)[1]
    assert "folgt aus der Definition" in text and E.de(v8["lb"], 2) in text


def _fake(joint, **kw):
    base = {m: joint for m in C.METHODS}
    base.update(kw)
    return {"kind": "random", "P": 8, "T": 4, "seed": 1, "joint": joint, "lb": joint * 0.95, "assign_at_opt_order": joint, "basins": [(joint, 3), (joint * 1.1, 1)], **base}


def test_compare_family_and_basin_verdicts_by_hand():
    rows = [_fake(100.0, practice=104.0), _fake(100.0, practice=102.0)]
    kind, text = E.verdict_compare(rows)
    assert kind == "info" and "Maschinenpraxis, dann Reihenfolge 3,00 % (2 von 2 über dem Optimum, größte 4,00 %)" in text and "5,00 %" in text and "2 von 2 Instanzen (folgt aus der Definition, kein Befund)" in text
    fam = E.verdict_family(rows, "random")[1]
    assert "Familie Zufällig (2 Instanzen)" in fam and "Maschinenpraxis, dann Reihenfolge 3,00 %" in fam
    b = E.verdict_basins_fixed(rows)[1]
    assert "2 Instanzen" in b and "75 %" in b and "in 0 von 2 Instanzen führen alle zum Optimum" in b


def test_no_verdict_contains_seconds(v8):
    texts = [E.verdict_core(E.Settings(), v8)[1], E.verdict_alternating(E.Settings(), v8)[1], E.verdict_basins(E.Settings(), v8)[1], E.verdict_relax(E.Settings(), v8)[1], E.verdict_hand(E.hand_data())[1]]
    assert not any(" s)" in t or "Sekunde" in t for t in texts)
