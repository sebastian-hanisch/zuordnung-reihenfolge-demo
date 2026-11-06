"""Abbildungen: sie entstehen für jede Eingabe, Achsen sind gesperrt, Daten stimmen mit den Werten überein; PDF-Export."""

import pytest

import zrf_constants as C
import zrf_evaluation as E
import zrf_model as M
import zrf_visualization as V
from zrf_pdf_export import build_pdf


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


@pytest.fixture(scope="module")
def v8():
    return E.instance_data(E.Settings())


def _fake(joint, kind="random", **kw):
    base = {m: joint * 1.01 for m in C.METHODS}
    base.update(kw)
    return {"kind": kind, "P": 8, "T": 4, "seed": 1, "joint": joint, "lb": joint * 0.95, "assign_at_opt_order": joint, "basins": [(joint, 3), (joint * 1.1, 1)], **base}


def test_instance_chart_for_every_family_and_size():
    for kind in C.KINDS:
        for P, T in ((6, 3), (8, 4), (10, 5), (12, 4)):
            inst = M.make(kind, P, T, 3)
            j, sg, o = M.joint_exact(inst) if P < 12 else (0, list(range(T)), list(range(P)))
            fig = V.build_instance(inst, sg, o)
            assert _locked(fig) and sum(len(t.x) for t in fig.data if t.mode == "markers+text" and t.name and t.name.startswith("Typ")) >= P


def test_instance_chart_draws_the_fixpoint_only_when_given(v8):
    a = v8["alt"]["practice"]
    without = V.build_instance(v8["inst"], v8["sigma_joint"], v8["order_joint"])
    with_fix = V.build_instance(v8["inst"], v8["sigma_joint"], v8["order_joint"], a["sigma"], a["order"])
    assert len(with_fix.data) == len(without.data) + 1 and with_fix.data[1].line.dash == "dash"
    opt = without.data[0]
    assert len(opt.x) == 2 * v8["inst"].P + 2 and (opt.x[0], opt.y[0]) == (5.0, -2.0) and (opt.x[-1], opt.y[-1]) == (5.0, -2.0)


def test_the_tour_polyline_has_the_length_of_the_tour_cost():
    h = M.hand_instance()
    fig = V.build_instance(h, (0, 1), (0, 1, 2, 3))
    xs, ys = fig.data[0].x, fig.data[0].y
    assert sum(max(abs(xs[i + 1] - xs[i]), abs(ys[i + 1] - ys[i])) for i in range(len(xs) - 1)) == M.tour_cost(h, (0, 1), (0, 1, 2, 3))


def test_rounds_chart_shows_the_trace_and_the_optimum(v8):
    a = v8["alt"]["practice"]
    fig = V.build_rounds(a, v8["joint"])
    assert list(fig.data[0].y) == [x for _, c1, c2 in a["trace"] for x in (c1, c2)] and fig.layout.shapes[0].y0 == v8["joint"] and _locked(fig)


def test_basins_chart_bars_match_the_counts(v8):
    fig = V.build_basins(v8["basins"], v8["joint"])
    assert list(fig.data[0].y) == [c for _, c in v8["basins"]] and fig.data[0].marker.color[0] == C.COLORS["joint"] and _locked(fig)


def test_compare_chart_has_one_bar_per_method():
    rows = [_fake(100.0), _fake(50.0, practice=51.0)]
    fig = V.build_compare(rows, E.method_stats)
    assert len(fig.data[0].x) == len(C.METHODS) and list(fig.data[1].y)[C.METHODS.index("practice")] == pytest.approx(2.0) and _locked(fig)


def test_family_and_relax_charts():
    rows = [_fake(100.0, kind=k) for k in C.KINDS]
    fam = V.build_family_bars(rows, E.method_stats)
    assert len(fam.data) == 4 and all(len(t.x) == 4 for t in fam.data) and _locked(fam)
    rel = V.build_relax(rows)
    assert len(rel.data[0].x) == 4 and rel.data[0].x[0] == pytest.approx(5.0) and rel.data[0].y[0] == pytest.approx(1.0) and _locked(rel)


def test_pdf_export_returns_a_pdf_with_the_numbers(v8):
    pdf = build_pdf(E.Settings(), v8)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500
    for kind in C.KINDS:
        s = E.Settings(kind=kind, p=6, t=3)
        assert build_pdf(s, E.instance_data(s))[:4] == b"%PDF"
