"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import plotly.graph_objects as go

import zrf_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _tour_trace(inst, sigma, order, color, name, dash=None, width=2.5):
    """Linienzug der Fahrt: Ruheposition → Zuführplatz → Platzierung → … → Ruheposition."""
    S = inst.slots[list(sigma)]
    xs, ys = [inst.home[0]], [inst.home[1]]
    for i in order:
        s = S[inst.typ[i]]
        xs += [s[0], inst.pos[i, 0]]
        ys += [s[1], inst.pos[i, 1]]
    xs.append(inst.home[0])
    ys.append(inst.home[1])
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip")


def build_instance(inst, sigma_opt, order_opt, sigma_fix=None, order_fix=None, title: str = "Die Instanz und die Fahrten") -> go.Figure:
    """Zuführplätze (Quadrate, Nummer = Platz), Platzierungen (Kreise, Farbe = Typ, Nummer = Platzierung), die optimale Fahrt (grün) und, wenn gegeben, der Fixpunkt der wechselweisen Optimierung (rot gestrichelt)."""
    fig = go.Figure()
    fig.add_trace(_tour_trace(inst, sigma_opt, order_opt, C.COLORS["joint"], "gemeinsames Optimum"))
    if sigma_fix is not None:
        fig.add_trace(_tour_trace(inst, sigma_fix, order_fix, C.COLORS["fix"], "Fixpunkt der wechselweisen Optimierung", dash="dash", width=2))
    col = lambda t: C.COLORS["type"][t % len(C.COLORS["type"])]
    for t in range(inst.T):
        pi = [i for i in range(inst.P) if inst.typ[i] == t]
        fig.add_trace(go.Scatter(x=[inst.pos[i, 0] for i in pi], y=[inst.pos[i, 1] for i in pi], mode="markers+text", text=[str(i) for i in pi], textposition="middle center", textfont=dict(color="white", size=10),
                                 marker=dict(size=22, color=col(t)), name=f"Typ {t}", hovertemplate=f"Platzierung %{{text}}, Typ {t}<extra></extra>"))
    # Zuführplatz des Typs t unter der optimalen Zuordnung: Quadrat in der Farbe des Typs
    fig.add_trace(go.Scatter(x=[inst.slots[sigma_opt[t], 0] for t in range(inst.T)], y=[inst.slots[sigma_opt[t], 1] for t in range(inst.T)], mode="markers+text", text=[f"Typ {t}" for t in range(inst.T)], textposition="bottom center",
                             marker=dict(symbol="square", size=26, color=[col(t) for t in range(inst.T)], line=dict(width=2, color="white")), name="Zuführplatz (optimal belegt)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[inst.home[0]], y=[inst.home[1]], mode="markers", marker=dict(symbol="diamond", size=14, color="#333"), name="Ruheposition", hoverinfo="skip"))
    fig.update_xaxes(visible=False, range=[-0.8, 10.8], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[-3.4, 6.8])
    fig.update_layout(title=title)
    return _base(fig, height=420)


def build_rounds(a: dict, joint: float, title: str = "Kosten je Schritt der wechselweisen Optimierung") -> go.Figure:
    """Kosten nach jedem Reihenfolge- und Zuordnungsschritt; gestrichelt das gemeinsame Optimum."""
    xs, ys, labels = [], [], []
    for r, c1, c2 in a["trace"]:
        xs += [2 * r - 1, 2 * r]
        ys += [c1, c2]
        labels += [f"Runde {r}: Reihenfolge", f"Runde {r}: Zuordnung"]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", text=labels, hovertemplate="%{text}: %{y:.2f}<extra></extra>", line=dict(color=C.COLORS["line"]), marker=dict(size=9), name="Kosten"))
    fig.add_hline(y=joint, line=dict(color=C.COLORS["joint"], dash="dash"), annotation_text="gemeinsames Optimum", annotation_position="top right")
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title="Schritt (ungerade: Reihenfolge, gerade: Zuordnung)", dtick=1)
    fig.update_yaxes(title="Kosten")
    return _base(fig, height=320, legend_below=False)


def build_basins(basins: list, joint: float, title: str = "Wohin laufen die Startzuordnungen?") -> go.Figure:
    """Je Endkosten (Lücke zum Optimum in Prozent) die Zahl der Startzuordnungen, die dort enden."""
    gaps = [100.0 * (k - joint) / joint for k, _ in basins]
    fig = go.Figure(go.Bar(x=[f"{g:.2f} %".replace(".", ",") for g in gaps], y=[c for _, c in basins], marker_color=[C.COLORS["joint"] if g < 1e-6 else C.COLORS["fix"] for g in gaps], text=[c for _, c in basins], textposition="outside"))
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title="Endkosten: Lücke zum gemeinsamen Optimum")
    fig.update_yaxes(title="Zahl der Startzuordnungen", rangemode="tozero")
    return _base(fig, height=320, legend_below=False)


def build_compare(rows: list, method_stats, title: str = "Lücke zum gemeinsamen Optimum, Mittel") -> go.Figure:
    """Mittlere Lücke je Verfahren (Balken) und die größte (Raute)."""
    ks = list(C.METHODS)
    st = {k: method_stats(rows, k) for k in ks}
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[C.METHOD_NAMES[k] for k in ks], y=[st[k]["mean"] for k in ks], name="Mittel", marker_color=[C.COLORS["joint"] if st[k]["mean"] < 0.05 else C.COLORS["line"] for k in ks], text=[f"{st[k]['mean']:.2f}".replace(".", ",") for k in ks], textposition="outside"))
    fig.add_trace(go.Scatter(x=[C.METHOD_NAMES[k] for k in ks], y=[st[k]["max"] for k in ks], mode="markers", name="größte", marker=dict(symbol="diamond", size=10, color="white", line=dict(width=2, color="#333"))))
    fig.update_layout(title=title)
    fig.update_yaxes(title="Lücke in %", rangemode="tozero")
    fig.update_xaxes(tickangle=-35)
    return _base(fig, height=440)


def build_family_bars(rows: list, method_stats, title: str = "Lücke je Familie, Mittel") -> go.Figure:
    """Mittlere Lücke je Familie für vier Verfahren."""
    fig = go.Figure()
    for k, color in (("practice", "#9aa0a6"), ("centroid", C.COLORS["joint"]), ("alt_practice", C.COLORS["warn"]), ("order_first", C.COLORS["fix"])):
        ys = [method_stats([r for r in rows if r["kind"] == kd], k)["mean"] for kd in C.KINDS]
        fig.add_trace(go.Bar(x=[C.KIND_NAMES[kd] for kd in C.KINDS], y=ys, name=C.METHOD_NAMES[k], marker_color=color))
    fig.update_layout(title=title, barmode="group")
    fig.update_yaxes(title="Lücke in %", rangemode="tozero")
    return _base(fig, height=380)


def build_relax(rows: list, title: str = "Relaxation und Rundung") -> go.Figure:
    """Je Instanz die Lücke der unteren Schranke (x) gegen die Lücke der gerundeten Schwerpunkt-Zuordnung (y)."""
    xs = [100.0 * (r["joint"] - r["lb"]) / r["joint"] for r in rows]
    ys = [100.0 * (r["centroid"] - r["joint"]) / r["joint"] for r in rows]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=8, color=[C.COLORS["type"][C.KINDS.index(r["kind"])] for r in rows]), text=[f"{C.KIND_NAMES[r['kind']]}, P = {r['P']}, T = {r['T']}" for r in rows],
                               hovertemplate="%{text}<br>Schranke %{x:.2f} % unter dem Optimum<br>Rundung %{y:.2f} % darüber<extra></extra>"))
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title="Lücke der unteren Schranke in %")
    fig.update_yaxes(title="Lücke der gerundeten Lösung in %", rangemode="tozero")
    return _base(fig, height=360, legend_below=False)
