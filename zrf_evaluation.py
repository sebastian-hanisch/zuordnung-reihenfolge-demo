"""Eine Instanz mit allen Verfahren, Vergleiche über feste Instanzen und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Alle Aussagen gelten für **eine Düse**, die vier erfundenen Familien, die Tschebyscheff-Fahrstrecke mit Zuführplätzen auf einer Geraden und die kleinen Größen (bis 12 Platzierungen, 3 bis 5 Typen), für die das **gemeinsame Optimum exakt** gerechnet wird.
**Lücke** = (Kosten − gemeinsames Optimum) / gemeinsames Optimum in Prozent; „hängt“ heißt: Kosten über dem Optimum um mehr als `STUCK_EPS`. Die Maschinenpraxis ist eine **Stand-in-Regel**, keine Messung an einer Maschine.
Kosten sind Gleitkommazahlen: Gleichstände zwischen Zuordnungen ändern die Kosten nicht. Zeiten stehen nie in Verdikten."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import zrf_constants as C
import zrf_model as M


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    p: int = C.DEFAULT_P
    t: int = C.DEFAULT_T
    seed: int = C.DEFAULT_SEED
    start: str = C.DEFAULT_START


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    return de(x, digits) + " %"


def num(x: float) -> str:
    """Ganze Zahl ohne Nachkommastelle, sonst zwei Stellen: 34 / 74,09."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else de(x, 2)


def count(x: float) -> str:
    return C.n_text(int(round(x)))


def kind_name(k: str) -> str:
    return C.KIND_NAMES.get(k, k)


def gap(cost: float, joint: float) -> float:
    """Lücke zum gemeinsamen Optimum in Prozent; Unterschiede unter `STUCK_EPS` Kosteneinheiten werden zu 0."""
    if abs(cost - joint) < C.STUCK_EPS:
        return 0.0
    return 100.0 * (cost - joint) / joint if joint else float("nan")


def stuck(cost: float, joint: float) -> bool:
    return cost - joint > C.STUCK_EPS


def sigma_text(sigma) -> str:
    """Zuordnung als Text: „Typ 0 → Platz 1, Typ 1 → Platz 0“."""
    return ", ".join(f"Typ {t} → Platz {int(m)}" for t, m in enumerate(sigma))


# ----------------------------------------------------------------------------- eine Instanz

def run_methods(inst: M.Inst, seed: int) -> dict:
    """Alle Verfahren auf einer Instanz: Kosten, Zuordnung und Reihenfolge je Verfahren; das gemeinsame Optimum, die untere Schranke und die Fangbecken."""
    joint, sig_j, ord_j = M.joint_exact(inst)
    out = {"joint": joint, "sigma_joint": sig_j, "order_joint": ord_j, "lb": M.lower_bound(inst), "assign_at_opt_order": M.best_assignment(inst, ord_j)[0]}
    sp, sc = M.sigma_practice(inst), M.sigma_centroid(inst)
    out["practice"], out["order_practice"] = M.best_order(inst, sp)
    out["centroid"], out["order_centroid"] = M.best_order(inst, sc)
    out["sigma_practice"], out["sigma_centroid"] = sp, sc
    ap, ac = M.alternating(inst, sp), M.alternating(inst, sc)
    out["alt_practice"], out["alt_centroid"] = ap["cost"], ac["cost"]
    out["rounds_practice"], out["rounds_centroid"] = ap["rounds"], ac["rounds"]
    randoms = [M.alternating(inst, M.sigma_random(inst, seed + k)) for k in range(C.RANDOM_STARTS)]
    out["alt_random1"] = randoms[0]["cost"]
    out["alt_random3"] = min(r["cost"] for r in randoms)
    out["rounds_random"] = [r["rounds"] for r in randoms]
    t_cost, t_order = M.order_through_placements(inst)
    c1, sig_o = M.best_assignment(inst, t_order)
    out["order_first"] = c1
    out["order_first_alt"] = M.alternating(inst, sig_o)["cost"]
    out["basins"] = [(k, c) for k, c in M.basins(inst).items()]
    out["alt"] = {"practice": ap, "centroid": ac, "random": randoms[0]}
    return out


def instance_data(s: Settings) -> dict:
    inst = M.make(s.kind, s.p, s.t, s.seed)
    return {"inst": inst, **run_methods(inst, s.seed)}


def hand_data() -> dict:
    """Das Handbeispiel: beide Fixpunkte der wechselweisen Optimierung, einzeln nachgerechnet."""
    inst = M.hand_instance()
    joint, sig, order = M.joint_exact(inst)
    out = {"inst": inst, "joint": joint, "sigma_joint": sig, "order_joint": order, "tour_joint": M.legs(inst, sig, order)}
    for key, s0 in (("good", (0, 1)), ("bad", (1, 0))):
        a = M.alternating(inst, s0)
        out[key] = {**a, "legs": M.legs(inst, a["sigma"], a["order"]), "check": M.tour_cost(inst, a["sigma"], a["order"])}
    out["lb"] = M.lower_bound(inst)
    return out


# ----------------------------------------------------------------------------- Zeilen der Vergleiche

ROW_KEYS = ("joint", "practice", "centroid", "alt_practice", "alt_centroid", "alt_random1", "alt_random3", "order_first", "order_first_alt", "lb", "assign_at_opt_order", "rounds_practice", "rounds_centroid", "rounds_random", "basins")


def row(kind: str, P: int, T: int, seed: int) -> dict:
    """Eine Instanz der festen Reihen (ohne Matrizen und Verläufe)."""
    r = run_methods(M.make(kind, P, T, seed), seed)
    return {"kind": kind, "P": P, "T": T, "seed": seed, **{k: r[k] for k in ROW_KEYS}, "sigma_joint": [int(x) for x in r["sigma_joint"]]}


def jobs() -> list:
    return [(k, P, T, C.FIXED_SEED0 + s) for (P, T) in C.CONFIGS for k in C.KINDS for s in range(C.SEEDS)]


def sweep(kind: str, P: int, T: int) -> list:
    return [row(*job) for job in jobs() if job[0] == kind and job[1] == P and job[2] == T]


# ----------------------------------------------------------------------------- Auswertungen

def method_stats(rows: list, key: str) -> dict:
    """Lücke zum gemeinsamen Optimum (Mittel, Median, größte in Prozent) und Zahl der Instanzen über dem Optimum."""
    g = [gap(r[key], r["joint"]) for r in rows]
    return {"count": len(rows), "mean": float(np.mean(g)), "median": float(np.median(g)), "max": float(max(g)), "stuck": sum(stuck(r[key], r["joint"]) for r in rows)}


def lb_stats(rows: list) -> dict:
    """Lücke der unteren Schranke zum Optimum in Prozent (Mittel, Median, größte)."""
    g = [100.0 * (r["joint"] - r["lb"]) / r["joint"] for r in rows]
    return {"mean": float(np.mean(g)), "median": float(np.median(g)), "max": float(max(g))}


def basin_stats(rows: list) -> dict:
    """Fangbecken: Zahl der verschiedenen Endkosten je Instanz (Median, größte), Anteil der Startzuordnungen, die zum Optimum führen (Median, kleinster), Instanzen, in denen alle zum Optimum führen."""
    ends = [len(r["basins"]) for r in rows]
    share = []
    for r in rows:
        total = sum(c for _, c in r["basins"])
        share.append(sum(c for k, c in r["basins"] if k - r["joint"] <= C.STUCK_EPS) / total)
    return {"count": len(rows), "ends_median": float(np.median(ends)), "ends_max": int(max(ends)), "share_median": float(np.median(share)), "share_min": float(min(share)), "all_to_optimum": sum(len(r["basins"]) == 1 and r["basins"][0][0] - r["joint"] <= C.STUCK_EPS for r in rows)}


def assign_at_optimum(rows: list) -> int:
    """Zahl der Instanzen, in denen die beste Zuordnung bei der optimalen Reihenfolge das Optimum ist."""
    return sum(abs(r["assign_at_opt_order"] - r["joint"]) < C.STUCK_EPS for r in rows)


def select(rows: list, kind=None, config=None) -> list:
    return [r for r in rows if (kind is None or r["kind"] == kind) and (config is None or (r["P"], r["T"]) == config)]


# ----------------------------------------------------------------------------- Verdikte

def _head(s: Settings, v: dict) -> str:
    return f"Familie **{kind_name(s.kind)}**, {s.p} Platzierungen, {s.t} Typen, Seed {s.seed}"


def verdict_core(s: Settings, v: dict):
    """Die Kernfrage: was kostet es, Zuordnung und Reihenfolge zu trennen?"""
    j = v["joint"]
    a = v["alt"][s.start]
    g_alt = gap(a["cost"], j)
    g_c = gap(v["centroid"], j)
    g_p = gap(v["practice"], j)
    g_of = gap(v["order_first"], j)
    hangs = stuck(a["cost"], j)
    kind = "info" if hangs else "success"
    start = f"Wechselweise Optimierung, Start {C.START_NAMES[s.start]}: " + (f"**hängt** nach {a['rounds']} Runden an einem Fixpunkt mit Kosten {num(a['cost'])} ({pct(g_alt, 2)} über dem Optimum)" if hangs else f"erreicht nach {a['rounds']} Runden das Optimum")
    return kind, (f"{_head(s, v)}: gemeinsames Optimum **{num(j)}** (alle {_fact(s.t)} Zuordnungen mit je exakter Reihenfolge). {start}. "
                  f"Getrennt: Maschinenpraxis, dann Reihenfolge {num(v['practice'])} ({pct(g_p, 2)}); **Schwerpunkt-Zuordnung**, dann Reihenfolge {num(v['centroid'])} ({pct(g_c, 2)}); erst die Reihenfolge, dann die Zuordnung {num(v['order_first'])} ({pct(g_of, 2)}). "
                  f"Von den {_fact(s.t)} Startzuordnungen führen {_reach(v).split(' ')[0]} zum Optimum. Eine Instanz, kein Befund über die Familie.")


def _fact(t: int) -> int:
    import math

    return math.factorial(t)


def _reach(v: dict) -> str:
    total = sum(c for _, c in v["basins"])
    ok = sum(c for k, c in v["basins"] if k - v["joint"] <= C.STUCK_EPS)
    return f"{ok} von {total}"


def verdict_hand(h: dict):
    g, b = h["good"], h["bad"]
    return "info", (f"Zwei Typen, vier Platzierungen (A (0, 0), B (4, 4) vom Typ 0; C (10, 0), D (9, 3) vom Typ 1), Zuführplätze bei x = 1 und x = 9, Ruheposition (5, −2): gemeinsames Optimum **{num(h['joint'])}** mit der Zuordnung {sigma_text(h['sigma_joint'])} und der Reihenfolge "
                    f"{''.join(C.HAND['names'][i] for i in h['order_joint'])}. Von der vertauschten Zuordnung aus endet die wechselweise Optimierung nach {b['rounds']} Runden bei **{num(b['cost'])}** (Reihenfolge {''.join(C.HAND['names'][i] for i in b['order'])}, {sigma_text(b['sigma'])}): "
                    f"ein Fixpunkt, {pct(gap(b['cost'], h['joint']), 0)} über dem Optimum; weder die Reihenfolge noch die Zuordnung verbessert ihn. Untere Schranke der Relaxation {de(h['lb'], 1)}.")


def verdict_alternating(s: Settings, v: dict):
    a = v["alt"][s.start]
    steps = " → ".join(num(c2) for _, _, c2 in a["trace"])
    j = v["joint"]
    return "info", (f"{_head(s, v)}, Start {C.START_NAMES[s.start]}: Kosten nach den Runden {steps}; Fixpunkt {num(a['cost'])} gegen das Optimum {num(j)} ({pct(gap(a['cost'], j), 2)}). "
                    + ("Der Fixpunkt ist **nicht** das Optimum: ein lokales Optimum beider Schritte." if stuck(a["cost"], j) else "Der Fixpunkt ist das gemeinsame Optimum."))


def verdict_basins(s: Settings, v: dict):
    total = sum(c for _, c in v["basins"])
    ends = len(v["basins"])
    best = sorted(v["basins"], key=lambda kc: -kc[1])[0]
    return "info", (f"{_head(s, v)}: Die {total} Startzuordnungen laufen zu **{ends}** verschiedenen Endkosten; {_reach(v)} führen zum Optimum {num(v['joint'])}. Die häufigsten Endkosten sind {num(best[0])} ({best[1]} Startzuordnungen, {pct(gap(best[0], v['joint']), 2)} über dem Optimum). "
                    + ("Alle Starts führen zum Optimum." if ends == 1 and not stuck(v["basins"][0][0], v["joint"]) else "Der Start entscheidet, wohin die wechselweise Optimierung läuft."))


def verdict_relax(s: Settings, v: dict):
    j = v["joint"]
    return "info", (f"{_head(s, v)}: Die untere Schranke der Relaxation ist {de(v['lb'], 2)} ({pct(100.0 * (j - v['lb']) / j, 2)} unter dem Optimum {num(j)}); die gerundete Lösung, die Schwerpunkt-Zuordnung mit der besten Reihenfolge, kostet {num(v['centroid'])} ({pct(gap(v['centroid'], j), 2)}). "
                    f"Die beste Zuordnung bei der **optimalen** Reihenfolge kostet {num(v['assign_at_opt_order'])}: " + ("das Optimum (das folgt aus der Definition des Optimums und prüft nur den linearen Schritt: er verschenkt nichts)." if abs(v["assign_at_opt_order"] - j) < C.STUCK_EPS else "**nicht** das Optimum."))


def verdict_compare(rows: list):
    """Die Trennungen im Vergleich über feste Instanzen."""
    parts = []
    for key in C.METHODS:
        st = method_stats(rows, key)
        parts.append(f"{C.METHOD_NAMES[key]} {pct(st['mean'], 2)} ({st['stuck']} von {st['count']} über dem Optimum, größte {pct(st['max'], 2)})")
    lb = lb_stats(rows)
    return "info", (f"Lücke zum gemeinsamen Optimum im Mittel über {len(rows)} Instanzen: " + "; ".join(parts) + f". Untere Schranke der Relaxation {pct(lb['mean'], 2)} unter dem Optimum (größte {pct(lb['max'], 2)}). "
                    f"Gegenprobe des linearen Schritts: die beste Zuordnung bei der optimalen Reihenfolge ist das Optimum in {assign_at_optimum(rows)} von {len(rows)} Instanzen (folgt aus der Definition, kein Befund).")


def verdict_family(rows: list, kind: str):
    rs = select(rows, kind=kind)
    cells = ", ".join(f"{C.METHOD_NAMES[k]} {pct(method_stats(rs, k)['mean'], 2)}" for k in ("practice", "centroid", "alt_practice", "alt_random1", "order_first"))
    return "info", f"Familie {kind_name(kind)} ({len(rs)} Instanzen), Lücke im Mittel: {cells}."


def verdict_basins_fixed(rows: list):
    b = basin_stats(rows)
    return "info", (f"Fangbecken über {b['count']} Instanzen: Die Startzuordnungen laufen im Median zu {num(b['ends_median'])} verschiedenen Endkosten (größte Zahl {b['ends_max']}); im Median führen {pct(100 * b['share_median'], 0)} der Startzuordnungen zum Optimum (kleinster Anteil {pct(100 * b['share_min'], 0)}); "
                    f"in {b['all_to_optimum']} von {b['count']} Instanzen führen alle zum Optimum.")
