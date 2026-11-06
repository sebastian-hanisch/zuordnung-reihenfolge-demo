"""Bestückung mit einem Kopf: Zuordnung (Typ → Zuführplatz) und Reihenfolge (der Platzierungen), exakt und wechselweise. Eigenes Modell (wie der Abschnitt „Bestückung“ der Leiterplatte-Demo, hier mit einer Düse); die Literatur ist **nicht eingesehen**.

`P` Platzierungen (Ort `p_i`, Typ `t_i`), `T` Typen auf `T` Zuführplätzen `s_m` (auf einer Geraden unterhalb des Bretts); der Kopf holt je Teil eines vom Zuführplatz seines Typs und setzt es; Tschebyscheff-Fahrstrecke, Start und Ende an der Ruheposition `h`.
    Kosten(σ, o) = d(h, s_σ(t_o1)) + d(s_σ(t_o1), p_o1) + Σ_k [ d(p_ok, s_σ(t_ok+1)) + d(s_σ(t_ok+1), p_ok+1) ] + d(p_oP, h),
`σ` ist eine Permutation (Typ → Zuführplatz), `o` eine Permutation der Platzierungen.
**Bei fester Reihenfolge** ist die Zuordnung linear: Kosten = Σ_{t,m} c[t][m] · x[t][m] mit der Summe der Wege über den Zuführplatz `m` für alle Platzierungen des Typs `t` (Zuordnungsproblem, Ungarische Methode, exakt).
**Bei fester Zuordnung** ist die Reihenfolge ein asymmetrisches Handlungsreisendenproblem: exakt durch Held–Karp über Teilmengen (`P ≤ 12`).
**Gemeinsam exakt:** alle `T!` Zuordnungen, je mit Held–Karp. **Wechselweise:** die Reihenfolge exakt bei festem σ, dann σ exakt bei fester Reihenfolge, bis die Kosten nicht mehr sinken (beide Schritte exakt: der günstigste Fall für das Verfahren).
**Relaxation:** das Zuordnungsproblem mit c[t][m] = 2 · Σ_{i ∈ t} d(s_m, p_i) (Schwerpunkte: nur die Wege Zuführplatz → Platzierungen, ohne Reihenfolge) und eine untere Schranke, die die Kopplung der Reihenfolge weglässt."""

from __future__ import annotations

import itertools
import math

import numpy as np

import zrf_constants as C
from zrf_assignment import hungarian
from zrf_rng import SplitMix64


def cheb(a, b) -> np.ndarray:
    """Tschebyscheff-Abstände zwischen den Zeilen von `a` und `b` (Matrix a × b)."""
    return np.max(np.abs(np.asarray(a, dtype=float)[:, None, :] - np.asarray(b, dtype=float)[None, :, :]), axis=-1)


class Inst:
    """Eine Instanz: Orte, Typen und die Abstände zu den Zuführplätzen."""

    def __init__(self, pos, typ, T: int):
        self.pos = np.asarray(pos, dtype=float)
        self.typ = np.asarray(typ, dtype=int)
        self.P, self.T = len(self.pos), int(T)
        self.slots = np.array([[C.FEEDER_X[0] + (C.FEEDER_X[1] - C.FEEDER_X[0]) * m / max(T - 1, 1), C.FEEDER_Y] for m in range(T)])
        self.home = np.array(C.HOME)
        self.dps = cheb(self.pos, self.slots)                    # P × T: Platzierung → Zuführplatz
        self.dpp = cheb(self.pos, self.pos)
        self.dhs = cheb(self.home[None, :], self.slots)[0]       # T: Ruheposition → Zuführplatz
        self.dhp = cheb(self.home[None, :], self.pos)[0]         # P: Ruheposition → Platzierung (= zurück)
        self._best_order: dict = {}                              # Held–Karp je Zuordnung, einmal gerechnet


def _normal(rng: SplitMix64) -> float:
    """Normalverteilte Zahl (Box–Muller) aus SplitMix64."""
    return math.sqrt(-2.0 * math.log(1.0 - rng.random())) * math.cos(2.0 * math.pi * rng.random())


def make(kind: str, P: int, T: int, seed: int) -> Inst:
    """Instanz der Familie `kind` mit `P` Platzierungen und `T` Typen (jeder Typ mindestens einmal); alle Zufallszahlen aus SplitMix64."""
    if kind not in C.KINDS:
        raise ValueError(f"unbekannte Familie {kind!r}")
    if not (T >= 1 and P >= T):
        raise ValueError("P muss mindestens T sein")
    rng = SplitMix64(seed * 1009 + P * 31 + T)
    W, H = C.BOARD
    uni = lambda: (rng.random() * W, rng.random() * H)
    if kind == "random":
        pos = [uni() for _ in range(P)]
        typ = list(range(T)) + [rng.below(T) for _ in range(P - T)]
    elif kind == "clustered":
        centres = [(1.0 + rng.random() * (W - 2.0), 1.0 + rng.random() * (H - 2.0)) for _ in range(T)]
        typ = list(range(T)) + [rng.below(T) for _ in range(P - T)]
        pos = [(min(max(centres[t][0] + 0.8 * _normal(rng), 0.0), W), min(max(centres[t][1] + 0.8 * _normal(rng), 0.0), H)) for t in typ]
    elif kind == "skewed":
        extra = P - T
        typ = list(range(T)) + [0] * (extra // 2) + [1 + rng.below(max(T - 1, 1)) if T > 1 else 0 for _ in range(extra - extra // 2)]
        pos = [uni() for _ in range(P)]
    else:                                                              # interleaved
        side = math.ceil(math.sqrt(P))
        cells = [(r, c) for r in range(side) for c in range(side)][:P]
        pos = [(0.5 + 9.0 * c / max(side - 1, 1) + 0.15 * _normal(rng), 0.5 + 5.0 * r / max(side - 1, 1) + 0.15 * _normal(rng)) for r, c in cells]
        typ = [(r + c) % T for r, c in cells]
        typ[:T] = list(range(T))
    idx = list(range(P))
    rng.shuffle(idx)
    return Inst([pos[i] for i in idx], [typ[i] for i in idx], T)


def hand_instance() -> Inst:
    """Das Handbeispiel: zwei Typen, vier Platzierungen A (0,0), B (4,4) vom Typ 0, C (10,0), D (9,3) vom Typ 1."""
    return Inst(C.HAND["pos"], C.HAND["typ"], C.HAND["T"])


# ----------------------------------------------------------------------------- Kosten

def tour_cost(inst: Inst, sigma, order) -> float:
    """Kosten der Fahrt: Ruheposition → Zuführplatz → Platzierung → … → Ruheposition (einzeln gerechnet, ohne Matrizen)."""
    S = inst.slots[np.asarray(sigma)]
    prev = inst.home
    total = 0.0
    for i in order:
        s = S[inst.typ[i]]
        total += max(abs(prev[0] - s[0]), abs(prev[1] - s[1])) + max(abs(s[0] - inst.pos[i, 0]), abs(s[1] - inst.pos[i, 1]))
        prev = inst.pos[i]
    return float(total + max(abs(prev[0] - inst.home[0]), abs(prev[1] - inst.home[1])))


def legs(inst: Inst, sigma, order) -> list:
    """Die einzelnen Wege der Fahrt als (von, nach, Länge) mit Beschriftungen: ‚Ruhe‘, ‚Zuführplatz m‘, ‚Platzierung i‘."""
    S = inst.slots[np.asarray(sigma)]
    prev, prev_name = inst.home, "Ruhe"
    out = []
    for i in order:
        m = int(np.asarray(sigma)[inst.typ[i]])
        s = S[inst.typ[i]]
        out.append((prev_name, f"Zuführplatz {m}", float(max(abs(prev[0] - s[0]), abs(prev[1] - s[1])))))
        out.append((f"Zuführplatz {m}", f"Platzierung {i}", float(max(abs(s[0] - inst.pos[i, 0]), abs(s[1] - inst.pos[i, 1])))))
        prev, prev_name = inst.pos[i], f"Platzierung {i}"
    out.append((prev_name, "Ruhe", float(max(abs(prev[0] - inst.home[0]), abs(prev[1] - inst.home[1])))))
    return out


# ----------------------------------------------------------------------------- feste Zuordnung: Reihenfolge (Held–Karp)

def _first_min(x: np.ndarray) -> int:
    """Index des ersten Eintrags, der höchstens `EPS` über dem Minimum liegt. Gleichstände (in Tschebyscheff-Abständen häufig) werden so auf jedem Rechner gleich aufgelöst; ein reines `argmin` folgt dem Gleitkommarauschen (Linux und Windows wählten verschiedene gleich gute Reihenfolgen)."""
    return int(np.flatnonzero(x <= x.min() + C.EPS)[0])


def _held_karp(P: int, w: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple:
    """Kürzester Weg über alle Knoten: Kosten start[j] für den ersten, w[i, j] von i nach j, end[i] am Ende. Held–Karp über Teilmengen. Rückgabe (Kosten, Reihenfolge)."""
    INF = float("inf")
    full = (1 << P) - 1
    dp = np.full((1 << P, P), INF)
    par = np.full((1 << P, P), -1, dtype=np.int8)
    for j in range(P):
        dp[1 << j, j] = start[j]
    for S in range(1, 1 << P):
        row = dp[S]
        if not np.isfinite(row).any():
            continue
        for j in range(P):
            if S >> j & 1:
                continue
            cand = row + w[:, j]
            i = _first_min(cand)
            v = cand[i]
            T_ = S | (1 << j)
            if v < dp[T_, j] - C.EPS:
                dp[T_, j] = v
                par[T_, j] = i
    fin = dp[full] + end
    last = _first_min(fin)
    order, S, j = [], full, last
    while j != -1:
        order.append(j)
        pj = int(par[S, j])
        S ^= 1 << j
        j = pj
    return float(fin[last]), order[::-1]


def best_order(inst: Inst, sigma) -> tuple:
    """Beste Reihenfolge bei fester Zuordnung (exakt). Rückgabe (Kosten, Reihenfolge)."""
    key = tuple(int(m) for m in sigma)
    hit = inst._best_order.get(key)                                # dieselbe Zuordnung kommt in der Aufzählung, den Fangbecken und der wechselweisen Optimierung immer wieder vor
    if hit is None:
        P = inst.P
        slot_of = np.asarray(key)[inst.typ]                        # Zuführplatz je Platzierung
        pick = inst.dps[np.arange(P), slot_of]                     # Zuführplatz → eigene Platzierung
        w = inst.dps[:, slot_of] + pick[None, :]                   # Platzierung i → Zuführplatz von j → Platzierung j
        hit = inst._best_order[key] = _held_karp(P, w, inst.dhs[slot_of] + pick, inst.dhp)
    return hit[0], list(hit[1])


def order_through_placements(inst: Inst) -> tuple:
    """Kürzeste Fahrt durch die Platzierungen ohne Zuführplätze (Ruheposition → alle Orte → Ruheposition), exakt. Rückgabe (Kosten, Reihenfolge)."""
    return _held_karp(inst.P, inst.dpp, inst.dhp, inst.dhp)


# ----------------------------------------------------------------------------- feste Reihenfolge: Zuordnung (linear)

def order_costs(inst: Inst, order) -> np.ndarray:
    """c[t][m]: Summe der Wege über den Zuführplatz m für alle Platzierungen des Typs t bei der Reihenfolge `order` (ohne den Rückweg zur Ruheposition)."""
    c = np.zeros((inst.T, inst.T))
    prev = None
    for i in order:
        pre = inst.dhs if prev is None else inst.dps[prev]
        c[inst.typ[i]] += pre + inst.dps[i]
        prev = i
    return c


def best_assignment(inst: Inst, order) -> tuple:
    """Beste Zuordnung bei fester Reihenfolge: Zuordnungsproblem, exakt. Rückgabe (Kosten, σ)."""
    c = np.round(order_costs(inst, order), 9)                       # auf 9 Stellen gerundet: Gleichstände der Zuordnung hängen nicht am Gleitkommarauschen
    assign, total = hungarian(c)
    return float(total + inst.dhp[order[-1]]), np.array([int(a) for a in assign])


# ----------------------------------------------------------------------------- gemeinsam und wechselweise

def joint_exact(inst: Inst) -> tuple:
    """Gemeinsames Optimum: alle T! Zuordnungen, je mit Held–Karp. Rückgabe (Kosten, σ, Reihenfolge); bei Gleichstand die erste Zuordnung in lexikographischer Reihenfolge."""
    best = (float("inf"), None, None)
    for sigma in itertools.permutations(range(inst.T)):
        c, o = best_order(inst, sigma)
        if c < best[0] - C.EPS:
            best = (c, np.array(sigma), o)
    return best


def alternating(inst: Inst, sigma0, max_rounds: int = C.MAX_ROUNDS) -> dict:
    """Wechselweise: Reihenfolge exakt bei festem σ, dann σ exakt bei fester Reihenfolge, solange die Kosten sinken.

    Rückgabe: Kosten, σ, Reihenfolge, Runden und der Verlauf als Liste (Runde, Kosten nach dem Reihenfolgeschritt, Kosten nach dem Zuordnungsschritt)."""
    sigma = np.asarray(sigma0)
    cost, order, trace = float("inf"), None, []
    for rounds in range(1, max_rounds + 1):
        c1, order = best_order(inst, sigma)
        c2, sig2 = best_assignment(inst, order)
        trace.append((rounds, c1, c2))
        if c2 >= cost - C.EPS:
            return {"cost": c1, "sigma": sigma, "order": order, "rounds": rounds, "trace": trace}      # Fixpunkt: weder die Reihenfolge noch die Zuordnung verbessert ihn (c1 = Kosten von σ mit der besten Reihenfolge)
        cost, sigma = c2, sig2
    return {"cost": cost, "sigma": sigma, "order": order, "rounds": max_rounds, "trace": trace}


def sigma_practice(inst: Inst) -> np.ndarray:
    """Maschinenpraxis (Stand-in-Regel): die häufigsten Typen auf die Zuführplätze nahe der Ruheposition."""
    counts = np.bincount(inst.typ, minlength=inst.T)
    slot_rank = np.argsort(inst.dhs, kind="stable")
    sigma = np.empty(inst.T, dtype=int)
    for rank, t in enumerate(np.argsort(-counts, kind="stable")):
        sigma[t] = slot_rank[rank]
    return sigma


def sigma_centroid(inst: Inst) -> np.ndarray:
    """Zuordnung der Schwerpunkte: Zuordnungsproblem auf c[t][m] = 2 · Σ_{i ∈ t} d(s_m, p_i) (die Relaxation ohne Reihenfolge)."""
    c = np.zeros((inst.T, inst.T))
    for t in range(inst.T):
        c[t] = 2.0 * inst.dps[inst.typ == t].sum(axis=0)
    assign, _ = hungarian(np.round(c, 9))
    return np.array([int(a) for a in assign])


def sigma_random(inst: Inst, seed: int) -> np.ndarray:
    """Zufällige Zuordnung (SplitMix64)."""
    rng = SplitMix64(seed * 7919 + inst.T)
    sigma = list(range(inst.T))
    rng.shuffle(sigma)
    return np.array(sigma)


def lower_bound(inst: Inst) -> float:
    """Untere Schranke durch Relaxation: die Beine Zuführplatz → Platzierung allein (Zuordnungsproblem, c[t][m] = Σ_{i ∈ t} d(s_m, p_i)), dazu für jede Platzierung bis auf eine das kürzeste Bein zu irgendeinem
    Zuführplatz (die Beine zur nächsten Abholung sind nie kürzer), den kürzesten Weg von der Ruheposition zu einem Zuführplatz und den kürzesten Rückweg."""
    c = np.zeros((inst.T, inst.T))
    for t in range(inst.T):
        c[t] = inst.dps[inst.typ == t].sum(axis=0)
    _, first = hungarian(np.round(c, 9))
    back = float(inst.dps.min(axis=1).sum() - inst.dps.min(axis=1).max())
    return float(first + back + inst.dhs.min() + inst.dhp.min())


def basins(inst: Inst) -> dict:
    """Fangbecken: die Endkosten der wechselweisen Optimierung von jeder der T! Startzuordnungen. Rückgabe {gerundete Kosten: Zahl der Startzuordnungen}."""
    out: dict = {}
    for s in itertools.permutations(range(inst.T)):
        k = round(alternating(inst, s)["cost"], 6)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
