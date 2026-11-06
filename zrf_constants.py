"""Konstanten: Größen, Familien, Startzuordnungen, Verfahren, feste Reihen der Experimente, Farben und Presets."""

P_CHOICES = (6, 8, 10, 12)                            # Platzierungen
T_CHOICES = (3, 4, 5)                                 # Typen = Zuführplätze
DEFAULT_P, DEFAULT_T = 8, 4
HEAVY = (12, 5)                                       # ab dieser Größe nur auf Knopfdruck (2^12 Teilmengen mal 5! Zuordnungen)
SEED_MAX, DEFAULT_SEED = 999999, 46
FIXED_SEED0 = 100000                                  # feste Instanzen: Seeds ab 100000 (nie ein einzelner Seed als Befund)
SEEDS = 20                                            # Seeds je Familie und Größe
CONFIGS = ((8, 4), (10, 5), (12, 4))                  # (P, T) der festen Reihen: 3 Größen × 4 Familien × 20 Seeds = 240 Instanzen
RANDOM_STARTS = 3                                     # Zufallsstarts der wechselweisen Optimierung („beste von 3“)
MAX_ROUNDS = 50                                       # Runden der wechselweisen Optimierung (Sicherung)
EPS = 1e-9                                            # Kosten gelten als gleich, wenn sie sich um weniger unterscheiden
STUCK_EPS = 1e-6                                      # „über dem Optimum“: Lücke in Kosteneinheiten (Gleitkomma)

BOARD = (10.0, 6.0)                                   # Brett [0, 10] × [0, 6]; Zuführplätze auf der Geraden y = −2 zwischen x = 1 und x = 9
FEEDER_Y = -2.0
FEEDER_X = (1.0, 9.0)
HOME = (5.0, -2.0)                                    # Ruheposition des Kopfes

KINDS = ("random", "clustered", "skewed", "interleaved")
KIND_NAMES = {"random": "Zufällig", "clustered": "Cluster", "skewed": "Schief", "interleaved": "Verschränkt"}
KIND_HELP = {"random": "Orte gleichverteilt auf dem Brett, Typen zufällig (jeder Typ mindestens einmal).",
             "clustered": "Jeder Typ sammelt sich um einen eigenen Mittelpunkt (wie die Kondensatoren eines ICs).",
             "skewed": "Ein Typ hat etwa die Hälfte aller Platzierungen (wie die Kondensatoren), die übrigen wenige.",
             "interleaved": "Die Typen sind im Schachbrett über ein Gitter verteilt: nahe Orte haben verschiedene Typen."}
DEFAULT_KIND = "clustered"

STARTS = ("practice", "centroid", "random")
START_NAMES = {"practice": "Maschinenpraxis", "centroid": "Schwerpunkte (Ungarische Methode)", "random": "Zufall (Seed)"}
START_HELP = {"practice": "Häufigste Typen auf die mittleren Zuführplätze (eine Stand-in-Regel, keine Messung an einer Maschine).",
              "centroid": "Zuordnungsproblem auf der Summe der Wege Zuführplatz → Platzierungen des Typs, ohne Reihenfolge: die Relaxation.",
              "random": "Eine zufällige Zuordnung (SplitMix64 mit dem Seed der Instanz)."}
DEFAULT_START = "practice"

METHODS = ("practice", "centroid", "order_first", "alt_practice", "alt_centroid", "alt_random1", "alt_random3", "order_first_alt")
METHOD_NAMES = {"practice": "Maschinenpraxis, dann Reihenfolge", "centroid": "Schwerpunkt-Zuordnung, dann Reihenfolge", "order_first": "Erst die Reihenfolge, dann die Zuordnung",
                "alt_practice": "Wechselweise aus der Maschinenpraxis", "alt_centroid": "Wechselweise aus den Schwerpunkten", "alt_random1": "Wechselweise aus Zufall (ein Start)",
                "alt_random3": "Wechselweise, beste von 3 Zufallsstarts", "order_first_alt": "Erst die Reihenfolge, dann wechselweise"}
COLORS = {"best": "#2F6B65", "bad": "#b34a4a", "line": "#4c78a8", "warn": "#b08a2e", "joint": "#2F6B65", "fix": "#c0504d",
          "type": ["#4c78a8", "#b08a2e", "#2F6B65", "#8e6bb0", "#c0504d", "#9c755f"]}

HAND = {"pos": ((0, 0), (4, 4), (10, 0), (9, 3)), "typ": (0, 0, 1, 1), "T": 2, "names": "ABCD"}   # Handbeispiel: zwei Typen, vier Platzierungen

_BASE = {"kind": DEFAULT_KIND, "p": DEFAULT_P, "t": DEFAULT_T, "seed": DEFAULT_SEED, "start": DEFAULT_START}
PRESETS = {
    "Standardfall: Maschinenpraxis, Cluster": dict(_BASE),
    "Schwerpunkte als Start": dict(_BASE, start="centroid"),
    "Zufallsstart": dict(_BASE, start="random"),
    "Verschränkte Typen": dict(_BASE, kind="interleaved"),
    "Schiefe Typen": dict(_BASE, kind="skewed"),
    "Fünf Typen": dict(_BASE, p=10, t=5, kind="random"),
    "Zwölf Platzierungen": dict(_BASE, p=12),
    "Wenig Platzierungen": dict(_BASE, p=6, t=3),
}
PRESET_HELP = {
    "Standardfall: Maschinenpraxis, Cluster": "Acht Platzierungen, vier Typen: die wechselweise Optimierung startet bei der Maschinenpraxis und endet an einem Fixpunkt, der oft nicht das Optimum ist.",
    "Schwerpunkte als Start": "Dieselbe Instanz mit der Relaxation als Start: schon die Zuordnung der Schwerpunkte plus beste Reihenfolge ist (fast) das Optimum.",
    "Zufallsstart": "Ein zufälliger Start: die wechselweise Optimierung bleibt im Fangbecken des Starts hängen.",
    "Verschränkte Typen": "Die Typen sind im Schachbrett verteilt: die Maschinenpraxis und die wechselweise Optimierung sind hier am schlechtesten.",
    "Schiefe Typen": "Ein Typ hat etwa die Hälfte der Platzierungen: der Weg zu seinem Zuführplatz entscheidet.",
    "Fünf Typen": "Zehn Platzierungen und fünf Typen: 120 Zuordnungen, die Fangbecken werden zahlreicher.",
    "Zwölf Platzierungen": "Zwölf Platzierungen, vier Typen: die Held–Karp-Rechnung über 4 096 Teilmengen je Zuordnung (etwa zwei Sekunden).",
    "Wenig Platzierungen": "Sechs Platzierungen, drei Typen: so klein, dass man alle 6 Zuordnungen von Hand nachprüfen kann.",
}


def n_text(n: int) -> str:
    """Anzahl mit Leerzeichen als Tausendertrenner: 362880 → „362 880“."""
    return f"{n:,}".replace(",", " ")


assert set(PRESETS) == set(PRESET_HELP) and set(KIND_NAMES) == set(KINDS) == set(KIND_HELP) and set(START_NAMES) == set(STARTS) == set(START_HELP) and set(METHOD_NAMES) == set(METHODS)
assert all(p["kind"] in KINDS and p["p"] in P_CHOICES and p["t"] in T_CHOICES and p["p"] >= p["t"] and p["start"] in STARTS and 0 <= p["seed"] <= SEED_MAX for p in PRESETS.values())
assert all(p >= t for p, t in CONFIGS)
