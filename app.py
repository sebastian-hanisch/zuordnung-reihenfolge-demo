"""Zuordnung und Reihenfolge - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 14 (gelb) der Reihe „Anordnen: relaxieren und runden“: beim Bestücken mit einem Kopf hängen zwei Entscheidungen zusammen, welcher Typ an welchen Zuführplatz kommt und in welcher Reihenfolge der Kopf die Platzierungen anfährt.
Eigene Fassung: bei fester Reihenfolge ist die Zuordnung linear (Ungarische Methode), bei fester Zuordnung die Reihenfolge ein Rundreiseproblem (Held–Karp). Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import zrf_constants as C
from zrf_evaluation import (
    Settings,
    assign_at_optimum,
    de,
    gap,
    hand_data,
    instance_data,
    jobs,
    kind_name,
    lb_stats,
    method_stats,
    num,
    pct,
    row,
    sigma_text,
    stuck,
    verdict_alternating,
    verdict_basins,
    verdict_basins_fixed,
    verdict_compare,
    verdict_core,
    verdict_family,
    verdict_hand,
    verdict_relax,
)
from zrf_layout import dunkles_design, kopf_css
from zrf_pdf_export import build_pdf
from zrf_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from zrf_visualization import build_basins, build_compare, build_family_bars, build_instance, build_relax, build_rounds

st.set_page_config(page_title="Zuordnung und Reihenfolge – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne alle Zuordnungen …")
def _instance(settings: Settings):
    return instance_data(settings)


@st.cache_data(show_spinner=False)
def _hand():
    return hand_data()


@st.cache_data(show_spinner=False)
def _row(kind, p, t, seed):
    return row(kind, p, t, seed)


def _rows():
    return [_row(*job) for job in jobs()]


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in zrf_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔀 Zuordnung und Reihenfolge")
st.markdown(
    """
**Welcher Typ an welchen Zuführplatz, und in welcher Reihenfolge?** Ein Bestückkopf holt jedes Bauteil an einem **Zuführplatz** (Gurt) ab und setzt es auf seine **Platzierung**. Zwei Entscheidungen hängen zusammen:
die **Zuordnung** (welcher Bauteiltyp an welchen Zuführplatz) und die **Reihenfolge** der Platzierungen. Bei fester Reihenfolge ist die Zuordnung ein **lineares Zuordnungsproblem** (Ungarische Methode), bei fester Zuordnung die Reihenfolge ein **Rundreiseproblem** (klein genug für Held–Karp).
Man kann also **wechselweise** optimieren – und bleibt an einem Fixpunkt hängen, der nicht das Optimum ist. Wie oft, wie weit daneben und was die Relaxation (Zuordnung nach den Schwerpunkten der Wege) daran ändert, zeigen die Experimente.
Alle Familien sind **eigene Erfindungen**, die Maschinenpraxis ist eine **Stand-in-Regel**; Stück 14 der Reihe „Anordnen: relaxieren und runden“ (gelb, optional).

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Die Maschine:** ein Kopf, der immer nur ein Bauteil trägt (**Tschebyscheff-Abstand**: x und y fahren gleichzeitig). Er startet in der **Ruheposition**, holt am Zuführplatz, setzt auf die Platzierung, holt das nächste … und fährt zurück. Die **Zuführplätze** liegen auf einer Geraden unter dem Brett.
2. **Die Entscheidungen:** `σ` ordnet jedem Typ einen Zuführplatz zu (`T` Typen, `T` Plätze, also `T!` Möglichkeiten); `o` ist die Reihenfolge der `P` Platzierungen. Die Kosten sind die Länge der ganzen Fahrt.
3. **Gemeinsames Optimum:** alle `T!` Zuordnungen, je mit der exakten Reihenfolge (Held–Karp). Das ist die Messlatte; jede „Lücke“ in der App ist die Abweichung davon in Prozent.
4. **Wechselweise Optimierung:** Reihenfolge exakt bei festem `σ`, dann `σ` exakt (Ungarische Methode) bei fester Reihenfolge, bis nichts mehr besser wird. Das Ergebnis ist ein **Fixpunkt** beider Schritte – nicht unbedingt das Optimum.
5. **Relaxation und Rundung:** die Zuordnung nach den **Schwerpunkten** (Summe der Wege Zuführplatz → Platzierungen des Typs, ohne Reihenfolge) ist ein Zuordnungsproblem; ihr Wert ist die untere Schranke, ihre Lösung die gerundete Zuordnung.
6. **Feste Instanzen:** Vergleiche laufen über 240 Instanzen mit festen Seeds ab 100000 (vier Familien × drei Größen × 20 Seeds), nie über einen einzelnen Lauf.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Instanz**")
    kind = st.selectbox("Familie der Platzierungen", C.KINDS, key="kind_select", format_func=kind_name, help="Vier erfundene Familien. Die Beschreibung steht unter dem Regler.")
    st.caption(C.KIND_HELP[kind])
    p = st.select_slider("Platzierungen P", options=C.P_CHOICES, key="p_select", help="Wie viele Bauteile gesetzt werden. Bis P = 10 schnell; P = 12 mit fünf Typen läuft nur auf Knopfdruck.")
    t = st.select_slider("Typen T (= Zuführplätze)", options=C.T_CHOICES, key="t_select", help="Wie viele Bauteiltypen; so viele Zuführplätze, also T! Zuordnungen.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt Orte und Typen der Instanz und die Zufallsstarts.")
    st.button("🎲 Neue Instanz", width="stretch", on_click=randomize_seed)
    st.markdown("**Die wechselweise Optimierung**")
    start = st.selectbox("Start", C.STARTS, key="start_select", format_func=lambda k: C.START_NAMES[k], help="Von welcher Zuordnung die wechselweise Optimierung ausgeht.")
    st.caption(C.START_HELP[start])

if int(t) > int(p):
    st.warning(f"{int(t)} Typen brauchen mindestens {int(t)} Platzierungen; es werden {int(t)} Platzierungen gerechnet.")
    p = max(int(p), int(t))

sync_query_params({"kind_select": kind, "p_select": int(p), "t_select": int(t), "seed_input": int(seed), "start_select": start})
settings = Settings(kind, int(p), int(t), int(seed), start)

v = None
if (settings.p, settings.t) != C.HEAVY or st.session_state.get("heavy_go") == settings:
    v = _instance(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if v is None:
    st.info(f"Für {settings.p} Platzierungen und {settings.t} Typen sind {C.n_text(120)} Zuordnungen mit je 4 096 Teilmengen zu rechnen (etwa eine halbe Minute, auf Streamlit Cloud länger). Mit dem Knopf starten:")
    if st.button("⏱️ Rechnen", key="heavy_button"):
        st.session_state["heavy_go"] = settings
        st.rerun()
else:
    a = v["alt"][settings.start]
    j = v["joint"]
    m1, m2, m3 = st.columns(3)
    m4, m5 = st.columns(2)
    m1.metric("Gemeinsames Optimum", num(j), help="Alle Zuordnungen, je mit der exakten Reihenfolge (Held–Karp): die Messlatte.")
    m2.metric("Wechselweise Optimierung", num(a["cost"]), delta=("hängt: " if stuck(a["cost"], j) else "") + pct(gap(a["cost"], j), 2) + " über dem Optimum", delta_color="inverse" if stuck(a["cost"], j) else "off", help="Kosten am Fixpunkt vom gewählten Start aus.")
    m3.metric("Schwerpunkt-Zuordnung, dann Reihenfolge", num(v["centroid"]), delta=pct(gap(v["centroid"], j), 2) + " über dem Optimum", delta_color="inverse" if stuck(v["centroid"], j) else "off", help="Die gerundete Relaxation: eine Zuordnung, dann die beste Reihenfolge.")
    m4.metric("Untere Schranke", de(v["lb"], 2), delta=pct(100.0 * (j - v["lb"]) / j, 2) + " unter dem Optimum", delta_color="off", help="Wert der Relaxation: nie über dem Optimum.")
    total = sum(c for _, c in v["basins"])
    ok = sum(c for k, c in v["basins"] if not stuck(k, j))
    m5.metric("Startzuordnungen, die zum Optimum führen", f"{ok} von {total}", help="Von jeder der T! Zuordnungen aus wird die wechselweise Optimierung gestartet.")
    st.plotly_chart(build_instance(v["inst"], v["sigma_joint"], v["order_joint"], a["sigma"] if stuck(a["cost"], j) else None, a["order"] if stuck(a["cost"], j) else None), width="stretch", key="instance_main")
    if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
        st.session_state["pdf_made"] = (settings, build_pdf(settings, v))
    made = st.session_state.get("pdf_made")
    if made and made[0] == settings:
        st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="zuordnung-und-reihenfolge.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kostet es, Zuordnung und Reihenfolge zu trennen?")
if v is not None:
    verdict(verdict_core(settings, v))

st.markdown("---")

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Handbeispiel")
st.caption("Zwei Typen, vier Platzierungen: so klein, dass man beide Fixpunkte von Hand nachrechnen kann. Grün das Optimum, rot gestrichelt der Fixpunkt, an dem die wechselweise Optimierung von der vertauschten Zuordnung aus hängen bleibt.")
h = _hand()
st.plotly_chart(build_instance(h["inst"], h["sigma_joint"], h["order_joint"], h["bad"]["sigma"], h["bad"]["order"], title="Handbeispiel"), width="stretch", key="instance_hand")
verdict(verdict_hand(h))
with st.expander("Handbeispiel zum Nachrechnen"):
    names = C.HAND["names"]
    st.markdown(
        f"**Platzierungen:** A (0, 0) und B (4, 4) vom Typ 0, C (10, 0) und D (9, 3) vom Typ 1; Zuführplätze bei x = 1 (Platz 0) und x = 9 (Platz 1) auf y = −2; Ruheposition (5, −2). Wege sind Tschebyscheff-Abstände (das größere von |Δx| und |Δy|)."
    )
    for key, label in (("good", "Zuordnung Typ 0 → Platz 0, Typ 1 → Platz 1 (Start)"), ("bad", "Zuordnung Typ 0 → Platz 1, Typ 1 → Platz 0 (vertauscht)")):
        r = h[key]
        st.markdown(f"**{label}:** Fixpunkt mit Reihenfolge {''.join(names[i] for i in r['order'])}, {sigma_text(r['sigma'])}, Kosten **{num(r['cost'])}** nach {r['rounds']} Runden.")
        st.dataframe(pd.DataFrame([{"von": a_, "nach": b_, "Länge": num(d)} for a_, b_, d in r["legs"]]), hide_index=True, width="stretch")
        st.caption(f"Summe der Wege: {num(sum(d for _, _, d in r['legs']))} (einzeln nachgerechnet: {num(r['check'])}).")

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wechselweise Optimierung")
st.caption("Kosten nach jedem Schritt vom gewählten Start aus; gestrichelt das gemeinsame Optimum. Ungerade Schritte setzen die Reihenfolge neu, gerade die Zuordnung.")
if v is not None:
    st.plotly_chart(build_rounds(v["alt"][settings.start], v["joint"]), width="stretch", key="rounds")
    verdict(verdict_alternating(settings, v))
    tab = [{"Start": C.START_NAMES[k], "Runden": v["alt"][k]["rounds"], "Fixpunkt": num(v["alt"][k]["cost"]), "Lücke": pct(gap(v["alt"][k]["cost"], v["joint"]), 2), "Zuordnung am Fixpunkt": sigma_text(v["alt"][k]["sigma"])} for k in C.STARTS]
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
else:
    st.caption("Die Instanz ist noch nicht gerechnet (Knopf oben).")

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Fangbecken")
st.caption("Von jeder der T! Startzuordnungen aus läuft die wechselweise Optimierung zu einem Fixpunkt. Das Histogramm zeigt, wie viele Starts bei welchen Endkosten enden.")
if v is not None:
    st.plotly_chart(build_basins(v["basins"], v["joint"]), width="stretch", key="basins")
    verdict(verdict_basins(settings, v))
else:
    st.caption("Die Instanz ist noch nicht gerechnet (Knopf oben).")

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Die Trennungen im Vergleich")
st.caption("Acht Wege, Zuordnung und Reihenfolge zu trennen, über 240 feste Instanzen (vier Familien × (P, T) = (8, 4), (10, 5), (12, 4) × 20 Seeds ab 100000): mittlere und größte Lücke zum gemeinsamen Optimum.")
rows = st.session_state.get("rows_cache")
if rows is None:
    st.info("Die Rechnung läuft über 240 Instanzen, je mit allen Zuordnungen und der exakten Reihenfolge (auf dem Entwicklungsrechner etwa sieben Minuten, bei den großen Instanzen etwa drei Sekunden je Instanz; auf Streamlit Cloud länger). Mit dem Knopf starten:")
    if st.button("⏱️ Feste Instanzen rechnen (240)", key="rows_button"):
        todo = jobs()
        bar = st.progress(0.0, text="Rechne …")
        out = []
        for i, job in enumerate(todo):
            out.append(_row(*job))
            if (i + 1) % 6 == 0 or i + 1 == len(todo):
                bar.progress((i + 1) / len(todo), text=f"{i + 1} von {len(todo)} Instanzen")
        st.session_state["rows_cache"] = out
        st.rerun()
else:
    st.plotly_chart(build_compare(rows, method_stats), width="stretch", key="compare")
    tab = []
    for k in C.METHODS:
        s_ = method_stats(rows, k)
        tab.append({"Verfahren": C.METHOD_NAMES[k], "Lücke im Mittel": pct(s_["mean"], 2), "Median": pct(s_["median"], 2), "größte": pct(s_["max"], 2), "über dem Optimum": f"{s_['stuck']} von {s_['count']}"})
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    verdict(verdict_compare(rows))
    st.plotly_chart(build_family_bars(rows, method_stats), width="stretch", key="family_bars")
    for kd in C.KINDS:
        verdict(verdict_family(rows, kd))
    verdict(verdict_basins_fixed(rows))

st.markdown("---")

# --- Experiment 5 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Relaxation und Rundung")
st.caption("Je Instanz: wie weit die untere Schranke unter dem Optimum liegt (x) und wie weit die gerundete Schwerpunkt-Zuordnung darüber (y). Dazu die Gegenprobe des linearen Schritts.")
if rows is None:
    st.caption("Die festen Instanzen sind noch nicht gerechnet (Knopf in Experiment 4).")
else:
    st.plotly_chart(build_relax(rows), width="stretch", key="relax")
    lb = lb_stats(rows)
    cen = method_stats(rows, "centroid")
    st.info(f"Über {len(rows)} Instanzen: die untere Schranke liegt im Mittel {pct(lb['mean'], 2)} unter dem Optimum (Median {pct(lb['median'], 2)}, größte {pct(lb['max'], 2)}); die gerundete Schwerpunkt-Zuordnung mit der besten Reihenfolge ist in {len(rows) - cen['stuck']} von {len(rows)} Instanzen optimal (größte Lücke {pct(cen['max'], 2)}). "
            f"Gegenprobe des linearen Schritts: die beste Zuordnung bei der optimalen Reihenfolge trifft das Optimum in {assign_at_optimum(rows)} von {len(rows)} Instanzen (folgt aus der Definition des Optimums, kein Befund).")
if v is not None:
    verdict(verdict_relax(settings, v))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Düse** | Mit mehreren Düsen oder Köpfen (Sammeln und Setzen, Portal mit Revolverkopf) entscheiden Bündelung und Wechselzeiten mit; die Zuordnung `σ` und die Reihenfolge `o` sind dann nicht mehr die ganze Entscheidung. | – |
| **Tschebyscheff-Abstand** | Wegen x und y gleichzeitig; echte Köpfe haben Beschleunigungsprofile, Drehungen der Bauteile und getrennte Achsen. | – |
| **Zuführplätze auf einer Geraden** | Mehrere Reihen oder bewegliche Wagen ändern die Geometrie; die Plätze liegen hier fest auf y = −2. | – |
| **Maschinenpraxis ist ein Stand-in** | Die Regel „häufigste Typen in die Mitte“ ist eine Annahme, keine Messung an einer Maschine; echte Praxis nimmt Rüstzeiten, Gurtlängen und Bauteilgrößen dazu. | – |
| **Exakt nur klein** | Das gemeinsame Optimum rechnet alle `T!` Zuordnungen mit Held–Karp (`2^P`-Teilmengen): bis `P = 12`, `T = 5`. Für große Platinen gibt es das Optimum nicht; die Messlatte fehlt dort. | Stück 8 (Schranken im Baum) |
| **Wechselweise: nur ein Fixpunkt-Verfahren** | Mehrere Starts, Abstiege mit Nachbarschaften oder Annealing sind nur in den Starts „bester von 3 Zufallsstarts“ geprüft. | Stück 10, 12 |
| **Relaxation ohne Schranke im Baum** | Die untere Schranke ist schwach (im Mittel einige Prozent unter dem Optimum); nützlich ist die Rundung, nicht die Schranke. | Stück 5, 7 |
| **Die Familien sind erfunden** | Echte Platinen haben andere Verteilungen der Typen (Kondensatoren in der Masse, wenige große Bauteile). Ein Befund über diese Familien ist keiner über Platinen. | Stück 2 |
| **Feste Instanzen, 240** | 20 Seeds je Familie und Größe; kein einzelner Seed ist ein Befund. Zählungen sind ganzzahlig, Prozente mit Gleitkomma gerechnet. | – |
| **Zeit ist implementierungsabhängig** | Python-Fassung; Zeiten schwanken je Rechner, in Tests stehen nie Zeiten. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe.** $P$ Platzierungen $p_1, \dots, p_P$ mit Typ $\tau(i) \in \{1, \dots, T\}$, $T$ Zuführplätze $s_1, \dots, s_T$, Ruheposition $h$, Abstand $d(a, b) = \max(|a_x - b_x|, |a_y - b_y|)$. Entscheidungen: eine Zuordnung $\sigma$ (Typ $\to$ Zuführplatz, bijektiv) und eine Reihenfolge $o$ der Platzierungen.

**Kosten.** Mit $o = (o_1, \dots, o_P)$:
$$\mathrm{cost}(\sigma, o) = d(h, s_{\sigma(\tau(o_1))}) + d(s_{\sigma(\tau(o_1))}, p_{o_1}) + \sum_{k=2}^{P} \big[ d(p_{o_{k-1}}, s_{\sigma(\tau(o_k))}) + d(s_{\sigma(\tau(o_k))}, p_{o_k}) \big] + d(p_{o_P}, h).$$

**Feste Reihenfolge: Zuordnung.** Jeder Typ $t$ an Platz $m$ kostet $c_{tm}(o) = \sum_{k : \tau(o_k) = t} \big[ d(\text{Vorgänger}_k, s_m) + d(s_m, p_{o_k}) \big]$ (der Vorgänger ist $h$ oder $p_{o_{k-1}}$). Die Kosten sind $\sum_t c_{t\sigma(t)}(o)$ plus ein fester Rest: ein **lineares Zuordnungsproblem**, exakt mit der Ungarischen Methode.

**Feste Zuordnung: Reihenfolge.** Jede Platzierung $i$ hat feste Eintrittskosten (Vorgänger $\to$ ihr Zuführplatz $\to$ $p_i$): ein **asymmetrisches Rundreiseproblem** über die $P$ Platzierungen mit Start und Ende $h$. Exakt mit Held–Karp in $O(2^P P^2)$.

**Gemeinsames Optimum.** $\min_\sigma \min_o \mathrm{cost}(\sigma, o)$: alle $T!$ Zuordnungen mit je einer Held–Karp-Rechnung.

**Wechselweise Optimierung.** $o_0 = \arg\min_o \mathrm{cost}(\sigma_0, o)$, $\sigma_1 = \arg\min_\sigma \mathrm{cost}(\sigma, o_0)$, $o_1 = \arg\min_o \mathrm{cost}(\sigma_1, o)$, … Die Kosten sinken nicht; sie bleiben bei einem Fixpunkt stehen, der ein lokales Optimum beider Schritte, aber nicht der Gesamtaufgabe ist. Bei Gleichstand wird nur bei echter Verbesserung ($> 10^{-9}$) weitergerechnet.

**Relaxation.** Ohne die Reihenfolge zählt von jeder Platzierung nur das Bein Zuführplatz $\to$ Platzierung: $c_{tm} = \sum_{i : \tau(i) = t} d(s_m, p_i)$. Das lineare Zuordnungsproblem $\min_\sigma \sum_t c_{t\sigma(t)}$ liefert die **Schwerpunkt-Zuordnung** (die gerundete Lösung); mit den kürzesten Wegen von und zur Ruheposition und je einem kürzesten Anfahrtsweg für alle Platzierungen bis auf eine ergibt sich eine **untere Schranke** des gemeinsamen Optimums.

**Literatur.** Zur Bestückreihenfolge und zur Zuführplatz-Zuordnung (Bestückautomaten, Rundreise mit gekoppelter Zuordnung) und zu alternierenden Minimierungsverfahren kenne ich Namen aus der Fachliteratur, **nicht nachgelesen**. Die Fassung hier ist eigen und wird durch Tests geprüft
(Aufzählung aller Zuordnungen und Reihenfolgen bei kleinen Größen, Held–Karp gegen Aufzählung, Ungarische Methode gegen scipy, Handbeispiel von Hand). Die Zahlen sind eigene Rechnungen an eigenen, erfundenen Familien.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
