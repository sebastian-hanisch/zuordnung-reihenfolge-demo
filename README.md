# Zuordnung und Reihenfolge – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-zuordnung-reihenfolge-demo.streamlit.app/)** -->

Stück 14 von 15 (gelb, optional) der **Konzepte-Reihe „Anordnen: relaxieren und runden“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Konzepte-Seite der Reihe gibt es noch nicht). Die Aufgabe stammt aus dem Abschnitt „Bestückung“ der [Leiterplatte-Demo](https://github.com/sebastian-hanisch/leiterplatte-demo); relaxieren und runden einer Zuordnung kennt die Reihe aus Stück [6, Birkhoff-Relaxation](https://github.com/sebastian-hanisch/birkhoff-relaxation-demo). Hier ist es eine **zweite Entscheidung neben der Zuordnung**: die Reihenfolge.

## Warum dieses Problem

Ein Bestückkopf holt jedes Bauteil an einem **Zuführplatz** ab und setzt es auf seine **Platzierung**. Zwei Entscheidungen hängen zusammen: **welcher Typ an welchen Zuführplatz** (die Zuordnung `σ`) und **in welcher Reihenfolge** der Kopf die Platzierungen anfährt (`o`). Jede Entscheidung allein ist leicht: bei fester Reihenfolge ist `σ` ein **lineares Zuordnungsproblem** (Ungarische Methode), bei festem `σ` ist `o` ein **Rundreiseproblem** (bei dieser Größe exakt durch Held–Karp). Naheliegend ist, **wechselweise** zu optimieren. Die Frage: **Was kostet es, die beiden Entscheidungen zu trennen?**

Die Antwort dieser Messung: **Die wechselweise Optimierung bleibt oft an einem Fixpunkt hängen, der nicht das Optimum ist** (aus der Maschinenpraxis in 176 von 240 Instanzen, im Mittel 2,14 % über dem Optimum, im schlimmsten Fall 12,25 %). Die **Relaxation, gerundet**, ist dagegen fast exakt: die Zuordnung nach den Schwerpunkten der Wege (ein einziges lineares Zuordnungsproblem) mit der besten Reihenfolge ist in **239 von 240** Instanzen optimal. Die **untere Schranke** der Relaxation ist dabei schwach (im Mittel 4,62 % unter dem Optimum).

## Modell

- **Maschine** (eigenes Modell, wie der Abschnitt „Bestückung“ der Leiterplatte-Demo, hier mit einer Düse; die Literatur ist **nicht eingesehen**): ein Kopf, der immer ein Bauteil trägt; Tschebyscheff-Abstand (`x` und `y` gleichzeitig). Brett `[0, 10] × [0, 6]`, die `T` Zuführplätze auf der Geraden `y = −2` zwischen `x = 1` und `x = 9`, Ruheposition `(5, −2)`. `P` Platzierungen mit Ort und Typ (jeder Typ mindestens einmal).
- **Kosten:** `cost(σ, o) = d(h, s_σ(t_o1)) + d(s_σ(t_o1), p_o1) + Σ_k [ d(p_ok, s_σ(t_ok+1)) + d(s_σ(t_ok+1), p_ok+1) ] + d(p_oP, h)`: die Länge der Fahrt. `σ` ist eine Permutation (Typ → Zuführplatz), `o` eine Permutation der Platzierungen.
- **Bei fester Reihenfolge** sind die Kosten `Σ_{t,m} c[t][m] · x[t][m]` plus einen festen Rest (Rückweg): Zuordnungsproblem, exakt mit der Ungarischen Methode. **Bei festem `σ`** ist `o` ein asymmetrisches Handlungsreisendenproblem, exakt durch **Held–Karp** über Teilmengen (`P ≤ 12`).
- **Gemeinsames Optimum:** alle `T!` Zuordnungen, je mit Held–Karp (die Messlatte jeder Lücke). **Lücke** = (Kosten − Optimum) / Optimum in Prozent; „hängt“ heißt: Kosten über dem Optimum um mehr als `10⁻⁶`.
- **Verfahren** (acht, an derselben Instanz): *Maschinenpraxis* (häufigste Typen auf die Zuführplätze nahe der Ruheposition; eine **Stand-in-Regel**, keine Messung an einer Maschine), dann die beste Reihenfolge; *Schwerpunkt-Zuordnung* (Zuordnungsproblem auf `c[t][m] = 2 · Σ_{i ∈ t} d(s_m, p_i)`: die Relaxation ohne Reihenfolge), dann die beste Reihenfolge; *erst die Reihenfolge* (kürzeste Fahrt durch die Platzierungen, ohne Zuführplätze), *dann die Zuordnung*; die **wechselweise Optimierung** (Reihenfolge exakt, dann `σ` exakt, solange die Kosten sinken) aus der Maschinenpraxis, aus den Schwerpunkten, aus einem Zufallsstart, als beste von drei Zufallsstarts, und nach „erst die Reihenfolge“.
- **Untere Schranke:** das Zuordnungsproblem der Beine Zuführplatz → Platzierung, dazu der kürzeste Weg von und zur Ruheposition und je der kürzeste Anfahrtsweg für alle Platzierungen bis auf eine.
- **Fangbecken:** von jeder der `T!` Startzuordnungen aus läuft die wechselweise Optimierung zu einem Fixpunkt; das Histogramm zählt, wie viele Starts bei welchen Endkosten enden.
- **Instanzen:** vier erfundene Familien (zufällig; Cluster je Typ; schief: ein Typ mit etwa der Hälfte der Platzierungen; verschränkt: Typen im Schachbrett), `P` = 6, 8, 10, 12 und `T` = 3, 4, 5. **Feste Reihen:** Seeds ab 100000, drei Größen (`P`, `T`) = (8, 4), (10, 5), (12, 4), vier Familien, je 20 Seeds: **240 Instanzen**. Zufall ist SplitMix64.
- **Experimente:** (1) das Handbeispiel (zwei Typen, vier Platzierungen, von Hand nachzurechnen); (2) die wechselweise Optimierung Schritt für Schritt; (3) die Fangbecken; (4) die Trennungen im Vergleich über die 240 festen Instanzen; (5) Relaxation und Rundung. Formeln im 📐-Expander der App.

## Methodik

Der Abschnitt „Ergebnis“ und die Experimente 1 bis 3 rechnen **eine** Instanz (Familie, `P`, `T`, Seed, Start der wechselweisen Optimierung; 12 Platzierungen mit 5 Typen nur auf Knopfdruck). Die Experimente 4 und 5 laufen auf Knopfdruck über die 240 festen Instanzen (nacheinander gerechnet etwa sieben Minuten auf dem Entwicklungsrechner, bei den großen Instanzen etwa drei Sekunden je Instanz; auf Streamlit Cloud länger). Standard: Cluster, `P = 8`, `T = 4`, Seed 46, Start Maschinenpraxis (eine Instanz, bei der die wechselweise Optimierung aus der Maschinenpraxis hängt: 83,75 gegen das Optimum 80,53). Zeiten stehen in keiner Anzeige und in keinem Test.

## Befunde (gemessen, keine Behauptungen)

Alles Messung an den vier erfundenen Familien und den drei Größen. Lücken sind im Mittel über die Instanzen, Zählungen („über dem Optimum“) ganzzahlig.

1. **Die wechselweise Optimierung hängt oft.** Aus der Maschinenpraxis bleibt sie in **176 von 240** Instanzen über dem Optimum (Lücke im Mittel **2,14 %**, größte 12,25 %); aus einem Zufallsstart in 151 von 240 (1,58 %, größte 11,75 %); als beste von drei Zufallsstarts noch in **90 von 240** (0,57 %, größte 8,59 %). Von den `T!` Startzuordnungen führt im Median nur **ein Viertel** zum Optimum (kleinster Anteil 2 %); die Starts laufen im Median zu 7 verschiedenen Endkosten (größte Zahl 38); nur in 7 von 240 Instanzen führen alle zum Optimum.
2. **Die gerundete Relaxation ist fast exakt.** Die Schwerpunkt-Zuordnung mit der besten Reihenfolge ist in **239 von 240** Instanzen optimal (Lücke im Mittel 0,003 %); die eine Ausnahme (Familie „zufällig“, Seed 100006) liegt 0,62 % darüber. Die wechselweise Optimierung aus den Schwerpunkten ändert daran in keiner Instanz etwas. Die Maschinenpraxis dagegen: 3,93 % im Mittel (213 von 240 über dem Optimum, größte 15,42 %); die Schwerpunkt-Zuordnung ist in jeder Instanz höchstens so teuer wie sie.
3. **Die untere Schranke ist schwach.** Sie liegt im Mittel 4,62 % unter dem Optimum (Median 4,51 %, größte 8,74 %); nützlich ist die **Rundung**, nicht die Schranke. Sie liegt in keiner der 240 Instanzen über dem Optimum.
4. **Erst die Reihenfolge, dann die Zuordnung ist keine Abkürzung:** 189 von 240 über dem Optimum, im Mittel 3,16 % (größte 15,21 %); besser als die Maschinenpraxis in 130 Instanzen. Schließt man **wechselweise** an, sinkt die Lücke auf 0,47 % (102 von 240, größte 4,90 %), also etwa auf das Niveau von drei Zufallsstarts.
5. **Die Familien unterscheiden sich stark.** Wechselweise aus der Maschinenpraxis, mittlere Lücke: Cluster **0,46 %**, zufällig 1,47 %, schief 2,03 %, verschränkt **4,61 %**. Bei den **verschränkten** Typen ändert die wechselweise Optimierung die Maschinenpraxis in 59 von 60 Instanzen gar nicht (sie ist schon ein Fixpunkt); „erst die Reihenfolge“ ist dort mit 7,37 % im Mittel am schlechtesten.
6. **Die Größe:** Maschinenpraxis im Mittel 3,36 % / 5,23 % / 3,19 % bei `(P, T)` = (8, 4) / (10, 5) / (12, 4); der Anteil der Starts, die zum Optimum führen, im Median 33 % / 18 % / 25 % (mit 5 / 13 / 6 verschiedenen Endkosten im Median): fünf Typen (120 Zuordnungen) haben mehr Fangbecken.
7. **Gegenprobe:** die beste Zuordnung bei der **optimalen** Reihenfolge ist in 240 von 240 Instanzen das Optimum. Das folgt aus der Definition des Optimums und ist **kein Befund**; es prüft den linearen Schritt (er verschenkt nichts).

**Handbeispiel:** zwei Typen, vier Platzierungen A (0, 0), B (4, 4) vom Typ 0, C (10, 0), D (9, 3) vom Typ 1, Zuführplätze bei `x = 1` und `x = 9`. Optimum **34** mit `σ = (0, 1)` und Reihenfolge ABCD (Wege 4, 2, 2, 6, 6, 2, 2, 5, 5). Startet die wechselweise Optimierung bei `σ = (1, 0)`, endet sie nach 2 Runden bei **51** (Reihenfolge DACB; Wege 4, 8, 5, 9, 2, 9, 2, 6, 6): ein Fixpunkt, 50 % über dem Optimum, den weder die Reihenfolge noch die Zuordnung verbessert.

### Lücke zum gemeinsamen Optimum, Mittel (Instanzen über dem Optimum)

| Verfahren | alle 240 | (8, 4) | (10, 5) | (12, 4) |
|---|---|---|---|---|
| Maschinenpraxis, dann Reihenfolge | 3,93 % (213) | 3,36 % (66) | 5,23 % (74) | 3,19 % (73) |
| Schwerpunkt-Zuordnung, dann Reihenfolge | 0,00 % (1) | 0,01 % (1) | 0,00 % (0) | 0,00 % (0) |
| Erst die Reihenfolge, dann die Zuordnung | 3,16 % (189) | 3,02 % (61) | 2,96 % (62) | 3,51 % (66) |
| Wechselweise aus der Maschinenpraxis | 2,14 % (176) | 1,26 % (50) | 3,56 % (63) | 1,61 % (63) |
| Wechselweise, beste von 3 Zufallsstarts | 0,57 % (90) | 0,41 % (26) | 0,86 % (36) | 0,45 % (28) |
| Erst die Reihenfolge, dann wechselweise | 0,47 % (102) | 0,46 % (22) | 0,59 % (46) | 0,36 % (34) |

Die genauen Werte (auch je Familie) stehen in `tests/test_claims.py`; Zeiten stehen in keinem Test.

## Ehrliche Grenzen

- **Eine Düse:** mit mehreren Düsen oder Köpfen (Sammeln und Setzen, Revolverkopf) entscheiden Bündelung und Wechselzeiten mit; `σ` und `o` sind dann nicht mehr die ganze Entscheidung.
- **Tschebyscheff-Abstand, Zuführplätze auf einer Geraden:** echte Köpfe haben Beschleunigungsprofile, Drehungen der Bauteile und getrennte Achsen; Wagen und mehrere Reihen ändern die Geometrie.
- **Die Maschinenpraxis ist ein Stand-in:** „häufigste Typen in die Mitte“ ist eine Annahme, keine Messung an einer Maschine. Der Befund „die Schwerpunkt-Zuordnung schlägt sie“ gilt für **diese** Regel.
- **Exakt nur klein:** das gemeinsame Optimum rechnet alle `T!` Zuordnungen mit Held–Karp (`2^P` Teilmengen): bis `P = 12`, `T = 5`. Für große Platinen gibt es die Messlatte nicht; die Lücken der Verfahren dort sind **nicht gemessen**.
- **Gleichstände:** Gibt es mehrere gleich gute Reihenfolgen oder Zuordnungen (bei Tschebyscheff-Abständen häufig), hängt der Fixpunkt der wechselweisen Optimierung davon ab, welche gewählt wird. Hier gilt: Held–Karp nimmt den ersten Index innerhalb von `10⁻⁹` des Minimums, die Zuordnungsmatrix ist auf neun Stellen gerundet. Eine erste Fassung ohne diese Regel ergab auf Linux andere Fixpunkte als auf Windows (170 statt 172 hängende Instanzen von 240); mit der Regel stimmen die Zahlen überein, aber **die Zählungen sind für diese Auflösung gemessen**, nicht für jede mögliche.
- **Wechselweise ist nur ein Fixpunktverfahren:** Nachbarschaftssuche, Annealing und Mehrfachstarts sind nur als „beste von 3 Zufallsstarts“ geprüft.
- **Die untere Schranke ist schwach** (Mittel 4,62 % darunter); im Baum (Stück 8) bräuchte man eine schärfere.
- **Die Familien sind erfunden**, die Typenverteilung echter Platinen ist eine andere; ein Befund über diese Familien ist keiner über Platinen.
- **240 Instanzen, drei Größen,** keine Instanzen aus der Literatur; die Lücken streuen stark (Median 2,38 % gegen Mittel 3,93 % bei der Maschinenpraxis).
- **Eigenes Modell, Literatur nicht nachgelesen:** zur Bestückreihenfolge und Zuführplatz-Zuordnung und zu alternierenden Minimierungsverfahren kenne ich nur Namen aus der Fachliteratur; keine Selbstangabe der Literatur wird übernommen.
- **Zeit ist implementierungsabhängig:** Python-Fassung; in Tests stehen nie Zeiten.

## Verifikation

- **Tests** (`pytest tests/`): 168 Tests, rund sechseinhalb Minuten (Windows, Python 3.14; die 240 Instanzen der Zahlen-Tests rechnen parallel). Handwerte (Handbeispiel 34 und 51 mit allen neun Wegen von Hand; Zuordnungsmatrizen), Aufzählung als Orakel (**alle** Zuordnungen und **alle** Reihenfolgen bei `P = 6`, `T = 3`: Held–Karp, Ungarische Methode, gemeinsames Optimum und kürzeste Fahrt durch die Platzierungen gleich der Aufzählung; Ungarische Methode zusätzlich gegen `scipy`), Eigenschaften (die wechselweise Optimierung steigt nie und endet an einem Fixpunkt beider Schritte, kein Verfahren liegt unter dem gemeinsamen Optimum, die untere Schranke nie darüber, die Schwerpunkt-Zuordnung minimiert die Wegesumme über alle Zuordnungen, die Wege einer Fahrt summieren sich zu den Kosten), Verdikte (Art und Text folgen den Zahlen), Abbildungen (Daten gleich den Werten, die gezeichnete Fahrt hat die Länge der Kosten), AppTest (alle Regler an ihren Grenzen, Presets, Rechenknopf mit verkleinerter Reihe, PDF), und die **Zahlen der README** an den 240 festen Instanzen (`tests/test_claims.py`, parallel gerechnet; Zählungen genau, Prozentwerte auf drei Stellen).
- **Nicht getestet:** Zeiten; kein Mutationstest; Linux-CI: grün im zweiten Lauf (168 bestanden, fünf Minuten); der erste war rot, weil Gleichstände auf Linux anders aufgelöst wurden (Abhilfe siehe „Gleichstände“ in den Grenzen); die Zahlen der Tabellen stammen aus dem Windows-Lauf.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, fünf Experimente, Grenzen, Formulierung |
| `zrf_model.py` | Instanzen (vier Familien), Fahrtkosten, Held–Karp, beste Zuordnung und Reihenfolge, gemeinsames Optimum, wechselweise Optimierung, Startzuordnungen, untere Schranke, Fangbecken |
| `zrf_evaluation.py` | Eine Instanz mit allen Verfahren, Zeilen der festen Reihen, Kennzahlen, Verdikte |
| `zrf_assignment.py` | Ungarische Methode (Kopie aus Stück 8) |
| `zrf_rng.py` | SplitMix64 |
| `zrf_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `zrf_pdf_export.py` | PDF-Export des Ergebnisses |
| `zrf_constants.py` | Größen, Familien, Verfahren, feste Reihen, Farben, Presets |
| `zrf_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Button |
| `zrf_layout.py` | Fester Kopf (Stylesheet) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für das Orakel der Ungarischen Methode) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; kein `scipy` zur Laufzeit) |
| `tests/` | Modell, Auswertung, Verdikte, Abbildungen, Presets, Layout, AppTest, Zahlen der README (`test_claims.py`, `rows.py`), Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Mehrere Düsen, Köpfe oder Zuführer** und Rüstzeiten; **Beschleunigungsprofile**.
- **Große Platinen** (kein gemeinsames Optimum; bräuchte schärfere Schranken oder Metaheuristiken, Stück 10 und 12).
- **Messung der Maschinenpraxis** an einer echten Maschine; **Instanzen aus der Literatur**.
- **Zeiten in Tests.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
