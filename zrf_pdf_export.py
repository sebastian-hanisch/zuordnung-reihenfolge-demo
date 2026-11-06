"""PDF-Export des aktuellen Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import zrf_constants as C
from zrf_evaluation import Settings, de, gap, kind_name, num, pct, sigma_text

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Δ": "Delta", "↔": "<->", "„": '"', "“": '"', "²": "2", "³": "3", "σ": "sigma"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def _table(pdf, widths, header, rows):
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(widths, header):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in rows:
        for w, x in zip(widths, r):
            pdf.cell(w, 5.5, _s(x), border=1)
        pdf.ln()


def build_pdf(s: Settings, v: dict) -> bytes:
    """`v`: das Ergebnis von `instance_data` (alle Verfahren auf einer Instanz)."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Zuordnung und Reihenfolge beim Bestücken"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Familie {kind_name(s.kind)}, {s.p} Platzierungen, {s.t} Typen, Seed {s.seed}; Start der wechselweisen Optimierung: {C.START_NAMES[s.start]}. "
                            "Eine Instanz, kein Befund über die Familie. Eine Düse, erfundene Familien, Tschebyscheff-Fahrstrecke; die Maschinenpraxis ist eine Stand-in-Regel. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    j = v["joint"]
    rows = [(C.METHOD_NAMES[k], num(v[k]), pct(gap(v[k], j), 2)) for k in C.METHODS]
    rows.insert(0, ("Gemeinsames Optimum (alle Zuordnungen, je exakte Reihenfolge)", num(j), pct(0, 2)))
    _table(pdf, (110, 25, 30), ("Verfahren", "Kosten", "Lücke"), rows)
    pdf.ln(3)
    pdf.multi_cell(0, 5, _s(f"Optimale Zuordnung: {sigma_text(v['sigma_joint'])}. Untere Schranke der Relaxation: {de(v['lb'], 2)}. Beste Zuordnung bei der optimalen Reihenfolge: {num(v['assign_at_opt_order'])}."), new_x="LMARGIN", new_y="NEXT")
    ends = ", ".join(f"{num(k)}: {c}" for k, c in sorted(v["basins"]))
    pdf.multi_cell(0, 5, _s(f"Fangbecken (Endkosten: Zahl der Startzuordnungen): {ends}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: zuordnung-reihenfolge-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
