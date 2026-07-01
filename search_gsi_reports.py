"""
search_gsi_reports.py

Searches GSI technical report PDFs for key geological terms relevant to
the rebuttal for Reviewer 1, Comment 3 (geological justification for
mixed deposit types in the Dharwar Craton).

Targets:
- Co-occurrence of gold + chromite + iron + polymetallic in greenstone belts
- Structural controls (shear zones, faults) common to multiple deposit types
- Buffer distance / deposit footprint information
- Metallogenic framework statements

Usage:
    python search_gsi_reports.py

Outputs:
    results/gsi_search/search_results.txt   -- ranked hits with excerpts
    results/gsi_search/relevant_reports.csv -- list of most relevant PDFs
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict

# Install pdfminer if needed
try:
    from pdfminer.high_level import extract_text
    from pdfminer.pdfparser import PDFSyntaxError
except ImportError:
    os.system("pip install pdfminer.six --break-system-packages -q")
    from pdfminer.high_level import extract_text
    from pdfminer.pdfparser import PDFSyntaxError

# ── Configuration ─────────────────────────────────────────────────────────────
REPORTS_DIR = Path("data/external/karnataka_ap/technical_reports")
OUT_DIR     = Path("results/gsi_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Terms we're searching for — grouped by purpose
SEARCH_TERMS = {
    "deposit_cooccurrence": [
        "gold and chromite", "gold and iron", "chromite and gold",
        "polymetallic and gold", "associated with gold",
        "co-occurrence", "cooccurrence", "multiple deposit",
        "gold.*chromite", "chromite.*gold", "gold.*iron.*chromite",
    ],
    "greenstone_framework": [
        "greenstone belt", "schist belt", "dharwar",
        "chitradurga", "hutti", "sandur", "nuggihalli",
        "archean", "supracrustal", "banded iron formation", "bif",
        "ultramafic", "mafic-ultramafic",
    ],
    "structural_control": [
        "shear zone", "fault intersection", "structural control",
        "orogenic gold", "hydrothermal", "fluid conduit",
        "brittle-ductile", "mineralization controlled",
    ],
    "deposit_footprint": [
        "buffer", "zone of influence", "mineralized zone width",
        "strike length", "deposit dimension", "ore zone",
        "500 m", "500m", "mineralization extent",
    ],
    "metallogenic_episode": [
        "metallogenic", "mineralization episode", "tectono-magmatic",
        "late archean", "proterozoic", "2700 ma", "2500 ma",
    ],
}

# Toposheets covering the study area (57A, 57B, 57E, 57F series)
TARGET_TOPOSHEETS = ["57A", "57B", "57E", "57F", "57C", "57D"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_target_toposheet(pdf_path):
    """Check if PDF is from a relevant toposheet."""
    path_str = str(pdf_path).upper()
    return any(ts in path_str for ts in TARGET_TOPOSHEETS)


def extract_text_safe(pdf_path, max_pages=10):
    """Extract text from PDF, limiting to first max_pages pages."""
    try:
        text = extract_text(str(pdf_path), maxpages=max_pages)
        return text.lower() if text else ""
    except (PDFSyntaxError, Exception):
        return ""


def find_context(text, term, window=200):
    """Find all occurrences of term in text with surrounding context."""
    contexts = []
    # Handle regex terms
    try:
        pattern = re.compile(term, re.IGNORECASE)
        for m in pattern.finditer(text):
            start = max(0, m.start() - window)
            end   = min(len(text), m.end() + window)
            snippet = text[start:end].replace('\n', ' ').strip()
            contexts.append(snippet)
    except re.error:
        # Plain string search
        idx = 0
        term_lower = term.lower()
        while True:
            pos = text.find(term_lower, idx)
            if pos == -1:
                break
            start = max(0, pos - window)
            end   = min(len(text), pos + window + len(term))
            snippet = text[start:end].replace('\n', ' ').strip()
            contexts.append(snippet)
            idx = pos + 1
    return contexts[:3]  # max 3 per term per doc


# ── Main search ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GSI Technical Report Search — Dharwar Craton")
    print("=" * 60)

    # Find all PDFs
    all_pdfs = list(REPORTS_DIR.rglob("*.pdf"))
    target_pdfs = [p for p in all_pdfs if is_target_toposheet(p)]
    other_pdfs  = [p for p in all_pdfs if not is_target_toposheet(p)]

    print(f"\nTotal PDFs     : {len(all_pdfs)}")
    print(f"Target sheets  : {len(target_pdfs)}")
    print(f"Other sheets   : {len(other_pdfs)}")
    print(f"\nSearching target toposheets first...")

    results = []  # (score, pdf_path, hits)

    def search_pdf_list(pdf_list, label):
        for i, pdf in enumerate(pdf_list):
            if i % 50 == 0:
                print(f"  [{label}] {i}/{len(pdf_list)} PDFs searched...")

            text = extract_text_safe(pdf, max_pages=15)
            if not text or len(text) < 100:
                continue

            hits = {}
            score = 0
            for category, terms in SEARCH_TERMS.items():
                cat_hits = []
                for term in terms:
                    contexts = find_context(text, term)
                    if contexts:
                        cat_hits.append((term, contexts))
                        score += len(contexts)
                if cat_hits:
                    hits[category] = cat_hits

            if score > 0:
                results.append((score, pdf, hits))

    search_pdf_list(target_pdfs, "target")

    # Only search other PDFs if we have fewer than 20 good results
    if len([r for r in results if r[0] >= 3]) < 20:
        print(f"\nSearching remaining PDFs...")
        search_pdf_list(other_pdfs[:200], "other")  # cap at 200

    # Sort by score
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\nFound {len(results)} PDFs with relevant content")
    print(f"Top hits: {[r[0] for r in results[:10]]}")

    # ── Write results ─────────────────────────────────────────────────────────
    out_txt = OUT_DIR / "search_results.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("GSI TECHNICAL REPORT SEARCH RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total PDFs searched: {len(all_pdfs)}\n")
        f.write(f"PDFs with hits: {len(results)}\n\n")
        f.write("PURPOSE: Support rebuttal for Reviewer 1, Comment 3\n")
        f.write("(Geological justification for mixed deposit types)\n\n")

        for rank, (score, pdf, hits) in enumerate(results[:30], 1):
            f.write(f"\n{'='*60}\n")
            f.write(f"RANK {rank} | Score: {score} | {pdf.name}\n")
            f.write(f"Path: {pdf}\n")
            f.write(f"Categories: {list(hits.keys())}\n")
            f.write("-" * 40 + "\n")

            for category, term_hits in hits.items():
                f.write(f"\n[{category.upper()}]\n")
                for term, contexts in term_hits[:2]:
                    f.write(f"  Term: '{term}'\n")
                    for ctx in contexts[:2]:
                        f.write(f"  >>> ...{ctx[:300]}...\n\n")

    print(f"\nDetailed results: {out_txt}")

    # ── Write CSV of relevant reports ─────────────────────────────────────────
    out_csv = OUT_DIR / "relevant_reports.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "filename", "path",
                         "categories", "toposheet"])
        for rank, (score, pdf, hits) in enumerate(results[:50], 1):
            toposheet = next((ts for ts in TARGET_TOPOSHEETS
                              if ts in str(pdf).upper()), "other")
            writer.writerow([rank, score, pdf.name, str(pdf),
                             "|".join(hits.keys()), toposheet])

    print(f"Report list   : {out_csv}")

    # ── Quick summary for rebuttal ────────────────────────────────────────────
    out_summary = OUT_DIR / "rebuttal_citations.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("POTENTIAL GSI CITATIONS FOR REBUTTAL\n")
        f.write("=" * 60 + "\n\n")
        f.write("Top reports with deposit co-occurrence evidence:\n\n")

        cooccurrence_hits = [(s, p, h) for s, p, h in results
                              if "deposit_cooccurrence" in h][:10]

        if cooccurrence_hits:
            for rank, (score, pdf, hits) in enumerate(cooccurrence_hits, 1):
                f.write(f"{rank}. {pdf.name}\n")
                f.write(f"   Path: {pdf}\n")
                terms_found = [t for t, _ in
                               hits.get("deposit_cooccurrence", [])]
                f.write(f"   Terms found: {terms_found}\n\n")
        else:
            f.write("No direct deposit co-occurrence terms found.\n")
            f.write("Check greenstone_framework hits instead.\n\n")
            greenstone_hits = [(s, p, h) for s, p, h in results
                                if "greenstone_framework" in h][:10]
            for rank, (score, pdf, hits) in enumerate(greenstone_hits, 1):
                f.write(f"{rank}. {pdf.name}\n")
                f.write(f"   Path: {pdf}\n\n")

    print(f"Citation list : {out_summary}")
    print(f"\nDone. Check results/gsi_search/ for all outputs.")
    print(f"Most important file: search_results.txt (top 30 ranked reports)")


if __name__ == "__main__":
    main()