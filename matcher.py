from __future__ import annotations
import os
import io
import csv
from typing import List, Dict, Tuple, Set

from extractors import (
    load_text_from_file,
    extract_skills,
    extract_periods,
    months_gap_chain,
)

# ---------- core matching ----------

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    return 100.0 * (len(inter) / max(1, len(union)))

def summarize_periods_html(periods: List[Tuple[str,str]]) -> str:
    if not periods:
        return "—"
    return "<br>".join(f"{s} — {e}" for s,e in periods)

def gaps_html(gaps: List[int]) -> str:
    if not gaps:
        return "None"
    return " → ".join(f"{m} months" for m in gaps)

def first_job_gap_months(edu_periods: List[Tuple[str,str]], exp_periods: List[Tuple[str,str]]) -> int | None:
    if not edu_periods or not exp_periods:
        return None
    # last education end vs first job start
    def _ord_end(p):  # end ordinal
        from extractors import _to_y_m, _ym_to_ord
        end = _to_y_m(p[1])
        start = _to_y_m(p[0])
        return _ym_to_ord(*(end or start or (0,1)))
    def _ord_start(p):
        from extractors import _to_y_m, _ym_to_ord
        start = _to_y_m(p[0])
        return _ym_to_ord(*(start or (0,1)))

    last_edu_end = sorted(edu_periods, key=_ord_end)[-1]
    first_job_start = sorted(exp_periods, key=_ord_start)[0]
    from extractors import _to_y_m, _ym_to_ord
    e_end = _to_y_m(last_edu_end[1]) or _to_y_m(last_edu_end[0])
    j_start = _to_y_m(first_job_start[0])
    if not e_end or not j_start:
        return None
    return max(0, _ym_to_ord(*j_start) - _ym_to_ord(*e_end))

def match_resume_to_jds(resume_path: str, jd_paths: List[str]) -> List[Dict]:
    r_text = load_text_from_file(resume_path)
    r_skills = extract_skills(r_text)

    # simple heuristic splits: education/experience can be anywhere; gather all periods in text
    r_periods = extract_periods(r_text)
    # without labeled sections, we just show all periods under both groups but keep gaps separate
    edu_periods = r_periods
    exp_periods = r_periods
    edu_gaps = months_gap_chain(edu_periods)
    exp_gaps = months_gap_chain(exp_periods)
    edu_first_job_gap = first_job_gap_months(edu_periods, exp_periods)

    rows = []
    for jp in jd_paths:
        j_text = load_text_from_file(jp)
        j_skills = extract_skills(j_text)

        matched = sorted((r_skills & j_skills))
        missing = sorted((j_skills - r_skills))
        score = jaccard(r_skills, j_skills)

        rows.append({
            "jd_name": os.path.basename(jp),
            "score": round(score, 2),
            "resume_loc": "",  # (placeholder columns your UI expects)
            "jd_loc": "",
            "matched": matched,
            "missing": missing,
            "edu_first_job_gap": f"{edu_first_job_gap} months" if edu_first_job_gap is not None else "—",
            "edu_periods_html": summarize_periods_html(edu_periods),
            "exp_periods_html": summarize_periods_html(exp_periods),
            "edu_gaps_html": gaps_html(edu_gaps),
            "exp_gaps_html": gaps_html(exp_gaps),
        })
    # high → low
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows

# ---------- renderers ----------

def as_html_table(rows: List[Dict]) -> str:
    cols = [
        "JD Name","Score %","Resume Loc","JD Loc",
        "Matched Skills","Missing Skills","Edu → First Job Gap",
        "Education Periods","Experience Periods","Education Gaps","Experience Gaps",
    ]
    html = io.StringIO()
    w = html.write
    w("<table><thead><tr>")
    for c in cols: w(f"<th>{c}</th>")
    w("</tr></thead><tbody>")
    for r in rows:
        w("<tr>")
        w(f"<td>{r['jd_name']}</td>")
        w(f"<td>{r['score']}</td>")
        w(f"<td>{r['resume_loc']}</td>")
        w(f"<td>{r['jd_loc']}</td>")
        w(f"<td>{', '.join(r['matched'])}</td>")
        w(f"<td>{', '.join(r['missing'])}</td>")
        w(f"<td>{r['edu_first_job_gap']}</td>")
        w(f"<td>{r['edu_periods_html']}</td>")
        w(f"<td>{r['exp_periods_html']}</td>")
        w(f"<td>{r['edu_gaps_html']}</td>")
        w(f"<td>{r['exp_gaps_html']}</td>")
        w("</tr>")
    w("</tbody></table>")
    return html.getvalue()

def as_csv(rows: List[Dict]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "jd_name","score","resume_loc","jd_loc",
        "matched_skills","missing_skills","edu_first_job_gap",
        "education_periods","experience_periods","education_gaps","experience_gaps",
    ])
    for r in rows:
        writer.writerow([
            r["jd_name"], r["score"], r["resume_loc"], r["jd_loc"],
            "; ".join(r["matched"]), "; ".join(r["missing"]), r["edu_first_job_gap"],
            r["edu_periods_html"].replace("<br>", " | "),
            r["exp_periods_html"].replace("<br>", " | "),
            r["edu_gaps_html"], r["exp_gaps_html"],
        ])
    return out.getvalue().encode("utf-8")
