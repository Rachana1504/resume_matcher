# matcher.py
from __future__ import annotations
from typing import Dict, List
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from extractors import extract_skills, extract_periods, split_resume_sections, get_nlp

# Cache the embedder
_EMBEDDER: SentenceTransformer | None = None

def _embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(os.getcwd()))
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER

def _text_from_cache_entry(entry) -> str:
    if isinstance(entry, dict):
        return entry.get("text", "") or ""
    return str(entry or "")

def _locations_from_entry(entry) -> str:
    if isinstance(entry, dict):
        return entry.get("location","") or ""
    return ""

def _embed(text: str) -> np.ndarray:
    model = _embedder()
    vec = model.encode([text or ""], normalize_embeddings=True)
    return np.asarray(vec)

def _score(resume_text: str, jd_text: str, resume_skills: List[str], jd_skills: List[str]) -> float:
    v1 = _embed(resume_text)
    v2 = _embed(jd_text)
    emb = float(cosine_similarity(v1, v2)[0][0])  # 0..1
    # Skill overlap (Jaccard)
    s1, s2 = set(map(str.lower, resume_skills)), set(map(str.lower, jd_skills))
    overlap = (len(s1 & s2) / max(1, len(s1 | s2)))
    # Blend
    return (0.75 * emb) + (0.25 * overlap)

def match_resume_to_jds(resume_path: str, jd_cache: Dict[str, dict | str]) -> List[Dict]:
    # Read resume
    text = _read_any(resume_path)
    resume_edu, resume_exp = split_resume_sections(text)
    resume_periods = extract_periods(text)
    edu_periods = extract_periods(resume_edu)
    exp_periods = extract_periods(resume_exp)
    # Skills (SkillNer only; if backend missing, returns [])
    resume_skills = extract_skills(text)

    # Location extraction (minimal, can be upgraded)
    resume_loc = _guess_location(text)

    results = []
    for jd_name, jd_entry in jd_cache.items():
        jd_text = _text_from_cache_entry(jd_entry)
        jd_loc = _locations_from_entry(jd_entry) or _guess_location(jd_text)

        jd_skills = extract_skills(jd_text)

        score = _score(text, jd_text, resume_skills, jd_skills)
        matched = sorted(set(map(str.lower, resume_skills)) & set(map(str.lower, jd_skills)))
        missing = sorted(set(map(str.lower, jd_skills)) - set(map(str.lower, resume_skills)))

        results.append({
            "jd_file": jd_name,
            "similarity_score_percent": round(score * 100, 1),
            "resume_location": resume_loc or "Not Mentioned",
            "jd_location": jd_loc or "Not Mentioned",
            "matched_skills": matched,
            "missing_skills": missing,
            "education_to_first_job_gap_months": _edu_to_first_job_gap_months(edu_periods, exp_periods),
            "education_periods": edu_periods or [],
            "experience_periods": exp_periods or resume_periods or [],
            "education_gaps": _gaps(edu_periods),
            "experience_gaps": _gaps(exp_periods),
        })

    results.sort(key=lambda r: r["similarity_score_percent"], reverse=True)
    return results

# ----------------- helpers -----------------
def _read_any(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        import fitz
        doc = fitz.open(path)
        try:
            txt = "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
        return txt
    if p.endswith(".docx"):
        import docx2txt
        return docx2txt.process(path) or ""
    # plaintext
    return Path(path).read_text(encoding="utf-8", errors="ignore")

from pathlib import Path
import re

LOC_RE = re.compile(r"\b([A-Z][a-z]+(?:[ ,][A-Z][a-z]+)*)\b[, ]+\b([A-Z]{2,})\b")
def _guess_location(text: str) -> str | None:
    m = LOC_RE.search(text or "")
    if m: 
        return m.group(0)
    return None

def _months(s: str) -> int | None:
    # crude month/year to absolute month index
    s = (s or "").lower()
    if "present" in s or "current" in s or "ongoing" in s:
        return None
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", s)
    y = re.search(r"(19|20)\d{2}", s)
    if not y:
        return None
    year = int(y.group(0))
    month = 1
    if m:
        month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec"].index(m.group(1)) + 1
        if month > 12: month = 9  # 'sept' maps to 9 above
    return year*12 + month

def _gaps(periods: List[Dict]) -> List[Dict]:
    # sort by start
    times = []
    for p in periods or []:
        s = _months(p.get("start","")); e = _months(p.get("end","")) or _months("present 2099")
        if s: times.append((s,e,p))
    times.sort(key=lambda x: x[0])
    gaps = []
    for (_, e_prev, p_prev), (s, e, p) in zip(times, times[1:]):
        if s > e_prev:
            gaps.append({"between": f"{p_prev.get('entry','')} → {p.get('entry','')}", "gap_months": s - e_prev})
    return gaps

def _edu_to_first_job_gap_months(edu_periods: List[Dict], exp_periods: List[Dict]) -> int | str:
    if not edu_periods or not exp_periods: return "N/A"
    edu_end = max((_months(p.get("end","")) or 0) for p in edu_periods)
    first_job = min((_months(p.get("start","")) or 0) for p in exp_periods)
    if edu_end and first_job and first_job >= edu_end:
        return first_job - edu_end
    return "N/A"
