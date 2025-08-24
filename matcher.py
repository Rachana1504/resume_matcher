from __future__ import annotations

import os, torch
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any, List, Tuple

from sentence_transformers import util

from extractors import (
    extract_text,
    extract_resume_data,
    extract_skills,
    normalize_skills,
    clean_entry_name,
)

@lru_cache(maxsize=1)
def _lazy_models():
    import spacy
    from sentence_transformers import SentenceTransformer
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, sbert

def warmup():
    _lazy_models()
    return True

def extract_location(text: str) -> str:
    try:
        nlp, _ = _lazy_models()
        doc = nlp(text)
        locs = [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]
        return locs[0] if locs else "Not Mentioned"
    except Exception:
        return "Not Mentioned"

def _containment_match(jd_norm: set, res_norm: set) -> Tuple[set, set]:
    matched = set()
    for k in jd_norm:
        for r in res_norm:
            if k in r or r in k:
                matched.add(k)
                break
    missing = jd_norm - matched
    return matched, missing

def _jd_skill_sets(jd_entry: Dict[str, Any]):
    jd_text = jd_entry.get("text", "") or ""
    base = jd_entry.get("skills") or extract_skills(jd_text) or []
    jd_norm = normalize_skills(list(base))
    jd_map: Dict[str, str] = {}
    for s in base:
        norm_key = next(iter(normalize_skills([s])), s)
        jd_map.setdefault(norm_key, s)
    for k in jd_norm:
        jd_map.setdefault(k, k)
    return jd_text, jd_norm, jd_map

def _compare(resume_embed, res_norm, resume_loc, jd_name, jd_entry, resume_name):
    if not jd_entry:
        return None
    jd_text, jd_norm, jd_map = _jd_skill_sets(jd_entry)
    jd_embed = torch.tensor(jd_entry["embedding"])
    score = util.pytorch_cos_sim(resume_embed, jd_embed)[0][0].item() * 100.0
    matched_keys, missing_keys = _containment_match(jd_norm, res_norm)
    matched = [jd_map[k] for k in matched_keys if k in jd_map]
    missing = [jd_map[k] for k in missing_keys if k in jd_map]
    return {
        "resume_file": resume_name,
        "jd_file": jd_name,
        "similarity_score_percent": round(score, 2),
        "matched_skills": sorted(set(matched)),
        "missing_skills": sorted(set(missing)),
        "resume_location": resume_loc,
        "jd_location": extract_location(jd_text),
    }

def _ensure_jd_embeddings(jd_cache: Dict[str, dict], sbert):
    """Non-invasive: if a JD lacks an embedding, compute & attach it."""
    import numpy as np
    for name, entry in jd_cache.items():
        if "embedding" not in entry or entry["embedding"] is None:
            txt = entry.get("text", "") or ""
            emb = sbert.encode(txt)
            entry["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb
    return jd_cache

def match_resume_to_jds(resume_path: str, jd_cache: Dict[str, dict]) -> List[Dict[str, Any]]:
    nlp, sbert = _lazy_models()
    text = extract_text(resume_path)
    resume_name = os.path.basename(resume_path)
    resume_embed = sbert.encode(text, convert_to_tensor=True)
    resume_skills, edu, exp, edu_gaps, exp_gaps, edu_to_exp = extract_resume_data(text)
    res_norm = normalize_skills(resume_skills)
    resume_loc = extract_location(text)

    # ensure JD embeddings (added, safe)
    _ensure_jd_embeddings(jd_cache, sbert)

    out: List[Dict[str, Any]] = []

    for jd_name in jd_cache.keys():
        base = _compare(
            resume_embed, res_norm, resume_loc, jd_name, jd_cache.get(jd_name), resume_name
        )
        if not base:
            continue

        def periods(items):
            rows = []
            for e in items or []:
                start = e[1].strftime("%b %Y")
                end_dt = e[2]
                end = "Present" if getattr(end_dt, "year", 0) == 9999 or end_dt > datetime.now() \
                      else end_dt.strftime("%b %Y")
                rows.append({"entry": clean_entry_name(e[0]), "start": start, "end": end})
            return rows

        out.append(
            {
                **base,
                "education_periods": periods(edu),
                "experience_periods": periods(exp),
                "education_gaps": edu_gaps,
                "experience_gaps": exp_gaps,
                "education_to_first_job_gap_months": edu_to_exp,
            }
        )
    return out

# ---------- Parallel multi-resume matcher (added) ----------
from concurrent.futures import ThreadPoolExecutor, as_completed

def _get_resume_text_fast(path: str) -> str:
    try:
        from extractors import extract_text_fast
        return extract_text_fast(path)
    except Exception:
        from extractors import extract_text
        return extract_text(path)

def _quick_skills(text: str, fast: bool) -> list[str]:
    if fast:
        try:
            from extractors import extract_skills_fast
            return extract_skills_fast(text)
        except Exception:
            pass
    from extractors import extract_skills
    return extract_skills(text)

def _match_one_resume_against_jds(resume_path: str, jd_paths: List[str], fast: bool) -> dict:
    from jd_cache import get_jd_text_fast
    _, sbert = _lazy_models()

    # Build a transient JD cache (with embeddings) once for this batch
    jd_cache: Dict[str, dict] = {}
    for jp in jd_paths:
        jt = get_jd_text_fast(jp)
        jd_cache[os.path.basename(jp)] = {"text": jt, "location": ""}

    _ensure_jd_embeddings(jd_cache, sbert)

    # Reuse your original scorer
    results = match_resume_to_jds(resume_path, jd_cache)

    # Also expose top-level resume skills/periods quickly if needed by UI
    text = _get_resume_text_fast(resume_path)
    skills, edu, exp, edu_gaps, exp_gaps, edu_to_exp = extract_resume_data(text)
    if fast and not skills:
        skills = _quick_skills(text, fast=True)
    return {
        "resume": os.path.basename(resume_path),
        "results": results,
        "skills": skills,
        "education_periods": [
            {"entry": clean_entry_name(e[0]), "start": e[1].strftime("%b %Y"),
             "end": ("Present" if getattr(e[2], "year", 0) == 9999 or e[2] > datetime.now() else e[2].strftime("%b %Y"))}
            for e in edu
        ],
        "experience_periods": [
            {"entry": clean_entry_name(e[0]), "start": e[1].strftime("%b %Y"),
             "end": ("Present" if getattr(e[2], "year", 0) == 9999 or e[2] > datetime.now() else e[2].strftime("%b %Y"))}
            for e in exp
        ],
        "education_gaps": edu_gaps,
        "experience_gaps": exp_gaps,
        "education_to_first_job_gap_months": edu_to_exp,
    }

def match_many(resume_paths: List[str], jd_paths: List[str], fast: bool = True, max_workers: int = 4) -> List[dict]:
    """
    Parallel, cached matching for multiple resumes x multiple JDs.
    - fast=True: uses cached text and lightweight skill extraction for speed.
    Returns a list of objects, one per resume, each containing its JD results.
    """
    if not resume_paths or not jd_paths:
        return []
    out: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_match_one_resume_against_jds, rp, jd_paths, fast): rp for rp in resume_paths}
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:
                rp = futs[f]
                out.append({"resume": os.path.basename(rp), "error": str(e), "results": []})
    return out
