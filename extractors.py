import os
import re
import io
from typing import List, Tuple, Set, Dict

import fitz  # PyMuPDF
import docx2txt
import spacy
from spacy.matcher import PhraseMatcher

# SkillNer
try:
    from skillNer.skill_extractor_class import SkillExtractor
except Exception:  # older package naming fallback
    from skillNer import SkillExtractor  # type: ignore

# ---------- file -> text ----------

ALLOWED_EXT = {".pdf", ".docx", ".txt"}

def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _pdf_text(path)
    if ext == ".docx":
        return _docx_text(path)
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {ext}")

def _pdf_text(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def _docx_text(path: str) -> str:
    # docx2txt is robust for mixed runs and tables
    return docx2txt.process(path) or ""

# ---------- NLP init (spaCy + SkillNer) ----------

_NLP = None
_SKILL_EXTRACTOR = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP

def _get_skill_extractor():
    """Build SkillExtractor once. Uses SkillNer’s built-in DB and matchers."""
    global _SKILL_EXTRACTOR
    if _SKILL_EXTRACTOR is None:
        nlp = _get_nlp()
        _SKILL_EXTRACTOR = SkillExtractor(nlp)  # default DB & matchers
    return _SKILL_EXTRACTOR

# ---------- skill extraction (SkillNer) ----------

def extract_skills(text: str) -> Set[str]:
    """
    Use SkillNer to annotate skills, merging all result buckets.
    Works across SkillNer 1.0.x variants by being defensive with keys.
    """
    text = (text or "").strip()
    if not text:
        return set()
    se = _get_skill_extractor()
    ann = se.annotate(text)  # type: ignore

    buckets = []
    results = ann.get("results") or {}
    # Collect any list under results
    for v in results.values():
        if isinstance(v, list):
            buckets.extend(v)

    found: Set[str] = set()
    for item in buckets:
        if not isinstance(item, dict):
            continue
        # common keys across SkillNer outputs
        for k in ("doc_node_value", "skill", "skill_name", "ngram", "text"):
            if k in item and item[k]:
                found.add(str(item[k]).strip().lower())
                break

    # fallback: noun chunks (if SkillNer finds nothing)
    if not found:
        nlp = _get_nlp()
        doc = nlp(text)
        for nc in doc.noun_chunks:
            s = re.sub(r"[^a-z0-9+\-#/. ]", "", nc.text.lower())
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > 1 and " " in s:
                found.add(s)

    return found

# ---------- date/period parsing ----------

_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_YEAR  = r"(20\d{2}|19\d{2})"
# forms: Jan 2020 — Mar 2021, Jan-2020 to Mar-2021, Jan 2020 – Present, 2020 - 2022, etc.
PERIOD_RE = re.compile(
    rf"(?P<m1>{_MONTH})\s*[-/ ]?\s*(?P<y1>{_YEAR})\s*(?:[–—\-to]+)\s*(?:(?P<m2>{_MONTH})\s*[-/ ]?\s*(?P<y2>{_YEAR})|(?P<present>present|current))",
    re.IGNORECASE,
)

# year-only backup
YEAR_RANGE_RE = re.compile(rf"(?P<y1>{_YEAR})\s*[–—\-to]+\s*(?P<y2>{_YEAR}|present|current)", re.IGNORECASE)

_MONTH_NUM = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
    "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,
    "september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}

def _ym_to_ord(y:int, m:int) -> int:
    return y*12 + (m-1)

def _months_between(start: Tuple[int,int], end: Tuple[int,int]) -> int:
    y1,m1 = start; y2,m2 = end
    return max(0, _ym_to_ord(y2,m2) - _ym_to_ord(y1,m1))

def extract_periods(text: str) -> List[Tuple[str,str]]:
    """
    Returns list of ('Mon YYYY', 'Mon YYYY/Present') strings found anywhere in the text.
    """
    out: List[Tuple[str,str]] = []
    for m in PERIOD_RE.finditer(text):
        m1, y1 = m.group("m1"), int(m.group("y1"))
        if m.group("present"):
            out.append((f"{m1} {y1}", "Present"))
        else:
            m2, y2 = m.group("m2"), int(m.group("y2"))
            out.append((f"{m1} {y1}", f"{m2} {y2}"))
    # backup: year ranges
    for m in YEAR_RANGE_RE.finditer(text):
        y1 = int(m.group("y1"))
        y2s = m.group("y2")
        if y2s.lower() in ("present","current"):
            out.append((f"Jan {y1}", "Present"))
        else:
            y2 = int(y2s)
            out.append((f"Jan {y1}", f"Jan {y2}"))
    # dedupe preserving order
    seen = set()
    final = []
    for a,b in out:
        key = (a.lower(), b.lower())
        if key not in seen:
            seen.add(key); final.append((a,b))
    return final

def months_gap_chain(periods: List[Tuple[str,str]]) -> List[int]:
    """
    Given periods [('Jun 2016','Aug 2019'), ('Sep 2019','Feb 2021'), ...] sorted by start ascending,
    return list of gaps in months between consecutive periods.
    """
    norm = []
    for a,b in periods:
        sa = _to_y_m(a)
        sb = _to_y_m(b)
        if sa and sb:
            norm.append((sa,sb))
    norm.sort(key=lambda p: p[0])  # by start
    gaps = []
    for i in range(1, len(norm)):
        prev_end = norm[i-1][1]
        curr_start = norm[i][0]
        # if overlap, gap is 0
        gap = max(0, _ym_to_ord(curr_start[0], curr_start[1]) - _ym_to_ord(prev_end[0], prev_end[1]))
        gaps.append(gap)
    return gaps

def _to_y_m(s: str) -> Tuple[int,int] | None:
    s = s.strip()
    if s.lower() in ("present","current"):
        from datetime import datetime
        now = datetime.utcnow()
        return (now.year, now.month)
    m = re.search(rf"{_MONTH}\s+({_YEAR})", s, re.IGNORECASE)
    if m:
        mo = _MONTH_NUM[m.group(1).lower()]
        yr = int(m.group(2))
        return (yr, mo)
    m = re.search(rf"({_YEAR})", s)
    if m:
        return (int(m.group(1)), 1)
    return None
