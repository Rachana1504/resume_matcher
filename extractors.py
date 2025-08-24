import re
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF
import docx
from dateutil import parser as dparser
from datetime import datetime as _DT_
import os  # (added for caching below)

# ---------- Lazy SkillNer (no predefined keyword lists) ----------
@lru_cache(maxsize=1)
def _lazy_skill_extractor():
    import spacy
    from spacy.matcher import PhraseMatcher
    from skillNer.skill_extractor_class import SkillExtractor
    from skillNer.general_params import SKILL_DB
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
    return SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# ---------- File Readers ----------
def extract_text(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        doc = fitz.open(path)
        try:
            return "\n".join([page.get_text() for page in doc])
        finally:
            doc.close()
    if p.endswith(".docx"):
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ----- Cached file key & cached extract (added, non-invasive) -----
def _file_cache_key(path: str) -> tuple:
    try:
        st = os.stat(path)
        return (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except Exception:
        return (os.path.abspath(path), 0, 0)

@lru_cache(maxsize=512)
def extract_text_cached(key: tuple) -> str:
    return extract_text(key[0])

def extract_text_fast(path: str) -> str:
    return extract_text_cached(_file_cache_key(path))

# ---------- Skills (SkillNer only) ----------
def extract_skills(text: str) -> List[str]:
    try:
        se = _lazy_skill_extractor()
        ann = se.annotate(text) or {}
        # Prefer "results" → "full_matches" if available
        results = ann.get("results", {})
        full = results.get("full_matches") or []
        # fallbacks: some SkillNer versions store flat "results" list
        if isinstance(results, list):
            full = results
        vals = []
        for it in full:
            if isinstance(it, dict):
                vals.append(it.get("doc_node_value") or it.get("skill_name") or it.get("skill") or it.get("label") or "")
            else:
                vals.append(str(it))
        return sorted({s.strip() for s in vals if s})
    except Exception:
        return []

# ----- FAST skills (added, optional) -----
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\+\.#-]{1,}")
STOP = set("""
a an the and or for with of on in to from by about into over than after before during under again further then once here there all any both each few more most other some such no nor not only own same so too very
""".split())

@lru_cache(maxsize=512)
def extract_skills_fast(text: str) -> list[str]:
    tokens = _WORD.findall(text or "")
    cand = []
    for t in tokens:
        k = t.strip().strip(".,;:()[]{}").lower()
        if not k or k in STOP:
            continue
        if any(ch.isupper() for ch in t) or any(ch.isdigit() for ch in t) or len(k) >= 3:
            cand.append(t.strip())
    seen = set()
    out = []
    for c in cand:
        ck = re.sub(r"[^a-z0-9]+", "", c.lower())
        if ck and ck not in seen:
            seen.add(ck)
            out.append(c)
    return out

def normalize_skills(skills: List[str]) -> set:
    out = set()
    for s in skills or []:
        k = re.sub(r"[^a-z0-9]+", "", s.lower()).strip()
        if k:
            out.add(k)
    return out

def clean_entry_name(s: str) -> str:
    s = re.sub(r"[•\u2022\u2023\u25E6\u2043\u2219]", "", s or "")
    s = re.sub(r"\s+", " ", s).strip(" -–—|\t")
    return s.strip()

# ---------- Multi-range Date Parsing ----------
MONTHS_RE = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
RANGE_RE = re.compile(
    rf"(?P<start>(?:{MONTHS_RE}\s+\d{{4}})|(?:\d{{1,2}}[/-]\d{{4}})|(?:\d{{4}}))\s*(?:-|–|—|to|until|through|thru)\s*(?P<end>(?:{MONTHS_RE}\s+\d{{4}})|(?:\d{{1,2}}[/-]\d{{4}})|(?:\d{{4}})|Present|Current|Now)",
    re.IGNORECASE
)

def _parse_date(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    if re.fullmatch(r"(?i)present|current|now", s):
        return datetime(9999, 1, 1)
    try:
        dt = dparser.parse(s, default=datetime(1900, 1, 1), fuzzy=True, dayfirst=False)
        return dt.replace(day=1)
    except Exception:
        return None

def parse_date_range(line: str) -> List[Tuple[Tuple[int,int], datetime, datetime]]:
    out = []
    for m in RANGE_RE.finditer(line or ""):
        start = _parse_date(m.group("start"))
        end   = _parse_date(m.group("end"))
        if start and end:
            out.append((m.span(), start, end))
    return out

def extract_periods(lines: List[str]) -> List[Tuple[str, datetime, datetime]]:
    periods: List[Tuple[str, datetime, datetime]] = []
    prev_nonempty = ""
    for line in lines or []:
        if not (line and line.strip()):
            continue
        ranges = parse_date_range(line)
        if not ranges:
            prev_nonempty = line.strip()
            continue
        # Handle ALL date ranges found in a single line
        for span, start, end in ranges:
            before = line[:span[0]].strip()
            after  = line[span[1]:].strip()
            entry  = before or after or prev_nonempty
            entry  = clean_entry_name(entry) or "Experience"
            periods.append((entry, start, end))
        prev_nonempty = line.strip()

    # Dedup + sort
    seen, uniq = set(), []
    for entry, start, end in periods:
        key = (entry.lower(), start.year, start.month, end.year, end.month)
        if key not in seen:
            seen.add(key)
            uniq.append((entry, start, end))
    uniq.sort(key=lambda x: (x[1], x[2]))
    return uniq

# ---------- Gaps & Aggregation ----------
def _months_between(a: datetime, b: datetime) -> int:
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))

def calculate_gaps(periods: List[Tuple[str, datetime, datetime]]) -> List[Dict[str, Any]]:
    gaps = []
    if not periods:
        return gaps
    periods_sorted = sorted(periods, key=lambda x: x[1])
    for i in range(len(periods_sorted) - 1):
        end_a   = periods_sorted[i][2]
        start_b = periods_sorted[i + 1][1]
        gap = _months_between(end_a, start_b)
        if gap > 0:
            gaps.append({"between": f"{periods_sorted[i][0]} → {periods_sorted[i+1][0]}", "gap_months": gap})
    return gaps

def education_to_first_job_gap(edu, exp) -> int | None:
    if not edu or not exp:
        return None
    last_edu_end    = max(e[2] for e in edu)
    first_job_start = min(e[1] for e in exp)
    return _months_between(last_edu_end, first_job_start)

HEADERS = ["education","experience","work experience","professional experience","projects","skills","certifications","achievements"]

# -------- Season Date Support (already integrated) --------
_SEASON_TO_MONTH = {
    "winter": 1,
    "spring": 2,
    "summer": 6,
    "fall":   9,
    "autumn": 9,
}
_SEASON_TOKEN_RE = re.compile(r"(?i)\b(winter|spring|summer|fall|autumn)\s+(\d{4})\b")

def _normalize_season_terms(_s: str) -> str:
    def _repl(m: re.Match) -> str:
        season = m.group(1).lower()
        year   = m.group(2)
        month  = _SEASON_TO_MONTH.get(season, 1)
        _MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        month_name = _MONTH_NAMES[month-1]
        return f"{month_name} {year}"
    return _SEASON_TOKEN_RE.sub(_repl, _s or "")

try:
    _ORIG__PARSE_DATE = _parse_date
except NameError:
    _ORIG__PARSE_DATE = None

if _ORIG__PARSE_DATE is not None:
    def _parse_date(s: str):  # noqa: F811
        s_norm = _normalize_season_terms(s)
        dt = _ORIG__PARSE_DATE(s_norm)
        if dt is not None:
            return dt
        m = _SEASON_TOKEN_RE.search(s or "")
        if m:
            season = m.group(1).lower()
            year   = int(m.group(2))
            month  = _SEASON_TO_MONTH.get(season, 1)
            try:
                return _DT_(year, month, 1)
            except Exception:
                pass
        return _ORIG__PARSE_DATE(s)

def _split_sections(text: str) -> Dict[str, List[str]]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    sections: Dict[str, List[str]] = {}
    current = "misc"; sections[current] = []
    for ln in lines:
        low = ln.strip().lower()
        if low in HEADERS:
            current = low
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(ln)
    return sections

def is_education_institution(s: str) -> bool:
    return bool(re.search(r"university|college|institute|school|bachelor|master|bsc|msc|ba|ma|phd|diploma", s, re.I))

def extract_resume_data(text: str):
    sections = _split_sections(text)
    skills = extract_skills(text)

    edu_lines = list(sections.get("education", []))
    edu_lines += [ln for ln in sections.get("misc", []) if is_education_institution(ln)]
    edu = extract_periods(edu_lines)

    exp_lines = sections.get("experience", []) + sections.get("work experience", []) + sections.get("professional experience", [])
    if not exp_lines:
        all_lines = [ln for ln in text.splitlines() if ln.strip()]
        exp_lines = [ln for ln in all_lines if ln not in edu_lines]
    exp = extract_periods(exp_lines)

    gaps_edu = calculate_gaps(edu)
    gaps_exp = calculate_gaps(exp)
    edu_to_exp_gap = education_to_first_job_gap(edu, exp)

    return skills, edu, exp, gaps_edu, gaps_exp, edu_to_exp_gap
