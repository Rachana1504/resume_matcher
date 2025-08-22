# extractors.py
import re, fitz, docx, dateutil.parser
from datetime import datetime

# --- robust SkillNer imports (works across package layouts) ---
try:
    # Preferred explicit paths
    from skillNer.general_params import SKILL_DB
    from skillNer.skill_extractor_class import SkillExtractor
except Exception:
    # Fallbacks for older/alternate layouts
    try:
        from skillNer import SkillExtractor  # type: ignore
        from skillNer.general_params import SKILL_DB  # type: ignore
    except Exception as e:  # last resort: give a clearer error
        raise ImportError(
            "Could not import SkillExtractor from skillNer. "
            "Ensure 'skillNer' is installed and available, e.g. pin skillNer==1.0.3."
        ) from e

from spacy.matcher import PhraseMatcher
import spacy

nlp = spacy.load("en_core_web_sm")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# ------------------ File Readers ------------------
def extract_text(path):
    if path.endswith(".pdf"):
        return "\n".join([p.get_text() for p in fitz.open(path)])
    elif path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

# ------------------ Skill extraction (SkillNer-only) ------------------
def extract_skills(text: str) -> list[str]:
    """Use SkillNer to extract unique, normalized skill names from free text."""
    doc = nlp(text)
    out = skill_extractor.annotate(doc)
    # out["results"]["full_matches"] and out["results"]["ngram_scored"]
    skills = set()

    # full matches
    for m in out.get("results", {}).get("full_matches", []):
        name = m.get("doc_node_value") or m.get("skill_name")
        if name:
            skills.add(name.strip())

    # n-gram scored (SkillNer’s fuzzy/partial)
    for m in out.get("results", {}).get("ngram_scored", []):
        name = m.get("doc_node_value") or m.get("skill_name")
        if name:
            skills.add(name.strip())

    # Deduplicate case-insensitively while keeping a pretty form
    dedup = {}
    for s in skills:
        k = s.lower()
        if k not in dedup:
            dedup[k] = s
    return sorted(dedup.values(), key=str.lower)

# ------------------ Experience / Education extraction helpers ------------------
MONTH = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9,
    "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

SEASON = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}

DATE_WORD = r"(?:\d{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|spring|summer|fall|autumn|winter)"

def _to_dt(month_word: str | None, year: int | None) -> datetime | None:
    if not year:
        return None
    if month_word:
        m = MONTH.get(month_word.lower())
        if not m:
            m = SEASON.get(month_word.lower(), 1)
    else:
        m = 1
    try:
        return datetime(year, m, 1)
    except Exception:
        return None

def parse_date_range(line: str) -> tuple[datetime | None, datetime | None]:
    """
    Parse flexible ranges like:
      - Jan 2020 – Mar 2022
      - 2021 to Present
      - Summer 2019 — Winter 2020
      - 2018-2020
    Returns (start_dt, end_dt) where None means unknown/present.
    """
    s = line.strip()

    # normalize separators
    s = re.sub(r"[–—\-to]+", "-", s, flags=re.I)

    # capture potential month + year tokens
    tokens = re.findall(
        rf"({DATE_WORD})\s*(\d{{4}})?", s, flags=re.I
    )

    # find "present/current"
    if re.search(r"\b(present|current|now)\b", s, flags=re.I):
        end_year = None
        end_month = None
    else:
        # try to read the last year in the string as end
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", s)]
        end_year = years[-1] if years else None
        end_month = None
        # if we saw an explicit month right before last year, use it
        m = re.search(rf"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|spring|summer|fall|autumn|winter)\s+(19\d{{2}}|20\d{{2}})\D*$", s, flags=re.I)
        if m:
            end_month = m.group(1)

    # try to read the first year in the string as start
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", s)]
    start_year = years[0] if years else None

    # if there is a month before the first year, use it
    m = re.search(
        rf"^(?:.*?)(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|spring|summer|fall|autumn|winter)\s+(19\d{{2}}|20\d{{2}})",
        s, flags=re.I
    )
    start_month = m.group(1) if m else None

    return _to_dt(start_month, start_year), _to_dt(end_month, end_year)

def find_experience_periods(text: str) -> list[str]:
    """
    Extract all lines that look like job date ranges.
    """
    lines = [re.sub(r"\s+", " ", L).strip() for L in text.splitlines()]
    res: list[str] = []
    for L in lines:
        if re.search(r"\b(19\d{2}|20\d{2})\b", L) and re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|spring|summer|fall|autumn|winter|present|current|now)\b", L, flags=re.I):
            # require that parse_date_range recognizes it
            sdt, edt = parse_date_range(L)
            if sdt or edt:
                res.append(L)
    # de-dupe while preserving order
    seen = set()
    out = []
    for r in res:
        k = r.lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out

def find_education_periods(text: str) -> list[str]:
    """
    Education date lines – similar heuristic to experience.
    """
    edu_tokens = r"(university|college|bachelor|master|mba|ph\.?d|school|institute|academy|degree|b\.?tech|b\.?e\.|m\.?s\.|m\.?tech)"
    lines = [re.sub(r"\s+", " ", L).strip() for L in text.splitlines()]
    res: list[str] = []
    for L in lines:
        if re.search(edu_tokens, L, flags=re.I) and re.search(r"\b(19\d{2}|20\d{2})\b", L):
            sdt, edt = parse_date_range(L)
            if sdt or edt:
                res.append(L)
    # de-dupe while preserving order
    seen = set()
    out = []
    for r in res:
        k = r.lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out

# ------------------ Public API ------------------
def parse_resume(path: str) -> dict:
    """
    Returns {
        "text": <full text>,
        "skills": [list of SkillNer skill names],
        "experience_periods": [strings],
        "education_periods": [strings],
    }
    """
    text = extract_text(path)
    return {
        "text": text,
        "skills": extract_skills(text),
        "experience_periods": find_experience_periods(text),
        "education_periods": find_education_periods(text),
    }

def parse_jd(path: str) -> dict:
    """
    Parse a single JD file (pdf/docx/txt) and return text + extracted skills.
    """
    text = extract_text(path)
    return {"text": text, "skills": extract_skills(text)}
