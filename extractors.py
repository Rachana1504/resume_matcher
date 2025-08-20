# extractors.py
import re, fitz, docx, dateutil.parser
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from spacy.matcher import PhraseMatcher
import spacy

# ---------- NLP / Models ----------
nlp = spacy.load("en_core_web_sm")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# ---------- File Readers ----------
def extract_text(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return "\n".join(page.get_text() for page in fitz.open(path))
    if p.endswith(".docx"):
        return "\n".join(par.text for par in docx.Document(path).paragraphs)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ---------- Skills ----------
def extract_skills(text: str) -> List[str]:
    """
    1) skillNer full matches
    2) High-confidence n-grams
    3) Fallback: frequent NOUN/PROPN lemmas
    4) Last resort: top noun chunks (phrases)
    """
    try:
        annotations = skill_extractor.annotate(text)
        res = annotations.get("results", {})
        out = set(m["doc_node_value"] for m in res.get("full_matches", []))

        for m in res.get("ngram_scored", []):
            if m.get("score", 0) >= 0.85:
                out.add(m["doc_node_value"])

        if not out:
            doc = nlp(text)
            counts: Dict[str, int] = {}
            for t in doc:
                if t.pos_ in ("NOUN", "PROPN") and t.is_alpha and 2 <= len(t.text) <= 30:
                    w = t.lemma_.lower()
                    counts[w] = counts.get(w, 0) + 1
            out = {w for (w, c) in counts.items() if c >= 2}

        if not out:
            doc = nlp(text)
            chunks = [c.text.strip().lower() for c in doc.noun_chunks if 3 <= len(c.text) <= 40]
            for ch in chunks[:25]:
                out.add(ch)

        return list(out)
    except Exception:
        return []

def normalize_skills(skill_list: List[str]) -> set:
    # normalize heavily; also collapse spaces/hyphens so contains() works later
    cleaned = set()
    for s in skill_list:
        s = str(s).strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9]", "", s)
        if s:
            cleaned.add(s)
    return cleaned

# ---------- Sections & Date Utilities ----------
_EDU_HEAD = r"(education|academic|qualification|degree|certification|certifications|education\s*&\s*certifications)"
_EXP_HEAD = r"(experience|work experience|professional experience|employment|employment history|work history|career|career history|internship|internships|work\s*history|experience\s*summary)"
_PROJ_HEAD = r"(project|projects|publications|research|thesis|capstone)"

def extract_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    patt = re.compile(rf"\b({_EDU_HEAD}|{_EXP_HEAD}|{_PROJ_HEAD})\b", re.I)
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if patt.search(line):
            current = line.lower()
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections

SEASON_MONTH_MAP = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}

def parse_date_range(line: str) -> List[datetime]:
    # strip bullets
    line = re.sub(r"[•\u2022\u2023\u25E6\u2043\u2219]", "", line).strip()
    dates: List[datetime] = []

    # Month Year (Jan 2022), Season Year (Fall 2023)
    for m in re.findall(r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{4})", line, re.I):
        try:
            dates.append(dateutil.parser.parse(f"{m[0]} {m[1]}"))
        except Exception:
            pass
    for m in re.findall(r"(Spring|Summer|Fall|Autumn|Winter)\s+(\d{4})", line, re.I):
        try:
            dates.append(datetime(int(m[1]), SEASON_MONTH_MAP.get(m[0].lower(), 1), 1))
        except Exception:
            pass

    if re.search(r"\b(present|current)\b", line, re.I):
        dates.append(datetime.now())

    for sy, ey in re.findall(r"(\d{4})\s*[-–]\s*(\d{4})", line):
        try:
            dates.append(datetime(int(sy), 1, 1)); dates.append(datetime(int(ey), 12, 31))
        except Exception:
            pass

    # MM/YYYY
    for mm, yy in re.findall(r"\b(\d{1,2})/(\d{4})\b", line):
        try:
            dates.append(datetime(int(yy), int(mm), 1))
        except Exception:
            pass

    # Single Year (as part of a range or context)
    for y in re.findall(r"\b(19|20)\d{2}\b", line):
        try:
            dates.append(datetime(int("".join(y)), 1, 1))
        except Exception:
            pass

    # unique & sorted
    uniq = []
    seen = set()
    for d in dates:
        k = (d.year, d.month)
        if k not in seen:
            seen.add(k); uniq.append(d)
    return sorted(uniq)

def extract_periods(lines: List[str]) -> List[Tuple[str, datetime, datetime]]:
    periods: List[Tuple[str, datetime, datetime]] = []
    for line in lines:
        if not line.strip(): continue
        ds = parse_date_range(line)
        if len(ds) >= 2:
            start, end = ds[0], ds[-1]
            text = re.sub(r"([A-Za-z]{3,9})\.?\s+\d{4}", "", line, flags=re.I)
            text = re.sub(r"(Spring|Summer|Fall|Autumn|Winter)\s+\d{4}", "", text, flags=re.I)
            text = re.sub(r"\b(19|20)\d{2}\b", "", text)
            text = re.sub(r"\b(present|current)\b", "", text, flags=re.I)
            text = re.sub(r"[|•\u2022\u2023\u25E6\u2043\u2219]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 3:
                periods.append((text, start, end))

    # dedupe by (name,start y/m)
    seen, uniq = set(), []
    for name, s, e in periods:
        key = (name.lower(), s.year, s.month)
        if key not in seen:
            seen.add(key); uniq.append((name, s, e))
    return sorted(uniq, key=lambda x: x[1])

def extract_periods_anywhere(text: str) -> List[Tuple[str, datetime, datetime]]:
    """Fallback: scan *all* lines for date ranges, independent of sections."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return extract_periods(lines)

# ---------- Gaps / Helpers ----------
def calculate_gaps(periods: List[Tuple[str, datetime, datetime]]):
    if len(periods) < 2: return None
    gaps = []
    for i in range(1, len(periods)):
        p_name, p_start, p_end = periods[i-1]
        c_name, c_start, c_end = periods[i]
        if c_start > p_end:
            gap = (c_start.year - p_end.year)*12 + (c_start.month - p_end.month)
            if gap > 1:
                gaps.append({
                    "between": f"{p_name} → {c_name}",
                    "gap_months": gap,
                    "gap_start": p_end.strftime("%b %Y"),
                    "gap_end": c_start.strftime("%b %Y"),
                })
    return gaps or None

def education_to_first_job_gap(edu_periods, exp_periods):
    if not edu_periods or not exp_periods: return None
    last_edu_end = max(edu_periods, key=lambda x: x[2])[2]
    first_exp_start = min(exp_periods, key=lambda x: x[1])[1]
    gap = (first_exp_start.year - last_edu_end.year)*12 + (first_exp_start.month - last_edu_end.month)
    return max(gap, 0)

def clean_entry_name(entry_text: str) -> str:
    text = re.sub(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}", "", entry_text, flags=re.I)
    text = re.sub(r"(Spring|Summer|Fall|Autumn|Winter)\s+\d{4}", "", text, flags=re.I)
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)
    text = re.sub(r"\b(present|current)\b", "", text, flags=re.I)
    text = re.sub(r"[|•\u2022\u2023\u25E6\u2043\u2219]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"[,\n\r|]", text)
    return (parts[0].strip() if parts else text.strip()) or text.strip()

def is_project(name: str) -> bool:
    return any(k in name.lower() for k in ["project", "publication", "research", "thesis", "capstone"])

def is_education_institution(name: str) -> bool:
    kw = ["university","college","institute","school","academy","polytechnic",
          "gpa","cgpa","bachelor","master","phd","degree","diploma","high school","b.s.","m.s.","b.a.","m.a."]
    t = name.lower()
    return any(k in t for k in kw)

# ---------- Bundle ----------
def extract_resume_data(text: str):
    skills = extract_skills(text)
    sections = extract_sections(text)

    # Education by headings
    edu_lines: List[str] = []
    for k in sections.keys():
        if re.search(_EDU_HEAD, k, re.I):
            edu_lines.extend(sections[k])
    edu = extract_periods(edu_lines)
    edu = [e for e in edu if is_education_institution(e[0]) and not is_project(e[0])]

    # Fallback if empty
    if not edu:
        any_periods = extract_periods_anywhere(text)
        edu = [e for e in any_periods if is_education_institution(e[0])]

    # Deduplicate education by name
    seen_names, edu_filtered = set(), []
    for e in edu:
        nm = clean_entry_name(e[0]).lower()
        if nm not in seen_names:
            seen_names.add(nm)
            edu_filtered.append((clean_entry_name(e[0]), e[1], e[2]))

    # Experience by headings
    exp_lines: List[str] = []
    for k in sections.keys():
        if re.search(_EXP_HEAD, k, re.I):
            exp_lines.extend(sections[k])
    exp = extract_periods(exp_lines)

    # Fallback if empty: anything that's NOT clearly education/project
    if not exp:
        any_periods = extract_periods_anywhere(text)
        exp = [e for e in any_periods if not is_education_institution(e[0]) and not is_project(e[0])]

    gaps_edu = calculate_gaps(edu_filtered)
    gaps_exp = calculate_gaps(exp)
    edu_to_exp_gap = education_to_first_job_gap(edu_filtered, exp)

    return skills, edu_filtered, exp, gaps_edu, gaps_exp, edu_to_exp_gap
