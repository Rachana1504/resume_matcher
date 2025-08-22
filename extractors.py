# extractors.py
from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import spacy

# ---------- spaCy lazy loader ----------
@lru_cache(maxsize=1)
def get_nlp():
    # small model to keep memory low on Render
    return spacy.load("en_core_web_sm")

# ---------- SkillNer lazy loader ----------
class SkillBackend:
    def __init__(self):
        self.ready = False
        self.extractor = None

    def load(self):
        if self.ready:
            return
        # Try the canonical imports for skillNer==1.0.3
        try:
            from skillNer.general_params import SKILL_DB  # type: ignore
            from skillNer.skill_extractor_class import SkillExtractor  # type: ignore
            from spacy.matcher import PhraseMatcher  # local import to avoid startup cost
        except Exception as e:
            # Graceful degradation: we keep running without skills rather than crashing
            # (UI will show no matched skills).
            self.ready = False
            self.extractor = None
            return
        nlp = get_nlp()
        self.extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
        self.ready = True

    def extract(self, text: str) -> List[str]:
        if not self.ready:
            self.load()
        if not self.ready or self.extractor is None:
            return []  # no predefined keywords; simply no skill output if SkillNer missing
        ann = self.extractor.annotate(text or "")
        # Collect normalized skill strings from both dicts
        skills = set()
        for m in ann.get("results", {}).get("full_matches", []):
            v = (m.get("doc_node_value") or "").strip().lower()
            if v: skills.add(v)
        for m in ann.get("results", {}).get("ngram_scored", []):
            v = (m.get("doc_node_value") or "").strip().lower()
            if v: skills.add(v)
        return sorted(skills)

skill_backend = SkillBackend()

def extract_skills(text: str) -> List[str]:
    """Extract skills with SkillNer (no predefined keyword lists)."""
    return skill_backend.extract(text or "")

# ---------- Dates / periods ----------
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
DATE_RE = re.compile(
    rf"(?P<start>(?:{MONTHS})\s+\d{{4}}|\d{{2}}/\d{{4}}|\d{{4}})\s*(?:-|to|–|—|—>)\s*(?P<end>(?:{MONTHS})\s+\d{{4}}|\d{{2}}/\d{{4}}|\d{{4}}|present|current|ongoing)",
    re.IGNORECASE
)

def extract_periods(text: str) -> List[Dict[str, str]]:
    """Returns all detected periods in order of appearance."""
    periods: List[Dict[str, str]] = []
    for m in DATE_RE.finditer(text or ""):
        start = m.group("start")
        end = m.group("end")
        snippet = text[max(0, m.start()-80): m.end()+80].splitlines()[0][:160]
        periods.append({"entry": snippet.strip(), "start": start, "end": end})
    return periods

def split_resume_sections(text: str) -> Tuple[str, str]:
    """Very light heuristic split into 'education' / 'experience' parts."""
    t = (text or "").lower()
    edu_idx = t.find("education")
    exp_idx = max(t.find("experience"), t.find("professional experience"))
    if edu_idx == -1 and exp_idx == -1:
        return "", text
    if edu_idx != -1 and (exp_idx == -1 or edu_idx < exp_idx):
        return text[edu_idx:exp_idx] if exp_idx != -1 else text[edu_idx:], text
    return text[:exp_idx], text[exp_idx:]
