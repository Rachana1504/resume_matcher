from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

def _read_text_any(p: Path) -> str:
    s = p.suffix.lower()
    if s == ".pdf":
        import fitz
        doc = fitz.open(p)
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    if s == ".docx":
        import docx2txt
        return docx2txt.process(str(p)) or ""
    return p.read_text(encoding="utf-8", errors="ignore")

def load_or_build_jd_cache(jd_dir: str, cache_path: str) -> Dict[str, dict]:
    jd_dir_path = Path(jd_dir)
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    cache: Dict[str, dict] = {}
    if jd_dir_path.exists():
        for p in jd_dir_path.iterdir():
            if p.is_file() and p.suffix.lower() in {".pdf",".docx",".txt"}:
                cache[p.name] = {"text": _read_text_any(p), "location": ""}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return cache

def build_jd_cache_from_uploads(named_bytes: List[Tuple[str, bytes]]) -> Dict[str, dict]:
    tmpdir = Path("/tmp/jd_uploads")
    tmpdir.mkdir(exist_ok=True, parents=True)
    out: Dict[str, dict] = {}
    for name, data in named_bytes:
        p = tmpdir / name
        p.write_bytes(data)
        out[name] = {"text": _read_text_any(p), "location": ""}
    return out

# --------- Fast text accessor (added) ----------
from functools import lru_cache

@lru_cache(maxsize=512)
def get_jd_text_fast(path_str: str) -> str:
    """Use extractors.extract_text_fast if present; else fall back."""
    try:
        from extractors import extract_text_fast
        return extract_text_fast(path_str)
    except Exception:
        from extractors import extract_text
        return extract_text(path_str)
