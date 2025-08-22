# app_main.py
import os
import io
import csv
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ✅ only import the matcher we actually need
from matcher import match_resume_to_jds
from jd_cache import load_or_build_jd_cache, build_jd_cache_from_uploads

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
RESUME_DIR = UPLOAD_DIR / "resumes"
JD_DIR = UPLOAD_DIR / "jds"
for d in (UPLOAD_DIR, RESUME_DIR, JD_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Resume–JD Matcher")

# CORS so the page works on Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_uploads(files: List[UploadFile], target_dir: Path) -> list[Path]:
    saved: list[Path] = []
    for f in files:
        # Normalize filename; keep only last path part
        name = os.path.basename(f.filename or "upload")
        dest = target_dir / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    return saved


def _serve_app_html() -> HTMLResponse:
    # Always read app.html next to this file (prevents wrong working-dir issues)
    html = (APP_DIR / "app.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def index():
    return _serve_app_html()


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    # Legacy route — same UI
    return _serve_app_html()


# ---------- Upload endpoints (single or multiple) ----------
@app.post("/upload_resume")
async def upload_resume(resume: UploadFile = File(...)):
    _save_uploads([resume], RESUME_DIR)
    return {"ok": True, "files": [resume.filename]}


@app.post("/upload_resumes")
async def upload_resumes(resume_files: List[UploadFile] = File(...)):
    saved = _save_uploads(resume_files, RESUME_DIR)
    return {"ok": True, "files": [p.name for p in saved]}


@app.post("/upload_jd")
async def upload_jd(jd: UploadFile = File(...)):
    _save_uploads([jd], JD_DIR)
    return {"ok": True, "files": [jd.filename]}


@app.post("/upload_jds")
async def upload_jds(jd_files: List[UploadFile] = File(...)):
    saved = _save_uploads(jd_files, JD_DIR)
    return {"ok": True, "files": [p.name for p in saved]}


# ---------- Matching + CSV ----------
@app.post("/match")
async def match_endpoint(
    min_match_pct: float = 0.0,
    min_skills: int = 0,
    sort_by: str = "score_desc",  # "score_desc" | "score_asc" | "skills_desc" | "skills_asc"
):
    """
    Called by the UI after uploads. Runs matcher and returns rendered HTML (card list).
    """
    # Build/refresh JD cache (fast if JDs unchanged)
    jd_cache = load_or_build_jd_cache(JD_DIR)

    # Collect resume and JD paths
    resumes = sorted([p for p in RESUME_DIR.iterdir() if p.is_file()])
    jds = sorted([p for p in JD_DIR.iterdir() if p.is_file()])

    if not resumes:
        return {"html": '<div class="warn">Please upload at least one resume.</div>'}
    if not jds:
        return {"html": '<div class="warn">Please upload at least one job description.</div>'}

    # Run matching (this returns a list of dicts per JD)
    results = match_resume_to_jds(resumes, jds, jd_cache=jd_cache)

    # Apply filters
    filtered = []
    for r in results:
        if r.get("match_pct", 0.0) >= min_match_pct and r.get("matched_skill_count", 0) >= min_skills:
            filtered.append(r)

    # Sort
    key_map = {
        "score_desc": lambda x: (-x.get("match_pct", 0.0), -x.get("matched_skill_count", 0)),
        "score_asc":  lambda x: (x.get("match_pct", 0.0), x.get("matched_skill_count", 0)),
        "skills_desc": lambda x: (-x.get("matched_skill_count", 0), -x.get("match_pct", 0.0)),
        "skills_asc":  lambda x: (x.get("matched_skill_count", 0), x.get("match_pct", 0.0)),
    }
    filtered.sort(key=key_map.get(sort_by, key_map["score_desc"]))

    # Render simple HTML cards
    cards = []
    for r in filtered:
        ms = r.get("matched_skills", [])
        ms_str = ", ".join(ms) if ms else "—"
        edu_periods = r.get("education_periods", [])
        exp_periods = r.get("experience_periods", [])
        edu_gaps = r.get("education_gaps", [])
        exp_gaps = r.get("experience_gaps", [])
        first_gap = r.get("edu_first_job_gap", None)

        # Period format helper
        def fmt_period(p):
            s = p.get("start")
            e = p.get("end")
            return f"{s} — {e}"

        edu_html = "<br>".join(
            (p.get("title", "") + " " + p.get("org", "") + " " + fmt_period(p)).strip()
            for p in edu_periods
        ) or "—"

        exp_html = "<br>".join(
            (p.get("title", "") + " | " + p.get("org", "") + " " + fmt_period(p)).strip()
            for p in exp_periods
        ) or "—"

        edu_gaps_html = "<br>".join(edu_gaps) or "None"
        exp_gaps_html = "<br>".join(exp_gaps) or "None"
        first_gap_html = f"{first_gap} months" if first_gap is not None else "—"

        cards.append(f"""
        <div class="card">
          <div class="card-title">{r.get('jd_name','(JD)')} ({r.get('match_pct',0.0):.2f}%)</div>
          <div><b>Matched Skills:</b> {ms_str}</div>
          <details open><summary><b>Education Periods</b></summary>{edu_html}</details>
          <details open><summary><b>Experience Periods</b></summary>{exp_html}</details>
          <details><summary><b>Education Gaps</b></summary>{edu_gaps_html}</details>
          <details><summary><b>Experience Gaps</b></summary>{exp_gaps_html}</details>
          <div><b>Edu → First Job Gap:</b> {first_gap_html}</div>
        </div>
        """.strip())

    if not cards:
        html = '<div class="warn">No matches after filters.</div>'
    else:
        html = "\n".join(cards)

    return {"html": html}


@app.get("/download_csv")
def download_csv():
    """CSV of the latest uploads’ matches."""
    jd_cache = load_or_build_jd_cache(JD_DIR)
    resumes = sorted([p for p in RESUME_DIR.iterdir() if p.is_file()])
    jds = sorted([p for p in JD_DIR.iterdir() if p.is_file()])
    if not (resumes and jds):
        return StreamingResponse(io.BytesIO(b"No data"), media_type="text/plain")

    results = match_resume_to_jds(resumes, jds, jd_cache=jd_cache)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "JD File",
        "Match %",
        "Matched Skills (comma-separated)",
        "Matched Skill Count",
        "Education Periods",
        "Experience Periods",
        "Education Gaps",
        "Experience Gaps",
        "Edu→First Job Gap (months)",
    ])
    for r in results:
        writer.writerow([
            r.get("jd_name", ""),
            f"{r.get('match_pct', 0.0):.2f}",
            ", ".join(r.get("matched_skills", [])),
            r.get("matched_skill_count", 0),
            " | ".join(f"{p.get('title','')} {p.get('org','')} {p.get('start','')} — {p.get('end','')}"
                       for p in r.get("education_periods", [])),
            " | ".join(f"{p.get('title','')} | {p.get('org','')} {p.get('start','')} — {p.get('end','')}"
                       for p in r.get("experience_periods", [])),
            " | ".join(r.get("education_gaps", [])),
            " | ".join(r.get("experience_gaps", [])),
            r.get("edu_first_job_gap", ""),
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8"))
    return StreamingResponse(
        data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="matches.csv"'},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app_main:app", host="0.0.0.0", port=port, reload=False)
