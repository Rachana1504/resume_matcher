# app_main.py
import io
import csv
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from matcher import match_resume_to_jds
from jd_cache import load_or_build_jd_cache, build_jd_cache_from_uploads

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Resume–JD Matching Plugin")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _serve_app_html() -> HTMLResponse:
    html = (APP_DIR / "app.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})

@app.get("/", response_class=HTMLResponse)
def index():
    return _serve_app_html()

@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    # legacy path, same UI
    return _serve_app_html()

@app.get("/healthz", response_class=PlainTextResponse)
def health():
    return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})

# Build fallback JD cache from local folder (once per process)
jd_cache_fallback = load_or_build_jd_cache(
    jd_dir=str(APP_DIR / "Dummy_data" / "JDS"),
    cache_path=str(APP_DIR / "Dummy_data" / "jd_cache.json"),
)

def _jd_cache_from_uploads(jd_files: Optional[List[UploadFile]]):
    if not jd_files:
        return None
    named_bytes = []
    for f in jd_files:
        named_bytes.append((f.filename, f.file.read()))
    return build_jd_cache_from_uploads(named_bytes)

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(
    resume: UploadFile = File(...),
    jd_files: Optional[List[UploadFile]] = File(None),
):
    # Save resume to disk
    resume_path = str(UPLOAD_DIR / resume.filename)
    with open(resume_path, "wb") as out:
        shutil.copyfileobj(resume.file, out)

    # Prefer uploaded JDs; fall back to prebuilt cache
    jd_cache = _jd_cache_from_uploads(jd_files) or jd_cache_fallback

    # Compute matches
    results = match_resume_to_jds(resume_path, jd_cache)

    # Render as simple table (front-end maps this into cards)
    def _join(xs): return ", ".join(xs or [])
    def _periods_html(periods):
        if not periods: return "—"
        return "<br>".join(f"{p.get('entry','')} ({p.get('start','')} — {p.get('end','')})" for p in periods)
    def _gaps_html(gaps):
        if not gaps: return "None"
        return "<br>".join(f"{g.get('between','')} – {g.get('gap_months','')} months" for g in gaps)

    rows = []
    for r in results:
        rows.append(f"""
        <tr>
          <td>{r.get('jd_file','')}</td>
          <td>{r.get('similarity_score_percent',0)}%</td>
          <td>{r.get('resume_location','Not Mentioned')}</td>
          <td>{r.get('jd_location','Not Mentioned')}</td>
          <td>{_join(r.get('matched_skills'))}</td>
          <td>{_join(r.get('missing_skills'))}</td>
          <td>{r.get('education_to_first_job_gap_months','N/A')} months</td>
          <td>{_periods_html(r.get('education_periods'))}</td>
          <td>{_periods_html(r.get('experience_periods'))}</td>
          <td>{_gaps_html(r.get('education_gaps'))}</td>
          <td>{_gaps_html(r.get('experience_gaps'))}</td>
        </tr>
        """)

    table = f"""
    <table>
      <tr>
        <th>JD File</th>
        <th>Match %</th>
        <th>Resume Location</th>
        <th>JD Location</th>
        <th>Matched Skills</th>
        <th>Missing Skills</th>
        <th>Edu → First Job Gap</th>
        <th>Education Periods</th>
        <th>Experience Periods</th>
        <th>Education Gaps</th>
        <th>Experience Gaps</th>
      </tr>
      {''.join(rows)}
    </table>
    """
    return HTMLResponse(table, headers={"Cache-Control": "no-store"})

@app.post("/download_csv")
async def download_csv(
    resume: UploadFile = File(...),
    jd_files: Optional[List[UploadFile]] = File(None),
):
    resume_path = str(UPLOAD_DIR / resume.filename)
    with open(resume_path, "wb") as out:
        shutil.copyfileobj(resume.file, out)

    jd_cache = _jd_cache_from_uploads(jd_files) or jd_cache_fallback
    results = match_resume_to_jds(resume_path, jd_cache)

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow([
        "JD File","Match %","Resume Location","JD Location",
        "Matched Skills","Missing Skills",
        "Edu → First Job Gap","Education Periods","Experience Periods",
        "Education Gaps","Experience Gaps"
    ])
    def _p(periods):
        return ", ".join(f"{p.get('entry','')} ({p.get('start','')} — {p.get('end','')})" for p in periods or [])
    def _g(gaps):
        return ", ".join(f"{g.get('between','')} – {g.get('gap_months','')} months" for g in gaps or [])
    for r in results:
        w.writerow([
            r.get("jd_file",""),
            r.get("similarity_score_percent",0),
            r.get("resume_location",""),
            r.get("jd_location",""),
            ", ".join(r.get("matched_skills") or []),
            ", ".join(r.get("missing_skills") or []),
            r.get("education_to_first_job_gap_months",""),
            _p(r.get("education_periods")),
            _p(r.get("experience_periods")),
            _g(r.get("education_gaps")),
            _g(r.get("experience_gaps")),
        ])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=resume_match_results.csv",
            "Cache-Control": "no-store",
        },
    )
