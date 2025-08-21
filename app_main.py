import os
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from matcher import match_resume_to_jds, as_html_table, as_csv

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Resume–JD Matcher (SkillNer)")

# Relaxed CORS for local + Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- routes ---------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # serve the SPA from file (works on Render & local)
    path = os.path.join(APP_DIR, "app.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(
    resume: UploadFile = File(...),
    jd_files: List[UploadFile] = File(...),
):
    # persist to disk so PyMuPDF/docx readers can open them
    r_path = _save_upload(resume)
    jd_paths = [_save_upload(j) for j in jd_files]

    # run match
    rows = match_resume_to_jds(r_path, jd_paths)

    # Return as HTML table (the UI parses & renders cards)
    table_html = as_html_table(rows)
    return HTMLResponse(table_html)

@app.post("/download_csv")
async def download_csv(
    resume: UploadFile = File(...),
    jd_files: List[UploadFile] = File(...),
):
    r_path = _save_upload(resume)
    jd_paths = [_save_upload(j) for j in jd_files]
    rows = match_resume_to_jds(r_path, jd_paths)
    payload = as_csv(rows)
    return StreamingResponse(
        iter([payload]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resume_match_results.csv"},
    )

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

# --------- helpers ---------

def _save_upload(u: UploadFile) -> str:
    suffix = os.path.splitext(u.filename or "")[1].lower() or ".bin"
    fd, path = tempfile.mkstemp(prefix="up_", suffix=suffix, dir=UPLOAD_DIR)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(u.file.read())
    return path
