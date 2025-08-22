# app_main.py
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Your existing matcher helpers
from matcher import match_resume_to_jds, as_html_table, as_csv  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Resume↔JD Matching Plugin", version="ui v2025-08-21")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _normalize(files: Optional[List[UploadFile]] | Optional[UploadFile]) -> List[UploadFile]:
    """
    Accept a single UploadFile or a list[UploadFile] (or None) and
    return a clean list[UploadFile].
    """
    if files is None:
        return []
    if isinstance(files, list):
        return [f for f in files if f is not None]
    return [files]

async def _to_docs(files: List[UploadFile]) -> List[dict]:
    """
    Turn UploadFile objects into the simple structure many matcher functions use:
       [{"name": "foo.pdf", "bytes": b"..."}]
    (Your matcher may only need the bytes or text; adapt there as desired.)
    """
    docs: List[dict] = []
    for f in files:
        content = await f.read()
        docs.append({"name": f.filename or "file", "bytes": content})
    return docs

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve the single-page UI."""
    try:
        with open("app.html", "r", encoding="utf-8") as fp:
            return HTMLResponse(fp.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>Resume–JD Matching Plugin</h1><p>app.html not found next to app_main.py.</p>",
            status_code=200,
        )

@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"

@app.post("/match")
async def match_endpoint(
    # Resumes: allow single or list and multiple possible field names
    resume_files: Optional[List[UploadFile]] = File(None),
    resume_file: Optional[UploadFile] = File(None),
    resumes: Optional[List[UploadFile]] = File(None),

    # JDs: allow single or list and multiple possible field names
    jd_files: Optional[List[UploadFile]] = File(None),
    jd_file: Optional[UploadFile] = File(None),
    jds: Optional[List[UploadFile]] = File(None),

    # Filters / options from your UI
    min_match: float = Form(0.0),
    min_skills: int = Form(0),
    sort_by: str = Form("score_desc"),  # "score_desc" | "score_asc" | "name" (adapt as your matcher expects)
):
    """
    Accepts resumes and JDs as single or multiple files.
    Normalizes everything to lists before passing to the matcher.
    """

    # Normalize all possible form fields
    resume_list: List[UploadFile] = (
        _normalize(resume_files) + _normalize(resume_file) + _normalize(resumes)
    )
    jd_list: List[UploadFile] = (
        _normalize(jd_files) + _normalize(jd_file) + _normalize(jds)
    )

    if not resume_list:
        raise HTTPException(status_code=400, detail="No resume files were uploaded.")
    if not jd_list:
        raise HTTPException(status_code=400, detail="No job description files were uploaded.")

    # Read bytes now (allows matcher to be pure / independent of Starlette UploadFile)
    resume_docs = await _to_docs(resume_list)
    jd_docs = await _to_docs(jd_list)

    # Call your existing matcher. Different projects wire this differently,
    # so we try kwargs first and fall back to args if needed.
    try:
        results = match_resume_to_jds(
            resumes=resume_docs,
            jds=jd_docs,
            min_match=min_match,
            min_skills=min_skills,
            sort_by=sort_by,
        )
    except TypeError:
        # Old signature compatibility:
        results = match_resume_to_jds(resume_docs, jd_docs, min_match, min_skills, sort_by)

    # Many UIs expect both table HTML and a CSV payload; adapt as your frontend expects
    try:
        table_html = as_html_table(results)
    except Exception:
        table_html = ""

    try:
        csv_text = as_csv(results)
    except Exception:
        csv_text = ""

    return JSONResponse(
        {
            "ok": True,
            "count": len(results) if hasattr(results, "__len__") else None,
            "table_html": table_html,
            "csv": csv_text,
            "data": results,  # keep the raw results too (your frontend may use them)
        }
    )
