"""
main.py  —  FastAPI backend
"""

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Career Assistant API",
    version="1.0.0",
    redirect_slashes=False,  # accept both /endpoint and /endpoint/
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Models ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_query:   str
    location:    Optional[str] = ""
    user_id:     Optional[str] = "default"

class QuickScoreRequest(BaseModel):
    resume_text:     str
    job_description: str

class JobSearchRequest(BaseModel):
    query:        str
    profile_text: str
    location:     Optional[str] = ""
    max_results:  Optional[int] = 50


# ── Health & Stats ────────────────────────────────────────────────────────────

@app.get("/api/health")
@app.get("/api/health/")
async def health():
    return {"status": "ok", "service": "Career Assistant API"}


@app.get("/api/stats")
@app.get("/api/stats/")
async def stats():
    try:
        from rag.retriever import RAGRetriever
        retriever = RAGRetriever()
        return retriever.stats()
    except Exception as e:
        return {"resume_chunks": 0, "job_chunks": 0, "error": str(e)}


# ── Upload Resume ─────────────────────────────────────────────────────────────

@app.post("/api/upload-resume")
@app.post("/api/upload-resume/")
async def upload_resume(
    file:    UploadFile = File(...),
    user_id: str        = Form("default"),
):
    if not file.filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, or TXT.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from rag.retriever import RAGRetriever
        retriever = RAGRetriever()
        result    = retriever.ingest_resume(tmp_path, user_id=user_id)
        return {
            "success":        True,
            "n_chunks":       result["n_chunks"],
            "n_achievements": result["n_achievements"],
            "resume_text":    result["raw_text"][:5000],
        }
    except Exception as e:
        raise HTTPException(500, f"Resume parse error: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ── Quick Score ───────────────────────────────────────────────────────────────

@app.post("/api/quick-score")
@app.post("/api/quick-score/")
async def quick_score(req: QuickScoreRequest):
    try:
        from agents.ats_agent import ATSFeedbackAgent
        agent = ATSFeedbackAgent()
        score = agent.quick_score(req.resume_text, req.job_description)
        return {"success": True, "score": score}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── ATS Feedback ──────────────────────────────────────────────────────────────

@app.post("/api/ats-feedback")
@app.post("/api/ats-feedback/")
async def ats_feedback(req: QuickScoreRequest):
    try:
        from agents.ats_agent import ATSFeedbackAgent
        agent  = ATSFeedbackAgent()
        report = agent.analyze(req.resume_text, req.job_description)
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Search Jobs ───────────────────────────────────────────────────────────────

@app.post("/api/search-jobs")
@app.post("/api/search-jobs/")
async def search_jobs(req: JobSearchRequest):
    try:
        from agents.scraper_agent import JobScraperAgent
        agent   = JobScraperAgent()
        # run async search directly since we are inside FastAPI event loop
        results = await agent.search_and_rank(
            query=req.query,
            profile_text=req.profile_text,
            location=req.location,
            max_results=req.max_results,
        )
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Full Pipeline ─────────────────────────────────────────────────────────────

@app.post("/api/analyze")
@app.post("/api/analyze/")
async def analyze(req: AnalyzeRequest):
    if len(req.resume_text.strip()) < 50:
        raise HTTPException(400, "Resume text too short (min 50 chars)")
    if len(req.job_query.strip()) < 3:
        raise HTTPException(400, "Job query too short")
    try:
        from graph.career_graph import CareerAssistant
        assistant = CareerAssistant()
        report = assistant.run(
            resume_text=req.resume_text,
            job_query=req.job_query,
            location=req.location,
            user_id=req.user_id,
        )
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {str(e)}")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app.main:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=True,
    )