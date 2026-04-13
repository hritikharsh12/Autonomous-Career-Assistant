"""
career_graph.py
LangGraph orchestration — wires all agents into a stateful graph:

  [START]
     │
     ▼
  ingest_resume ──► match_jobs ──► ats_analysis ──► feedback_loop ──► [END]
                        ▲                │
                        └────────────────┘ (re-rank after ATS boost)
"""

import asyncio
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END, START

from agents.ats_agent     import ATSFeedbackAgent
from agents.scraper_agent import JobScraperAgent
from rag.retriever        import RAGRetriever


# ── Graph State ───────────────────────────────────────────────────────────────

class CareerState(TypedDict):
    # Inputs
    resume_text:       str
    job_query:         str
    location:          str
    user_id:           str

    # Intermediate
    ingested:          bool
    job_results:       List[Dict[str, Any]]
    ats_report:        Dict[str, Any]
    feedback_loop_count: int

    # Outputs
    enhanced_resume:   str
    skill_gaps:        List[Dict[str, Any]]
    top_jobs:          List[Dict[str, Any]]
    final_report:      Dict[str, Any]
    errors:            Annotated[List[str], operator.add]


# ── Node Functions ────────────────────────────────────────────────────────────

def ingest_resume_node(state: CareerState) -> Dict[str, Any]:
    """Node 1: Parse and embed the resume into ChromaDB."""
    print("📄 [Node 1] Ingesting resume...")
    retriever = RAGRetriever()

    try:
        # Ingest raw text directly (file ingestion happens at API level)
        achievements = _extract_bullets(state["resume_text"])
        n = retriever.ingest_text_achievements(achievements, user_id=state["user_id"])
        print(f"   ✓ Stored {n} achievement chunks")
        return {"ingested": True, "errors": []}
    except Exception as e:
        return {"ingested": False, "errors": [f"Ingest error: {str(e)}"]}


def match_jobs_node(state: CareerState) -> Dict[str, Any]:
    """Node 2: Scrape and rank jobs by semantic similarity."""
    print(f"🔍 [Node 2] Searching jobs for: '{state['job_query']}'...")
    agent = JobScraperAgent()

    try:
        results = agent.search_sync(
            query=state["job_query"],
            profile_text=state["resume_text"],
            location=state.get("location", ""),
            max_results=50,
        )
        print(f"   ✓ Found {results['total_ranked']} matched jobs in {results['elapsed_seconds']}s")
        return {
            "job_results": results["jobs"],
            "top_jobs":    results["jobs"][:10],
            "errors":      [],
        }
    except Exception as e:
        return {
            "job_results": [],
            "top_jobs":    [],
            "errors":      [f"Scraper error: {str(e)}"],
        }


def ats_analysis_node(state: CareerState) -> Dict[str, Any]:
    """Node 3: Run ATS analysis against the top job description."""
    print("🎯 [Node 3] Running ATS analysis...")
    agent = ATSFeedbackAgent()

    top_jobs = state.get("job_results", [])
    if not top_jobs:
        return {"ats_report": {"ats_scores": {"overall_ats_score": 0}}, "errors": ["No jobs to analyze"]}

    top_job_desc = top_jobs[0].get("description", "")
    if not top_job_desc:
        return {"ats_report": {"ats_scores": {"overall_ats_score": 0}}, "errors": ["Empty job description"]}

    try:
        report = agent.analyze(
            resume_text=state["resume_text"],
            job_description=top_job_desc,
            user_id=state["user_id"],
        )
        
        # Final Safety Check: Ensure the LLM returned the expected structure
        if not report or 'ats_scores' not in report:
            raise ValueError("LLM returned incomplete data")

        print(f"   ✓ ATS Score: {report['ats_scores'].get('overall_ats_score', 0)}%")
        return {
            "ats_report":      report,
            "enhanced_resume": report.get("enhanced_resume", ""),
            "skill_gaps":      report.get("skill_gaps", []),
            "errors":          [],
        }
    except Exception as e:
        print(f"   ❌ Node 3 Error: {str(e)}")
        # This fallback prevents the 500 error by returning a "safe" empty report
        return {
            "ats_report": {
                "ats_scores": {"overall_ats_score": 0, "category_scores": {}},
                "semantic_similarity": 0,
                "llm_feedback": "Analysis failed. Please check your AI token."
            },
            "enhanced_resume": "",
            "skill_gaps": [],
            "errors": [f"ATS error: {str(e)}"]
        }

def feedback_loop_node(state: CareerState) -> Dict[str, Any]:
    """
    Node 4: Feedback loop — if ATS score is low, re-ingest enhanced resume
    and re-score. Runs max 2 iterations.
    """
    count     = state.get("feedback_loop_count", 0)
    ats_report = state.get("ats_report", {})

    print(f"🔄 [Node 4] Feedback loop iteration {count + 1}...")

    ats_score = ats_report.get("ats_scores", {}).get("overall_ats_score", 0)
    enhanced  = state.get("enhanced_resume", "")

    # If score improved enough or max iterations reached, finalize
    if ats_score >= 70 or count >= 2 or not enhanced:
        print(f"   ✓ Loop complete. Final ATS: {ats_score}%")
        final_report = _build_final_report(state)
        return {
            "final_report":      final_report,
            "feedback_loop_count": count + 1,
        }

    # Re-ingest the enhanced resume for next iteration
    retriever = RAGRetriever()
    achievements = _extract_bullets(enhanced)
    retriever.ingest_text_achievements(achievements, user_id=state["user_id"])

    return {
        "resume_text":       enhanced,  # use enhanced resume going forward
        "feedback_loop_count": count + 1,
    }


def build_report_node(state: CareerState) -> Dict[str, Any]:
    """Node 5: Compile the final structured report."""
    print("📊 [Node 5] Building final report...")
    return {"final_report": _build_final_report(state)}


# ── Conditional Edges ─────────────────────────────────────────────────────────

def should_loop(state: CareerState) -> str:
    """Decide whether to run another ATS feedback loop."""
    count     = state.get("feedback_loop_count", 0)
    ats_score = state.get("ats_report", {}).get("ats_scores", {}).get("overall_ats_score", 0)

    if count < 2 and ats_score < 70 and state.get("enhanced_resume"):
        return "ats_analysis"   # loop back for another pass
    return "build_report"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_career_graph() -> StateGraph:
    graph = StateGraph(CareerState)

    # Add nodes
    graph.add_node("ingest_resume", ingest_resume_node)
    graph.add_node("match_jobs",    match_jobs_node)
    graph.add_node("ats_analysis",  ats_analysis_node)
    graph.add_node("feedback_loop", feedback_loop_node)
    graph.add_node("build_report",  build_report_node)

    # Add edges
    graph.add_edge(START,           "ingest_resume")
    graph.add_edge("ingest_resume", "match_jobs")
    graph.add_edge("match_jobs",    "ats_analysis")
    graph.add_edge("ats_analysis",  "feedback_loop")

    # Conditional: loop or finish
    graph.add_conditional_edges(
        "feedback_loop",
        should_loop,
        {"ats_analysis": "ats_analysis", "build_report": "build_report"},
    )
    graph.add_edge("build_report", END)

    return graph.compile()


# ── Career Assistant ──────────────────────────────────────────────────────────

class CareerAssistant:
    """High-level interface to the LangGraph career assistant."""

    def __init__(self):
        self.graph = build_career_graph()

    def run(
        self,
        resume_text: str,
        job_query:   str,
        location:    str = "",
        user_id:     str = "default",
    ) -> Dict[str, Any]:
        """Run the full autonomous career assistant pipeline."""
        initial_state: CareerState = {
            "resume_text":        resume_text,
            "job_query":          job_query,
            "location":           location,
            "user_id":            user_id,
            "ingested":           False,
            "job_results":        [],
            "ats_report":         {},
            "feedback_loop_count": 0,
            "enhanced_resume":    "",
            "skill_gaps":         [],
            "top_jobs":           [],
            "final_report":       {},
            "errors":             [],
        }

        final_state = self.graph.invoke(initial_state)
        return final_state.get("final_report", {})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_bullets(text: str) -> List[str]:
    """Extract bullet-point lines from text."""
    import re
    bullets = re.findall(
        r"(?:^|\n)\s*[-•*▪►]\s*(.+?)(?=\n|$)", text, re.MULTILINE
    )
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]
    return list(set(bullets + sentences))[:50]


def _build_final_report(state: CareerState) -> Dict[str, Any]:
    ats = state.get("ats_report", {})
    return {
        "summary": {
            "ats_score":          ats.get("ats_scores", {}).get("overall_ats_score", 0),
            "semantic_similarity": ats.get("semantic_similarity", 0),
            "jobs_found":         len(state.get("job_results", [])),
            "skill_gaps_count":   len(state.get("skill_gaps", [])),
            "loop_iterations":    state.get("feedback_loop_count", 0),
            "errors":             state.get("errors", []),
        },
        "top_jobs":        state.get("top_jobs", []),
        "skill_gaps":      state.get("skill_gaps", []),
        "enhanced_resume": state.get("enhanced_resume", ""),
        "llm_feedback":    ats.get("llm_feedback", ""),
        "category_scores": ats.get("ats_scores", {}).get("category_scores", {}),
    }