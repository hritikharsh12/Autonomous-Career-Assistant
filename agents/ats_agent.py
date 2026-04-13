"""
ats_agent.py
ATS Feedback Agent — analyses resume vs job description and produces:
  - Keyword gap report across 10+ skill categories
  - ATS-compliant rewrite suggestions
  - Bullet-point improvements with metric injection
  - Overall compatibility score
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import RAGRetriever, ATS_SKILL_CATEGORIES

load_dotenv()

HF_TOKEN     = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")


def _build_llm() -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
        repo_id=HF_LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.4,
        max_new_tokens=1024,
    )
    return ChatHuggingFace(llm=llm)


# ── Prompt Templates ──────────────────────────────────────────────────────────

ATS_SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) optimization specialist and career coach.
Your job is to analyze resumes against job descriptions and provide highly specific, actionable feedback.
Always be precise, professional, and focus on measurable improvements.
Format your response in clear sections with bullet points."""

RESUME_REWRITE_PROMPT = """Given the following context, rewrite the resume summary/bullets to be ATS-optimized.

{context}

=== CURRENT RESUME TEXT ===
{resume_text}

Instructions:
1. Naturally incorporate the missing ATS keywords listed above
2. Inject the relevant achievements where appropriate  
3. Use strong action verbs (Built, Engineered, Implemented, Designed, Led, Optimized)
4. Quantify results wherever possible
5. Keep bullets concise (1-2 lines max)
6. Maintain professional tone

Return ONLY the rewritten resume text, no explanations."""

FEEDBACK_PROMPT = """Analyze this resume against the job description and provide structured ATS feedback.

=== JOB DESCRIPTION ===
{job_description}

=== RESUME TEXT ===
{resume_text}

=== ATS SCORE BREAKDOWN ===
{ats_breakdown}

Provide feedback in exactly this format:

**OVERALL COMPATIBILITY**: [score]% - [one line summary]

**CRITICAL GAPS** (must add):
• [gap 1 with specific suggestion]
• [gap 2 with specific suggestion]
• [gap 3 with specific suggestion]

**QUICK WINS** (easy improvements):
• [quick win 1]
• [quick win 2]
• [quick win 3]

**SKILL GAP ANALYSIS**:
{category_template}

**RECOMMENDED BULLET REWRITES**:
• Original: [original bullet]
  Improved: [ATS-optimized version with keywords]

**SUMMARY**: [2-3 sentence overall recommendation]"""


class ATSFeedbackAgent:
    """
    Autonomous ATS feedback agent.
    Combines rule-based keyword scoring with LLM-generated suggestions.
    """

    def __init__(self):
        self.retriever = RAGRetriever()
        self.llm       = _build_llm()

    # ── Main Entry Points ─────────────────────────────────────────────────────

    def analyze(
        self,
        resume_text: str,
        job_description: str,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Full ATS analysis pipeline:
        1. Score keyword density
        2. Retrieve relevant achievements via RAG
        3. Generate LLM feedback
        4. Return structured report
        """
        # Step 1: Rule-based ATS scoring
        ats_scores = self.retriever.score_ats(resume_text, job_description)

        # Step 2: RAG — get relevant achievements
        relevant_achievements = self.retriever.get_relevant_achievements(
            job_description, top_k=5
        )

        # Step 3: Semantic similarity
        similarity = self.retriever.profile_job_similarity(resume_text, job_description)

        # Step 4: LLM feedback
        llm_feedback = self._generate_llm_feedback(
            resume_text, job_description, ats_scores
        )

        # Step 5: Build enhanced resume
        enhanced_context = self.retriever.build_enhanced_resume_context(
            job_description, resume_text
        )
        enhanced_resume = self._rewrite_resume(resume_text, enhanced_context)

        return {
            "ats_scores":             ats_scores,
            "semantic_similarity":    round(similarity * 100, 1),
            "relevant_achievements":  relevant_achievements,
            "llm_feedback":           llm_feedback,
            "enhanced_resume":        enhanced_resume,
            "skill_gaps":             self._format_skill_gaps(ats_scores),
        }

    def quick_score(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Fast scoring without LLM call — for real-time feedback."""
        ats_scores = self.retriever.score_ats(resume_text, job_description)
        similarity = self.retriever.profile_job_similarity(resume_text, job_description)
        return {
            "overall_ats_score":   ats_scores["overall_ats_score"],
            "semantic_similarity": round(similarity * 100, 1),
            "top_missing":         self._get_top_missing(ats_scores, n=10),
        }

    # ── LLM Calls ─────────────────────────────────────────────────────────────

    def _generate_llm_feedback(
        self,
        resume_text: str,
        job_description: str,
        ats_scores: Dict,
    ) -> str:
        ats_breakdown  = self._format_ats_breakdown(ats_scores)
        category_template = "\n".join(
            f"  - {cat}: {data['score']}% match"
            for cat, data in ats_scores["category_scores"].items()
        )

        prompt = FEEDBACK_PROMPT.format(
            job_description=job_description[:2000],
            resume_text=resume_text[:2000],
            ats_breakdown=ats_breakdown,
            category_template=category_template,
        )

        messages = [
            SystemMessage(content=ATS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def _rewrite_resume(self, resume_text: str, context: str) -> str:
        prompt = RESUME_REWRITE_PROMPT.format(
            context=context[:2500],
            resume_text=resume_text[:2000],
        )
        messages = [
            SystemMessage(content=ATS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    # ── Formatting Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_ats_breakdown(ats_scores: Dict) -> str:
        lines = [f"Overall ATS Score: {ats_scores['overall_ats_score']}%"]
        for cat, data in ats_scores["category_scores"].items():
            lines.append(
                f"  {cat}: {data['score']}% "
                f"(matched: {', '.join(data['matched'][:3]) or 'none'} | "
                f"missing: {', '.join(data['missing'][:3]) or 'none'})"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_skill_gaps(ats_scores: Dict) -> List[Dict[str, Any]]:
        gaps = []
        for cat, data in ats_scores["category_scores"].items():
            if data["missing"]:
                gaps.append({
                    "category":        cat,
                    "current_score":   data["score"],
                    "missing_keywords": data["missing"],
                    "priority":        "HIGH" if data["score"] < 50 else "MEDIUM",
                })
        return sorted(gaps, key=lambda x: x["current_score"])

    @staticmethod
    def _get_top_missing(ats_scores: Dict, n: int = 10) -> List[str]:
        missing = []
        for data in ats_scores["category_scores"].values():
            missing.extend(data.get("missing", []))
        return missing[:n]