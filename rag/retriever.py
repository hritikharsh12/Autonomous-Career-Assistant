"""
retriever.py
Handles:
  1. PDF / DOCX resume parsing → text chunks
  2. Achievement extraction & upsertion into ChromaDB
  3. Context retrieval for resume enhancement (RAG injection)
  4. ATS keyword density scoring
"""

import re
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument

from rag.vectorstore import VectorStore

# ── ATS Skill Categories ──────────────────────────────────────────────────────
ATS_SKILL_CATEGORIES: Dict[str, List[str]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "scala", "kotlin", "swift", "r", "matlab",
    ],
    "ml_frameworks": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
        "huggingface", "transformers", "langchain", "langgraph", "spacy", "nltk",
    ],
    "data_engineering": [
        "spark", "kafka", "airflow", "dbt", "hadoop", "hive", "flink",
        "databricks", "snowflake", "bigquery", "redshift",
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "jenkins",
        "github actions", "ci/cd", "helm", "ansible",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "cassandra", "dynamodb", "sqlite", "neo4j", "chromadb", "pinecone",
    ],
    "web_frameworks": [
        "fastapi", "django", "flask", "react", "nextjs", "vue", "angular",
        "express", "spring boot", "graphql", "rest api",
    ],
    "ai_concepts": [
        "llm", "rag", "fine-tuning", "embeddings", "vector database",
        "prompt engineering", "agents", "nlp", "computer vision",
        "reinforcement learning", "generative ai",
    ],
    "soft_skills": [
        "leadership", "communication", "agile", "scrum", "collaboration",
        "problem-solving", "mentoring", "cross-functional",
    ],
    "data_tools": [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau",
        "power bi", "excel", "jupyter",
    ],
    "security": [
        "oauth", "jwt", "ssl", "encryption", "penetration testing",
        "vulnerability", "soc2", "gdpr", "zero trust",
    ],
    "methodologies": [
        "microservices", "serverless", "event-driven", "domain-driven design",
        "test-driven development", "tdd", "bdd", "design patterns",
    ],
}


class ResumeParser:
    """Parse PDF or DOCX resumes into clean text."""

    @staticmethod
    def parse_pdf(path: str) -> str:
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
        return "\n".join(text)

    @staticmethod
    def parse_docx(path: str) -> str:
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    @staticmethod
    def parse(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return ResumeParser.parse_pdf(path)
        elif ext in (".docx", ".doc"):
            return ResumeParser.parse_docx(path)
        elif ext == ".txt":
            return Path(path).read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")


class TextChunker:
    """Split long text into overlapping chunks for vectorization."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        words  = text.split()
        chunks = []
        start  = 0
        metadata = metadata or {}

        while start < len(words):
            end   = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 20:   # skip tiny fragments
                chunks.append({"text": chunk, "metadata": {**metadata, "chunk_index": len(chunks)}})
            start += self.chunk_size - self.overlap

        return chunks


class AchievementExtractor:
    """
    Extract bullet-point achievements from resume text.
    Focuses on quantified results (numbers, %, metrics).
    """

    ACHIEVEMENT_PATTERN = re.compile(
        r"(?:^|\n)\s*[-•*▪►]\s*(.+?)(?=\n\s*[-•*▪►]|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    METRIC_PATTERN = re.compile(
        r"\d+(?:\.\d+)?(?:\s*%|\s*x|\s*times|\s*hours?|\s*minutes?|\s*seconds?|\s*[KkMmBb]\b|\+)"
    )

    def extract(self, text: str) -> List[Dict[str, Any]]:
        bullets = self.ACHIEVEMENT_PATTERN.findall(text)
        achievements = []
        for b in bullets:
            b = b.strip().replace("\n", " ")
            if len(b) < 20:
                continue
            metrics = self.METRIC_PATTERN.findall(b)
            achievements.append({
                "text":        b,
                "has_metrics": len(metrics) > 0,
                "metrics":     metrics,
            })
        return achievements


class ATSScorer:
    """Score resume text against a job description for ATS keyword density."""

    def score(
        self, resume_text: str, job_description: str
    ) -> Dict[str, Any]:
        resume_lower = resume_text.lower()
        job_lower    = job_description.lower()

        category_scores: Dict[str, Dict] = {}
        total_matched  = 0
        total_keywords = 0
        missing_by_category: Dict[str, List[str]] = {}

        for category, keywords in ATS_SKILL_CATEGORIES.items():
            # Only check keywords that appear in the JD
            jd_keywords = [kw for kw in keywords if kw in job_lower]
            if not jd_keywords:
                continue

            matched = [kw for kw in jd_keywords if kw in resume_lower]
            missing = [kw for kw in jd_keywords if kw not in resume_lower]

            score = len(matched) / len(jd_keywords) if jd_keywords else 0.0
            category_scores[category] = {
                "score":    round(score * 100, 1),
                "matched":  matched,
                "missing":  missing,
                "total_jd": len(jd_keywords),
            }
            missing_by_category[category] = missing
            total_matched  += len(matched)
            total_keywords += len(jd_keywords)

        overall = (total_matched / total_keywords * 100) if total_keywords else 0.0

        return {
            "overall_ats_score":    round(overall, 1),
            "category_scores":      category_scores,
            "missing_by_category":  missing_by_category,
            "total_matched":        total_matched,
            "total_keywords_in_jd": total_keywords,
        }


class RAGRetriever:
    """
    Main retriever class.
    - Loads resume → chunks → ChromaDB
    - Retrieves relevant achievements for resume enhancement
    - Scores ATS keyword density
    """

    def __init__(self):
        self.vectorstore = VectorStore()
        self.parser      = ResumeParser()
        self.chunker     = TextChunker()
        self.extractor   = AchievementExtractor()
        self.ats_scorer  = ATSScorer()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_resume(self, file_path: str, user_id: str = "default") -> Dict[str, Any]:
        """Parse resume file and store chunks + achievements in ChromaDB."""
        raw_text     = self.parser.parse(file_path)
        chunks       = self.chunker.chunk(raw_text, metadata={"user_id": user_id, "source": "resume"})
        achievements = self.extractor.extract(raw_text)

        # Store raw chunks
        n_chunks = self.vectorstore.upsert_resume_chunks(chunks)

        # Store achievements as separate high-priority chunks
        ach_chunks = [
            {
                "text": a["text"],
                "metadata": {
                    "user_id":     user_id,
                    "source":      "achievement",
                    "has_metrics": a["has_metrics"],
                    "metrics":     str(a["metrics"]),
                },
            }
            for a in achievements
        ]
        n_achievements = self.vectorstore.upsert_resume_chunks(ach_chunks)

        return {
            "raw_text":      raw_text,
            "n_chunks":      n_chunks,
            "n_achievements": n_achievements,
            "achievements":  achievements,
        }

    def ingest_text_achievements(
        self, achievements: List[str], user_id: str = "default"
    ) -> int:
        """Directly ingest a list of achievement strings (no file needed)."""
        chunks = [
            {
                "text": a,
                "metadata": {"user_id": user_id, "source": "achievement"},
            }
            for a in achievements
        ]
        return self.vectorstore.upsert_resume_chunks(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_relevant_achievements(
        self, job_description: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve achievements most relevant to the job description."""
        results = self.vectorstore.query_resume(job_description, top_k=top_k)
        # Filter only achievement chunks
        return [r for r in results if r["metadata"].get("source") == "achievement"]

    def build_enhanced_resume_context(
        self, job_description: str, base_resume_text: str
    ) -> str:
        """
        Build a RAG-enhanced prompt context that injects relevant
        achievements into the resume for the given JD.
        """
        relevant = self.get_relevant_achievements(job_description, top_k=7)
        ats_info = self.ats_scorer.score(base_resume_text, job_description)

        achievement_block = "\n".join(
            f"  • {r['text']}  [similarity: {r['similarity']:.0%}]"
            for r in relevant
        )

        missing_keywords = []
        for cat, keywords in ats_info["missing_by_category"].items():
            if keywords:
                missing_keywords.extend(keywords[:3])   # top 3 per category

        context = f"""
=== JOB DESCRIPTION ===
{job_description}

=== RELEVANT ACHIEVEMENTS FROM CANDIDATE PROFILE ===
{achievement_block or 'No achievements stored yet.'}

=== ATS KEYWORD GAPS (add these naturally) ===
{', '.join(missing_keywords[:20]) or 'None detected'}

=== CURRENT ATS SCORE ===
Overall: {ats_info['overall_ats_score']}%
""".strip()

        return context

    # ── ATS Scoring ───────────────────────────────────────────────────────────

    def score_ats(
        self, resume_text: str, job_description: str
    ) -> Dict[str, Any]:
        return self.ats_scorer.score(resume_text, job_description)

    # ── Similarity ────────────────────────────────────────────────────────────

    def profile_job_similarity(
        self, profile_text: str, job_description: str
    ) -> float:
        """Return semantic similarity score (0–1) between profile and JD."""
        return self.vectorstore.semantic_similarity(profile_text, job_description)

    def stats(self) -> Dict[str, int]:
        return self.vectorstore.stats()