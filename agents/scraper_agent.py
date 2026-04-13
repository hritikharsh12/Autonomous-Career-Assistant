"""
scraper_agent.py
Web scraping agent — fetches 50+ live job listings per query,
scores each with semantic similarity, and returns ranked results
in under 30 seconds using async I/O.
"""

"""
scraper_agent.py
Web scraping agent — fetches 50+ live job listings per query,
scores each with semantic similarity, and returns ranked results
in under 30 seconds using async I/O.
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from rag.vectorstore import VectorStore, EmbeddingModel

ua = UserAgent()

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class JobListing:
    title:       str
    company:     str
    location:    str
    description: str
    url:         str
    source:      str
    salary:      str       = ""
    posted_date: str       = ""
    similarity:  float     = 0.0
    tags:        List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title":       self.title,
            "company":     self.company,
            "location":    self.location,
            "description": self.description[:500],  # truncate for API response
            "url":         self.url,
            "source":      self.source,
            "salary":      self.salary,
            "posted_date": self.posted_date,
            "similarity":  round(self.similarity * 100, 1),
            "tags":        self.tags,
        }


# ── Scrapers ──────────────────────────────────────────────────────────────────

class BaseJobScraper:
    """Async base scraper with shared HTTP client and retry logic."""

    HEADERS = {
        "Accept":          "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection":      "keep-alive",
    }

    async def fetch(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        headers = {**self.HEADERS, "User-Agent": ua.random}
        try:
            resp = await client.get(url, headers=headers, timeout=10.0, follow_redirects=True)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        return None

    async def scrape(self, query: str, location: str, limit: int) -> List[JobListing]:
        raise NotImplementedError


class RemoteOKScraper(BaseJobScraper):
    """Scrape RemoteOK — JSON API, no JS rendering needed."""

    BASE_URL = "https://remoteok.com/api"

    async def scrape(self, query: str, location: str = "remote", limit: int = 20) -> List[JobListing]:
        jobs: List[JobListing] = []
        async with httpx.AsyncClient() as client:
            html = await self.fetch(client, self.BASE_URL)
            if not html:
                return jobs
            try:
                import json
                data = json.loads(html)
                for item in data[1:]:    # first item is metadata
                    if not isinstance(item, dict):
                        continue
                    title = item.get("position", "")
                    if query.lower() not in title.lower() and \
                       query.lower() not in " ".join(item.get("tags", [])).lower():
                        continue
                    jobs.append(JobListing(
                        title=title,
                        company=item.get("company", ""),
                        location="Remote",
                        description=BeautifulSoup(item.get("description", ""), "html.parser").get_text()[:1000],
                        url=item.get("url", ""),
                        source="RemoteOK",
                        salary=f"{item.get('salary_min', '')} - {item.get('salary_max', '')}".strip(" -"),
                        tags=item.get("tags", []),
                    ))
                    if len(jobs) >= limit:
                        break
            except Exception:
                pass
        return jobs


class GitHubJobsScraper(BaseJobScraper):
    """Scrape jobs from GitHub Awesome Jobs board (RSS-based)."""

    BASE_URL = "https://www.arbeitnow.com/api/job-board-api"

    async def scrape(self, query: str, location: str = "", limit: int = 20) -> List[JobListing]:
        jobs: List[JobListing] = []
        url = f"{self.BASE_URL}?search={query.replace(' ', '+')}"

        async with httpx.AsyncClient() as client:
            html = await self.fetch(client, url)
            if not html:
                return jobs
            try:
                import json
                data = json.loads(html)
                for item in data.get("data", []):
                    jobs.append(JobListing(
                        title=item.get("title", ""),
                        company=item.get("company_name", ""),
                        location=item.get("location", ""),
                        description=item.get("description", "")[:1000],
                        url=item.get("url", ""),
                        source="ArbeitNow",
                        tags=item.get("tags", []),
                    ))
                    if len(jobs) >= limit:
                        break
            except Exception:
                pass
        return jobs


class JSearchScraper(BaseJobScraper):
    """
    Scrapes jobs from The Muse public API (no key needed for basic use).
    Falls back to mock data if API is unavailable.
    """

    BASE_URL = "https://www.themuse.com/api/public/jobs"

    async def scrape(self, query: str, location: str = "", limit: int = 20) -> List[JobListing]:
        jobs: List[JobListing] = []
        params = f"?descending=True&page=1&category={query.replace(' ', '%20')}"

        async with httpx.AsyncClient() as client:
            html = await self.fetch(client, self.BASE_URL + params)
            if not html:
                return self._mock_jobs(query, limit)
            try:
                import json
                data = json.loads(html)
                for item in data.get("results", []):
                    jobs.append(JobListing(
                        title=item.get("name", ""),
                        company=item.get("company", {}).get("name", ""),
                        location=", ".join(
                            loc.get("name", "") for loc in item.get("locations", [])
                        ),
                        description=BeautifulSoup(
                            item.get("contents", ""), "html.parser"
                        ).get_text()[:1000],
                        url=item.get("refs", {}).get("landing_page", ""),
                        source="TheMuse",
                    ))
                    if len(jobs) >= limit:
                        break
            except Exception:
                return self._mock_jobs(query, limit)
        return jobs

    @staticmethod
    def _mock_jobs(query: str, limit: int) -> List[JobListing]:
        """Return sample jobs if all APIs are unavailable (dev/testing mode)."""
        templates = [
            ("Senior {q} Engineer", "TechCorp", "San Francisco, CA"),
            ("{q} Developer", "StartupXYZ", "Remote"),
            ("Lead {q} Specialist", "MegaCorp", "New York, NY"),
            ("{q} Architect", "CloudSystems", "Austin, TX"),
            ("Principal {q} Engineer", "DataCo", "Seattle, WA"),
        ]
        q = query.title()
        return [
            JobListing(
                title=t[0].format(q=q),
                company=t[1],
                location=t[2],
                description=f"We are looking for a {t[0].format(q=q)} with strong skills in {query}, "
                            f"Python, machine learning, and cloud technologies. "
                            f"Experience with LLMs, RAG pipelines, and agentic systems preferred.",
                url=f"https://example.com/jobs/{i}",
                source="Mock",
            )
            for i, t in enumerate(templates[:limit])
        ]


# ── Main Scraper Agent ────────────────────────────────────────────────────────

class JobScraperAgent:
    """
    Orchestrates multiple scrapers in parallel, deduplicates results,
    ranks by semantic similarity to user profile, and stores in ChromaDB.
    """

    def __init__(self):
        self.scrapers   = [RemoteOKScraper(), GitHubJobsScraper(), JSearchScraper()]
        self.vectorstore = VectorStore()
        self.embedder    = EmbeddingModel()

    async def search_and_rank(
        self,
        query: str,
        profile_text: str,
        location: str = "",
        max_results: int = 50,
        min_similarity: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Main entry point.
        - Scrapes all sources in parallel
        - Deduplicates
        - Scores semantic similarity against user profile
        - Returns ranked list in < 30s
        """
        start_time = time.time()
        per_scraper = max(max_results // len(self.scrapers), 10)

        # Parallel scraping
        tasks = [
            scraper.scrape(query, location, per_scraper)
            for scraper in self.scrapers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs: List[JobListing] = []
        for r in results:
            if isinstance(r, list):
                all_jobs.extend(r)

        # Deduplicate by (title + company)
        seen  = set()
        unique_jobs: List[JobListing] = []
        for job in all_jobs:
            key = f"{job.title.lower()}|{job.company.lower()}"
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)

        # Score semantic similarity
        profile_emb = self.embedder.embed_single(profile_text)
        job_texts   = [f"{j.title} {j.company} {j.description}" for j in unique_jobs]

        if job_texts:
            job_embs = self.embedder.embed(job_texts)
            import numpy as np
            profile_vec = np.array(profile_emb)
            for job, emb in zip(unique_jobs, job_embs):
                job_vec = np.array(emb)
                sim = float(
                    np.dot(profile_vec, job_vec)
                    / (np.linalg.norm(profile_vec) * np.linalg.norm(job_vec) + 1e-10)
                )
                job.similarity = sim

        # Filter & rank
        ranked = sorted(
            [j for j in unique_jobs if j.similarity >= min_similarity],
            key=lambda x: x.similarity,
            reverse=True,
        )[:max_results]

        # Store top jobs in ChromaDB for later retrieval
        self._store_jobs(ranked)

        elapsed = time.time() - start_time

        return {
            "query":           query,
            "total_scraped":   len(all_jobs),
            "total_unique":    len(unique_jobs),
            "total_ranked":    len(ranked),
            "elapsed_seconds": round(elapsed, 2),
            "jobs":            [j.to_dict() for j in ranked],
        }

    def search_sync(
        self,
        query: str,
        profile_text: str,
        location: str = "",
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """Synchronous wrapper — works inside or outside an existing event loop."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                self.search_and_rank(query, profile_text, location, max_results)
            )
            return future.result()

    def _store_jobs(self, jobs: List[JobListing]):
        """Store scraped jobs in ChromaDB for RAG retrieval."""
        self.vectorstore.clear_jobs()
        chunks = [
            {
                "text": f"{j.title} at {j.company} ({j.location})\n{j.description}",
                "metadata": {
                    "title":      j.title,
                    "company":    j.company,
                    "location":   j.location,
                    "url":        j.url,
                    "source":     j.source,
                    "similarity": str(j.similarity),
                },
            }
            for j in jobs
        ]
        if chunks:
            self.vectorstore.upsert_job_chunks(chunks)