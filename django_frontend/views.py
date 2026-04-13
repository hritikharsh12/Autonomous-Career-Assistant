"""
views.py — Django views that proxy to the FastAPI backend.
All heavy logic lives in FastAPI; Django just renders templates.
"""

import json
import requests
from django.shortcuts    import render
from django.http         import JsonResponse
from django.views        import View
from django.conf         import settings

API = settings.FASTAPI_BASE_URL


def _post(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API}{endpoint}", json=payload, timeout=60)
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get(endpoint: str) -> dict:
    try:
        r = requests.get(f"{API}{endpoint}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Page Views ────────────────────────────────────────────────────────────────

def index(request):
    stats = _get("/api/stats")
    return render(request, "base/index.html", {"stats": stats})


def dashboard(request):
    return render(request, "dashboard/dashboard.html")


def jobs_page(request):
    return render(request, "jobs/jobs.html")


def resume_page(request):
    return render(request, "resume/resume.html")


# ── AJAX / API Proxy Views ────────────────────────────────────────────────────

class AnalyzeView(View):
    def post(self, request):
        data = json.loads(request.body)
        result = _post("/api/analyze", {
            "resume_text": data.get("resume_text", ""),
            "job_query":   data.get("job_query", ""),
            "location":    data.get("location", ""),
            "user_id":     data.get("user_id", "default"),
        })
        return JsonResponse(result)


class QuickScoreView(View):
    def post(self, request):
        data = json.loads(request.body)
        result = _post("/api/quick-score", {
            "resume_text":     data.get("resume_text", ""),
            "job_description": data.get("job_description", ""),
        })
        return JsonResponse(result)


class SearchJobsView(View):
    def post(self, request):
        data = json.loads(request.body)
        result = _post("/api/search-jobs", {
            "query":        data.get("query", ""),
            "profile_text": data.get("profile_text", ""),
            "location":     data.get("location", ""),
            "max_results":  data.get("max_results", 50),
        })
        return JsonResponse(result)


class UploadResumeView(View):
    def post(self, request):
        f       = request.FILES.get("file")
        user_id = request.POST.get("user_id", "default")
        if not f:
            return JsonResponse({"success": False, "error": "No file"}, status=400)
        try:
            r = requests.post(
                f"{API}/api/upload-resume",
                files={"file": (f.name, f.read(), f.content_type)},
                data={"user_id": user_id},
                timeout=30,
            )
            return JsonResponse(r.json())
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)


class StatsView(View):
    def get(self, request):
        return JsonResponse(_get("/api/stats"))