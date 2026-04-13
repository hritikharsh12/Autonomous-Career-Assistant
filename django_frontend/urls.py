from django.urls import path
from . import views

urlpatterns = [
    path("",               views.index,                        name="index"),
    path("dashboard/",     views.dashboard,                    name="dashboard"),
    path("jobs/",          views.jobs_page,                    name="jobs"),
    path("resume/",        views.resume_page,                  name="resume"),

    # AJAX endpoints
    path("api/analyze/",       views.AnalyzeView.as_view(),      name="analyze"),
    path("api/quick-score/",   views.QuickScoreView.as_view(),   name="quick_score"),
    path("api/search-jobs/",   views.SearchJobsView.as_view(),   name="search_jobs"),
    path("api/upload-resume/", views.UploadResumeView.as_view(), name="upload_resume"),
    path("api/stats/",         views.StatsView.as_view(),        name="stats"),
]