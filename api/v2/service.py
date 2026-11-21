"""Independently documented seller API mounted at /api/v2 by the server."""

from fastapi import FastAPI

from api.v2.contract import configure
from api.v2.interview_routes import router as interview_router
from api.v2.runs import router as runs_router

app = FastAPI(title="CampaignTwin seller API", version="2.0.0", root_path_in_servers=False)
app.include_router(interview_router)
app.include_router(runs_router)
configure(app)
