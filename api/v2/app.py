"""Opt-in combined server. The standard main:app entry point remains v1-only."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v2.interview_routes import REVIEWS as REVIEWS
from api.v2.interview_routes import discovery
from api.v2.service import app as seller_api
from main import app as legacy_api
from main import cors_origins

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_origin_regex=r"https://[a-z0-9-]+\.vercel\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.add_api_route("/api/v2", discovery, methods=["GET"])
app.mount("/api/v2", seller_api)
app.mount("/", legacy_api)
