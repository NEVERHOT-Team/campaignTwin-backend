import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.v1_contract import configure
from api.v1_reviews import router as reviews_router
from api.v1_runs import router as runs_router

app = FastAPI(title="CampaignTwin", version="0.1.0", separate_input_output_schemas=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   *(s.strip() for s in os.getenv("CORS_ORIGINS", "").split(",") if s.strip())],
    allow_origin_regex=r"https://[a-z0-9-]+\.vercel\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
app.include_router(reviews_router)
app.include_router(runs_router)
configure(app)
