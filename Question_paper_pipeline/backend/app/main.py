"""
FastAPI application entry point.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(
    title       = "Gecko Med — Question Paper Extraction API",
    description = "AI/ML pipeline for extracting and classifying questions from medical exam PDFs.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow React dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "Gecko Med Question Paper Extraction Pipeline",
        "docs":    "/docs",
        "health":  "/api/health",
    }
