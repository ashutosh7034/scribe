"""
Scribe FastAPI Application Entry Point

This module initializes the FastAPI application with all necessary middleware,
routers, and configuration for the real-time sign language translation platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import engine
from app.models import Base

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI application
app = FastAPI(
    title="Scribe API",
    description="Real-time sign language translation platform API",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancer and monitoring."""
    return {"status": "healthy", "service": "scribe-api"}


@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Scribe API",
        "version": "1.0.0",
        "description": "Real-time sign language translation platform",
    }