"""
API v1 Router

Main router that includes all API endpoints for version 1.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import translation, avatars, users, health

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(translation.router, prefix="/translate", tags=["translation"])
api_router.include_router(avatars.router, prefix="/avatars", tags=["avatars"])
api_router.include_router(users.router, prefix="/users", tags=["users"])