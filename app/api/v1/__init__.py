"""
API v1 package.

This package contains version 1 of the chatbot platform API endpoints.
"""

from fastapi import APIRouter

from . import chat, auth, health, knowledge, config, analytics

router = APIRouter()

# Include all v1 endpoints
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(auth.router, prefix="/auth", tags=["authentication"])
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])
router.include_router(config.router, prefix="/config", tags=["config"])
router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])

__all__ = ["router", "chat", "auth", "health", "knowledge", "config", "analytics"]
