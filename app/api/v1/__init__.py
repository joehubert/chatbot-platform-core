"""
API v1 package.

This package contains version 1 of the chatbot platform API endpoints.
"""

from fastapi import APIRouter

from . import chat, auth

router = APIRouter()

# Include all v1 endpoints
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(auth.router, prefix="/auth", tags=["authentication"])

__all__ = ["router", "chat", "auth"]
