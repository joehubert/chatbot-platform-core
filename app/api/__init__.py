"""
API package for the chatbot platform.

This package contains all API endpoints and routing logic for the chatbot platform.
It follows the FastAPI framework conventions and provides versioned API endpoints.
"""

from fastapi import APIRouter

from app.api.v1 import chat, auth

api_router = APIRouter()

# Include v1 API routes
api_router.include_router(chat.router, prefix="/v1/chat", tags=["chat"])
api_router.include_router(auth.router, prefix="/v1/auth", tags=["authentication"])

__all__ = ["api_router"]
