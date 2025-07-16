"""
API package for the chatbot platform.

This package contains all API endpoints and routing logic for the chatbot platform.
It follows the FastAPI framework conventions and provides versioned API endpoints.
"""

from fastapi import APIRouter

# Import the complete v1 router instead of individual routers
from app.api.v1 import router as v1_router

api_router = APIRouter()

# Include the complete v1 API router with all endpoints
api_router.include_router(v1_router)

__all__ = ["api_router"]