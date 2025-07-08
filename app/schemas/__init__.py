"""
Pydantic Schemas for API Request/Response Validation

This module contains all Pydantic schemas used for request and response
validation across the chatbot platform API endpoints.
"""

from .auth import (
    AuthRequest,
    AuthVerification,
    AuthResponse,
    TokenResponse,
)

from .chat import (
    ChatMessage,
    ChatResponse,
    ChatContext,
    ConversationHistory,
)

from .knowledge import (
    DocumentUpload,
    DocumentResponse,
    DocumentList,
    DocumentStatus,
)

from .config import (
    SystemConfig,
    ConfigUpdate,
    ModelConfig,
    RateLimitConfig,
)

__all__ = [
    # Auth schemas
    "AuthRequest",
    "AuthVerification", 
    "AuthResponse",
    "TokenResponse",
    
    # Chat schemas
    "ChatMessage",
    "ChatResponse",
    "ChatContext",
    "ConversationHistory",
    
    # Knowledge schemas
    "DocumentUpload",
    "DocumentResponse",
    "DocumentList", 
    "DocumentStatus",
    
    # Config schemas
    "SystemConfig",
    "ConfigUpdate",
    "ModelConfig",
    "RateLimitConfig",
]
