"""
Chat API Schemas

Pydantic schemas for chat-related API endpoints including message processing,
conversation management, and response validation.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field, validator


class ChatContext(BaseModel):
    """Context information for chat messages"""
    
    page_url: Optional[str] = Field(None, description="URL of the page where chat was initiated")
    user_agent: Optional[str] = Field(None, description="User agent string from browser")
    referrer: Optional[str] = Field(None, description="Referrer URL")
    session_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional session data")
    
    class Config:
        schema_extra = {
            "example": {
                "page_url": "https://example.com/support",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "referrer": "https://example.com/",
                "session_data": {"timezone": "UTC", "language": "en"}
            }
        }


class ChatMessage(BaseModel):
    """Schema for incoming chat messages"""
    
    message: str = Field(..., min_length=1, max_length=4000, description="User message content")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier if authenticated")
    context: Optional[ChatContext] = Field(default_factory=ChatContext, description="Context information")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Hello, I need help with my account",
                "session_id": "session_12345",
                "user_id": "user_67890",
                "context": {
                    "page_url": "https://example.com/support",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            }
        }


class ChatResponse(BaseModel):
    """Schema for chat response messages"""
    
    response: str = Field(..., description="Bot response message")
    session_id: str = Field(..., description="Session identifier")
    requires_auth: bool = Field(default=False, description="Whether authentication is required")
    auth_methods: List[str] = Field(default_factory=list, description="Available authentication methods")
    conversation_id: str = Field(..., description="Unique conversation identifier")
    cached: bool = Field(default=False, description="Whether response was served from cache")
    model_used: Optional[str] = Field(None, description="LLM model used for response generation")
    processing_time_ms: Optional[int] = Field(None, description="Response processing time in milliseconds")
    sources: Optional[List[str]] = Field(default_factory=list, description="Knowledge base sources used")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "I'd be happy to help you with your account. What specific issue are you experiencing?",
                "session_id": "session_12345",
                "requires_auth": False,
                "auth_methods": [],
                "conversation_id": "conv_98765",
                "cached": False,
                "model_used": "gpt-3.5-turbo",
                "processing_time_ms": 1250,
                "sources": ["user_manual.pdf", "faq.md"],
                "confidence_score": 0.85
            }
        }


class MessageHistory(BaseModel):
    """Individual message in conversation history"""
    
    id: UUID = Field(..., description="Message unique identifier")
    content: str = Field(..., description="Message content")
    role: str = Field(..., description="Message role: user, assistant, or system")
    timestamp: datetime = Field(..., description="Message timestamp")
    model_used: Optional[str] = Field(None, description="Model used for assistant messages")
    cached: bool = Field(default=False, description="Whether message was cached")
    processing_time_ms: Optional[int] = Field(None, description="Processing time for assistant messages")
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = {'user', 'assistant', 'system'}
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {allowed_roles}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "Hello, I need help with my account",
                "role": "user", 
                "timestamp": "2024-01-15T10:30:00Z",
                "model_used": None,
                "cached": False,
                "processing_time_ms": None
            }
        }


class ConversationHistory(BaseModel):
    """Schema for conversation history retrieval"""
    
    conversation_id: UUID = Field(..., description="Conversation unique identifier") 
    session_id: str = Field(..., description="Session identifier")
    user_identifier: Optional[str] = Field(None, description="User identifier if available")
    started_at: datetime = Field(..., description="Conversation start timestamp")
    ended_at: Optional[datetime] = Field(None, description="Conversation end timestamp")
    messages: List[MessageHistory] = Field(..., description="List of messages in conversation")
    resolved: bool = Field(default=False, description="Whether conversation was resolved")
    resolution_attempts: int = Field(default=0, ge=0, description="Number of resolution attempts")
    authenticated: bool = Field(default=False, description="Whether user was authenticated")
    total_messages: int = Field(..., ge=0, description="Total number of messages")
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "session_id": "session_12345", 
                "user_identifier": "user_67890",
                "started_at": "2024-01-15T10:30:00Z",
                "ended_at": None,
                "messages": [],
                "resolved": False,
                "resolution_attempts": 1,
                "authenticated": True,
                "total_messages": 4
            }
        }


class ConversationSummary(BaseModel):
    """Schema for conversation summary data"""
    
    conversation_id: UUID = Field(..., description="Conversation unique identifier")
    summary: str = Field(..., description="Conversation summary")
    key_topics: List[str] = Field(default_factory=list, description="Key topics discussed")
    resolution_status: str = Field(..., description="Resolution status")
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=5.0, description="User satisfaction score")
    duration_minutes: Optional[int] = Field(None, ge=0, description="Conversation duration in minutes")
    
    @validator('resolution_status')
    def validate_resolution_status(cls, v):
        allowed_statuses = {'resolved', 'unresolved', 'escalated', 'abandoned'}
        if v not in allowed_statuses:
            raise ValueError(f'Resolution status must be one of: {allowed_statuses}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "summary": "User inquired about account access issues and was guided through password reset",
                "key_topics": ["account access", "password reset", "security"],
                "resolution_status": "resolved",
                "user_satisfaction": 4.5,
                "duration_minutes": 8
            }
        }


class FeedbackRequest(BaseModel):
    """Schema for user feedback on responses"""
    
    conversation_id: UUID = Field(..., description="Conversation identifier")
    message_id: UUID = Field(..., description="Specific message being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Optional feedback text")
    helpful: bool = Field(..., description="Whether the response was helpful")
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "message_id": "987fcdeb-51a2-43d1-b789-012345678900",
                "rating": 4,
                "feedback_text": "The response was helpful but could have been more specific",
                "helpful": True
            }
        }


class FeedbackResponse(BaseModel):
    """Schema for feedback submission response"""
    
    success: bool = Field(..., description="Whether feedback was successfully recorded")
    feedback_id: UUID = Field(..., description="Unique identifier for submitted feedback")
    message: str = Field(..., description="Confirmation message")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "feedback_id": "456e7890-e12b-34c5-d678-901234567890", 
                "message": "Thank you for your feedback"
            }
        }
