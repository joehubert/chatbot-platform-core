"""
Authentication API Schemas

Pydantic schemas for authentication-related API endpoints including
OTP request/verification and session management.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, validator, EmailStr
import re


class AuthRequest(BaseModel):
    """Schema for authentication token request"""
    
    session_id: str = Field(..., description="Session identifier requiring authentication")
    contact_method: str = Field(..., description="Method for token delivery: 'sms' or 'email'")
    contact_value: str = Field(..., description="Phone number or email address for token delivery")
    
    @validator('contact_method')
    def validate_contact_method(cls, v):
        allowed_methods = {'sms', 'email'}
        if v not in allowed_methods:
            raise ValueError(f'Contact method must be one of: {allowed_methods}')
        return v
    
    @validator('contact_value')
    def validate_contact_value(cls, v, values):
        contact_method = values.get('contact_method')
        
        if contact_method == 'email':
            # Basic email validation (Pydantic EmailStr would be better but this is simpler)
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, v):
                raise ValueError('Invalid email address format')
        
        elif contact_method == 'sms':
            # Phone number validation (basic pattern for international numbers)
            phone_pattern = r'^\+?[1-9]\d{1,14}$'
            # Remove common separators for validation
            cleaned_phone = re.sub(r'[\s\-\(\)]', '', v)
            if not re.match(phone_pattern, cleaned_phone):
                raise ValueError('Invalid phone number format')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_12345",
                "contact_method": "email",
                "contact_value": "user@example.com"
            }
        }


class AuthVerification(BaseModel):
    """Schema for authentication token verification"""
    
    session_id: str = Field(..., description="Session identifier")
    token: str = Field(..., min_length=4, max_length=10, description="Authentication token received")
    
    @validator('token')
    def validate_token(cls, v):
        # Remove whitespace and convert to uppercase for consistency
        token = v.strip().upper()
        # Basic token format validation (alphanumeric)
        if not re.match(r'^[A-Z0-9]+$', token):
            raise ValueError('Token must contain only letters and numbers')
        return token
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_12345",
                "token": "ABC123"
            }
        }


class AuthResponse(BaseModel):
    """Schema for authentication request response"""
    
    success: bool = Field(..., description="Whether authentication request was successful")
    message: str = Field(..., description="Response message")
    expires_in: Optional[int] = Field(None, description="Token expiration time in seconds")
    retry_after: Optional[int] = Field(None, description="Seconds before retry is allowed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Authentication token sent to your email",
                "expires_in": 300,
                "retry_after": None
            }
        }


class TokenResponse(BaseModel):
    """Schema for token verification response"""
    
    success: bool = Field(..., description="Whether token verification was successful")
    message: str = Field(..., description="Response message")
    session_token: Optional[str] = Field(None, description="Session token for authenticated requests")
    expires_at: Optional[datetime] = Field(None, description="Session token expiration timestamp")
    user_id: Optional[UUID] = Field(None, description="User identifier if found")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Authentication successful",
                "session_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "expires_at": "2024-01-15T11:00:00Z",
                "user_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }


class SessionInfo(BaseModel):
    """Schema for session information"""
    
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[UUID] = Field(None, description="Authenticated user identifier")
    authenticated: bool = Field(..., description="Whether session is authenticated")
    created_at: datetime = Field(..., description="Session creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Session expiration timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_12345",
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "authenticated": True,
                "created_at": "2024-01-15T10:00:00Z",
                "expires_at": "2024-01-15T11:00:00Z",
                "last_activity": "2024-01-15T10:45:00Z"
            }
        }


class UserProfile(BaseModel):
    """Schema for user profile information"""
    
    id: UUID = Field(..., description="User unique identifier")
    mobile_number: Optional[str] = Field(None, description="User's mobile number")
    email: Optional[str] = Field(None, description="User's email address")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_authenticated: Optional[datetime] = Field(None, description="Last authentication timestamp")
    authentication_count: int = Field(default=0, ge=0, description="Total number of authentications")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "mobile_number": "+1234567890",
                "email": "user@example.com",
                "created_at": "2024-01-01T00:00:00Z",
                "last_authenticated": "2024-01-15T10:30:00Z",
                "authentication_count": 5
            }
        }


class AuthStatus(BaseModel):
    """Schema for authentication status check"""
    
    session_id: str = Field(..., description="Session identifier")
    authenticated: bool = Field(..., description="Whether session is authenticated")
    expires_at: Optional[datetime] = Field(None, description="Authentication expiration")
    time_remaining: Optional[int] = Field(None, description="Seconds until expiration")
    requires_renewal: bool = Field(default=False, description="Whether authentication needs renewal")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_12345",
                "authenticated": True,
                "expires_at": "2024-01-15T11:00:00Z",
                "time_remaining": 900,
                "requires_renewal": False
            }
        }


class AuthError(BaseModel):
    """Schema for authentication error responses"""
    
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    retry_allowed: bool = Field(default=True, description="Whether retry is allowed")
    retry_after: Optional[int] = Field(None, description="Seconds before retry is allowed")
    suggestions: List[str] = Field(default_factory=list, description="Suggested actions")
    
    @validator('error_code')
    def validate_error_code(cls, v):
        allowed_codes = {
            'INVALID_TOKEN',
            'EXPIRED_TOKEN',
            'RATE_LIMITED',
            'INVALID_CONTACT',
            'DELIVERY_FAILED',
            'SESSION_NOT_FOUND',
            'AUTHENTICATION_REQUIRED'
        }
        if v not in allowed_codes:
            raise ValueError(f'Error code must be one of: {allowed_codes}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "error_code": "INVALID_TOKEN",
                "error_message": "The provided token is invalid or has expired",
                "retry_allowed": True,
                "retry_after": None,
                "suggestions": ["Request a new token", "Check your email for the latest token"]
            }
        }
