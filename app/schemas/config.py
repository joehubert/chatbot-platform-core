"""
Configuration API Schemas

Pydantic schemas for system configuration management including
model settings, rate limiting, and system parameters.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"


class ModelRole(str, Enum):
    """Model roles in the system"""

    RELEVANCE_CHECK = "relevance_check"
    SIMPLE_QUERY = "simple_query"
    COMPLEX_QUERY = "complex_query"
    CLARIFICATION = "clarification"
    EMBEDDING = "embedding"


class CacheStrategy(str, Enum):
    """Cache strategies"""

    VECTOR_SIMILARITY = "vector_similarity"
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class ModelConfig(BaseModel):
    """Configuration for individual LLM models"""

    provider: ModelProvider = Field(..., description="Model provider")
    model_name: str = Field(..., description="Model identifier")
    role: ModelRole = Field(..., description="Model role in the system")
    api_endpoint: Optional[str] = Field(None, description="Custom API endpoint")
    api_key_name: Optional[str] = Field(
        None, description="Environment variable name for API key"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=32000, description="Maximum tokens per request"
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Model temperature"
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    timeout_seconds: int = Field(
        default=30, ge=1, le=300, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    cost_per_1k_tokens: Optional[float] = Field(
        None, ge=0.0, description="Cost per 1000 tokens"
    )
    enabled: bool = Field(default=True, description="Whether model is enabled")
    fallback_models: List[str] = Field(
        default_factory=list, description="Fallback model identifiers"
    )

    @field_validator("model_name")
    def validate_model_name(cls, v):
        if not v or v.isspace():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "role": "simple_query",
                "api_endpoint": None,
                "api_key_name": "OPENAI_API_KEY",
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.9,
                "timeout_seconds": 30,
                "max_retries": 3,
                "cost_per_1k_tokens": 0.002,
                "enabled": True,
                "fallback_models": ["gpt-3.5-turbo-backup"],
            }
        }

class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""

    enabled: bool = Field(default=True, description="Whether rate limiting is enabled")
    per_user_per_minute: int = Field(
        default=60, ge=1, le=1000, description="Requests per user per minute"
    )
    global_per_minute: int = Field(
        default=1000, ge=1, le=10000, description="Global requests per minute"
    )
    burst_capacity: int = Field(
        default=10, ge=1, le=100, description="Burst capacity for token bucket"
    )
    rate_limit_window_seconds: int = Field(
        default=60, ge=1, le=3600, description="Rate limit window in seconds"
    )
    cleanup_interval_seconds: int = Field(
        default=300, ge=60, le=3600, description="Cleanup interval for expired tokens"
    )
    block_duration_minutes: int = Field(
        default=5, ge=1, le=60, description="Block duration when limits exceeded"
    )

    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "per_user_per_minute": 60,
                "global_per_minute": 1000,
                "burst_capacity": 10,
                "rate_limit_window_seconds": 60,
                "cleanup_interval_seconds": 300,
                "block_duration_minutes": 5,
            }
        }


class ModelProviderConfig(BaseModel):
    """Configuration for model providers"""

    provider: ModelProvider = Field(..., description="Provider name")
    api_key_name: str = Field(..., description="Environment variable for API key")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    organization: Optional[str] = Field(None, description="Organization ID")
    models: Dict[str, ModelConfig] = Field(
        default_factory=dict, description="Available models for this provider"
    )
    rate_limits: Optional[Dict[str, int]] = Field(
        None, description="Provider-specific rate limits"
    )
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "openai",
                "api_key_name": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "organization": "org-123",
                "models": {
                    "gpt-3.5": {
                        "provider": "openai",
                        "model_name": "gpt-3.5-turbo",
                        "role": "simple_query"
                    }
                },
                "rate_limits": {
                    "requests_per_minute": 60
                },
                "enabled": True
            }
        }

class CacheConfig(BaseModel):
    """Semantic cache configuration"""

    enabled: bool = Field(
        default=True, description="Whether semantic caching is enabled"
    )
    strategy: CacheStrategy = Field(
        default=CacheStrategy.VECTOR_SIMILARITY, description="Cache strategy"
    )
    similarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Similarity threshold for cache hits"
    )
    ttl_hours: int = Field(default=24, ge=1, le=8760, description="Cache TTL in hours")
    max_cache_size: int = Field(
        default=10000, ge=100, le=100000, description="Maximum cache entries"
    )
    cleanup_interval_hours: int = Field(
        default=6, ge=1, le=24, description="Cache cleanup interval in hours"
    )

    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "strategy": "vector_similarity",
                "similarity_threshold": 0.85,
                "ttl_hours": 24,
                "max_cache_size": 10000,
                "cleanup_interval_hours": 6,
            }
        }

class AuthConfig(BaseModel):
    """Authentication configuration"""

    session_timeout_minutes: int = Field(
        default=30, ge=5, le=480, description="Session timeout in minutes"
    )
    otp_expiry_minutes: int = Field(
        default=5, ge=1, le=30, description="OTP expiry time in minutes"
    )
    max_auth_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum authentication attempts"
    )
    lockout_duration_minutes: int = Field(
        default=15, ge=5, le=120, description="Lockout duration after max attempts"
    )
    sms_provider: Optional[str] = Field(
        None, description="SMS provider (twilio, aws_sns)"
    )
    email_provider: Optional[str] = Field(
        None, description="Email provider (sendgrid, aws_ses)"
    )

    class Config:
        schema_extra = {
            "example": {
                "session_timeout_minutes": 30,
                "otp_expiry_minutes": 5,
                "max_auth_attempts": 3,
                "lockout_duration_minutes": 15,
                "sms_provider": "twilio",
                "email_provider": "sendgrid",
            }
        }

class VectorDBConfig(BaseModel):
    """Vector database configuration"""

    provider: str = Field(..., description="Vector database provider")
    api_key_name: Optional[str] = Field(
        None, description="Environment variable for API key"
    )
    endpoint: Optional[str] = Field(None, description="Database endpoint")
    index_name: str = Field(
        default="chatbot_knowledge", description="Index/collection name"
    )
    dimension: int = Field(
        default=1536, ge=128, le=4096, description="Vector dimension"
    )
    similarity_metric: str = Field(default="cosine", description="Similarity metric")
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of similar vectors to retrieve"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v):
        allowed_providers = {"pinecone", "weaviate", "chroma", "qdrant", "milvus"}
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {allowed_providers}")
        return v

    @field_validator("similarity_metric")
    @classmethod
    def validate_similarity_metric(cls, v):
        allowed_metrics = {"cosine", "euclidean", "dot_product"}
        if v not in allowed_metrics:
            raise ValueError(f"Similarity metric must be one of: {allowed_metrics}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "provider": "pinecone",
                "api_key_name": "PINECONE_API_KEY",
                "endpoint": "https://index-name.svc.region.pinecone.io",
                "index_name": "chatbot_knowledge",
                "dimension": 1536,
                "similarity_metric": "cosine",
                "top_k": 5,
            }
        }


class DocumentConfig(BaseModel):
    """Document processing configuration"""

    max_file_size_mb: int = Field(
        default=50, ge=1, le=500, description="Maximum file size in MB"
    )
    allowed_file_types: List[str] = Field(
        default_factory=lambda: ["pdf", "txt", "docx", "md", "html"],
        description="Allowed file extensions",
    )
    chunk_size: int = Field(
        default=1000, ge=100, le=8000, description="Text chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Overlap between chunks"
    )
    auto_expire_days: Optional[int] = Field(
        None, ge=1, le=3650, description="Auto-expire documents after days"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v, values):
        chunk_size = values.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    class Config:
        schema_extra = {
            "example": {
                "max_file_size_mb": 50,
                "allowed_file_types": ["pdf", "txt", "docx", "md"],
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "auto_expire_days": 365,
            }
        }




class ConfigUpdate(BaseModel):
    """Schema for partial configuration updates"""

    models: Optional[Dict[str, ModelConfig]] = Field(
        None, description="Model configurations to update"
    )
    rate_limiting: Optional[RateLimitConfig] = Field(
        None, description="Rate limiting settings"
    )
    cache: Optional[CacheConfig] = Field(None, description="Cache settings")
    auth: Optional[AuthConfig] = Field(None, description="Authentication settings")
    vector_db: Optional[VectorDBConfig] = Field(
        None, description="Vector database settings"
    )
    documents: Optional[DocumentConfig] = Field(
        None, description="Document processing settings"
    )
    fallback_error_message: Optional[str] = Field(
        None, description="Fallback error message"
    )
    debug_mode: Optional[bool] = Field(None, description="Enable debug logging")

    class Config:
        schema_extra = {
            "example": {
                "rate_limiting": {
                    "per_user_per_minute": 100,
                    "global_per_minute": 2000,
                },
                "cache": {"similarity_threshold": 0.9},
                "debug_mode": True,
            }
        }


class ConfigValidationResult(BaseModel):
    """Schema for configuration validation results"""

    valid: bool = Field(..., description="Whether configuration is valid")
    errors: List[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    missing_env_vars: List[str] = Field(
        default_factory=list, description="Missing environment variables"
    )

    class Config:
        schema_extra = {
            "example": {
                "valid": False,
                "errors": [
                    "Model 'gpt-4' requires OPENAI_API_KEY environment variable"
                ],
                "warnings": ["Cache TTL is very low (1 hour)"],
                "missing_env_vars": ["OPENAI_API_KEY", "PINECONE_API_KEY"],
            }
        }

class SystemConfig(BaseModel):
    """Complete system configuration"""

    models: Dict[str, ModelConfig] = Field(
        ..., description="Model configurations by identifier"
    )
    rate_limiting: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting settings"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache settings"
    )
    auth: AuthConfig = Field(
        default_factory=AuthConfig, description="Authentication settings"
    )
    vector_db: VectorDBConfig = Field(..., description="Vector database settings")
    documents: DocumentConfig = Field(
        default_factory=DocumentConfig, description="Document processing settings"
    )
    fallback_error_message: str = Field(
        default="I'm having trouble right now. Please contact support.",
        description="Fallback error message",
    )
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    last_updated: Optional[datetime] = Field(
        None, description="Last configuration update"
    )
    version: str = Field(default="1.0.0", description="Configuration version")

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        if not v:
            raise ValueError("At least one model must be configured")

        # Check that all required roles have at least one model
        required_roles = {
            ModelRole.RELEVANCE_CHECK,
            ModelRole.SIMPLE_QUERY,
            ModelRole.COMPLEX_QUERY,
        }
        configured_roles = {model.role for model in v.values()}
        missing_roles = required_roles - configured_roles

        if missing_roles:
            raise ValueError(f"Missing models for required roles: {missing_roles}")

        return v

    class Config:
        schema_extra = {
            "example": {
                "models": {
                    "gpt-3.5-simple": {
                        "provider": "openai",
                        "model_name": "gpt-3.5-turbo",
                        "role": "simple_query",
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "enabled": True,
                    }
                },
                "rate_limiting": {
                    "enabled": True,
                    "per_user_per_minute": 60,
                    "global_per_minute": 1000,
                },
                "cache": {
                    "enabled": True,
                    "similarity_threshold": 0.85,
                    "ttl_hours": 24,
                },
                "auth": {"session_timeout_minutes": 30, "otp_expiry_minutes": 5},
                "vector_db": {
                    "provider": "pinecone",
                    "index_name": "chatbot_knowledge",
                    "dimension": 1536,
                },
                "documents": {
                    "max_file_size_mb": 50,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                },
                "fallback_error_message": "I'm having trouble right now. Please contact support.",
                "debug_mode": False,
                "version": "1.0.0",
            }
        }


class ConfigResponse(BaseModel):
    """Response schema for configuration endpoints"""
    
    config: SystemConfig = Field(..., description="Current system configuration")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration metadata"
    )
    last_updated: Optional[datetime] = Field(
        None, description="Last configuration update timestamp"
    )
    version: str = Field(default="1.0.0", description="Configuration version")
    validation_status: Optional[ConfigValidationResult] = Field(
        None, description="Configuration validation results"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "config": {
                    "models": {
                        "gpt-3.5-simple": {
                            "provider": "openai",
                            "model_name": "gpt-3.5-turbo",
                            "role": "simple_query"
                        }
                    },
                    "rate_limiting": {
                        "enabled": True,
                        "per_user_per_minute": 60
                    },
                    "cache": {
                        "enabled": True,
                        "similarity_threshold": 0.85
                    }
                },
                "metadata": {
                    "total_models": 3,
                    "active_providers": ["openai", "anthropic"]
                },
                "last_updated": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "validation_status": {
                    "valid": True,
                    "errors": [],
                    "warnings": []
                }
            }
        }









class HealthStatus(BaseModel):
    """Schema for system health status"""

    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: int = Field(..., ge=0, description="System uptime in seconds")
    version: str = Field(..., description="System version")
    components: Dict[str, Dict[str, Any]] = Field(
        ..., description="Component health statuses"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed_statuses = {"healthy", "degraded", "unhealthy"}
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "uptime_seconds": 86400,
                "version": "1.0.0",
                "components": {
                    "database": {
                        "status": "healthy",
                        "response_time_ms": 25,
                        "last_check": "2024-01-15T10:30:00Z",
                    },
                    "redis": {
                        "status": "healthy",
                        "response_time_ms": 5,
                        "last_check": "2024-01-15T10:30:00Z",
                    },
                    "vector_db": {
                        "status": "healthy",
                        "response_time_ms": 150,
                        "last_check": "2024-01-15T10:30:00Z",
                    },
                },
            }
        }


class MetricsData(BaseModel):
    """Schema for system metrics"""

    requests_per_minute: int = Field(
        ..., ge=0, description="Current requests per minute"
    )
    average_response_time_ms: float = Field(
        ..., ge=0, description="Average response time"
    )
    cache_hit_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Cache hit rate percentage"
    )
    active_conversations: int = Field(
        ..., ge=0, description="Number of active conversations"
    )
    total_documents: int = Field(
        ..., ge=0, description="Total documents in knowledge base"
    )
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate percentage")
    model_usage: Dict[str, int] = Field(..., description="Usage count by model")

    class Config:
        schema_extra = {
            "example": {
                "requests_per_minute": 45,
                "average_response_time_ms": 1250.5,
                "cache_hit_rate": 0.35,
                "active_conversations": 12,
                "total_documents": 156,
                "error_rate": 0.02,
                "model_usage": {
                    "gpt-3.5-turbo": 320,
                    "gpt-4": 85,
                    "claude-3-sonnet": 45,
                },
            }
        }

# Add these to the __all__ export list at the end of the file
__all__ = [
    # Existing exports...
    "ModelProvider",
    "ModelRole", 
    "CacheStrategy",
    "ModelConfig",
    "RateLimitConfig",
    "CacheConfig",
    "AuthConfig",
    "VectorDBConfig",
    "DocumentConfig",
    "SystemConfig",
    "ConfigUpdate",
    "ConfigValidationResult",
    "HealthStatus",
    # New exports
    "ModelConfig",
    "ModelProviderConfig", 
    "ConfigResponse"
]