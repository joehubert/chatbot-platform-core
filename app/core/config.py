"""
Core configuration management for the Turnkey AI Chatbot platform.
Handles environment variables, validation, and configuration settings.
"""

import os
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import secrets
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.networks import AnyHttpUrl
from app.core.model_enums import ModelType, ModelProvider
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import json

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # or True, see note below
        extra="ignore",  # We will change this when we have time to a better approach: docker-compose.env and app.env
    )

    # Core Application Configuration
    PROJECT_NAME: str = "Turnkey AI Chatbot"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    # Environment
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")

    # Database Configuration
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")  # Add this line
    DATABASE_MAX_OVERFLOW: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    DATABASE_POOL_RECYCLE: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")

    # Redis Configuration
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_POOL_SIZE: int = Field(default=20, env="REDIS_POOL_SIZE")
    REDIS_SOCKET_TIMEOUT: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(
        default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT"
    )

    # LLM Provider Configuration
    DEFAULT_MODEL_PROVIDER: ModelProvider = Field(default=ModelProvider.OPENAI, env="DEFAULT_MODEL_PROVIDER")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")

    # ChromaDB Configuration,etc.
    chroma_host: str = Field(default="chroma", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    node_env: str = Field(default="development", env="NODE_ENV")
    show_detailed_errors: bool = Field(default=True, env="SHOW_DETAILED_ERRORS")

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
        default=None, env="AZURE_OPENAI_ENDPOINT"
    )
    AZURE_OPENAI_API_KEY: Optional[str] = Field(
        default=None, env="AZURE_OPENAI_API_KEY"
    )
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2023-12-01-preview", env="AZURE_OPENAI_API_VERSION"
    )

    # Vector Database Configuration
    VECTOR_DB_TYPE: str = Field(default="pinecone", env="VECTOR_DB_TYPE")
    VECTOR_DB_HOST: str = Field(default="localhost", env="VECTOR_DB_HOST")
    VECTOR_DB_PORT: int = Field(default=8000, env="VECTOR_DB_PORT")

    VECTOR_DB_API_KEY: Optional[str] = Field(default=None, env="VECTOR_DB_API_KEY")
    VECTOR_DB_ENVIRONMENT: Optional[str] = Field(
        default=None, env="VECTOR_DB_ENVIRONMENT"
    )
    VECTOR_DB_INDEX_NAME: str = Field(
        default="chatbot-knowledge", env="VECTOR_DB_INDEX_NAME"
    )

    EMBEDDING_DIMENSION: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    VECTOR_SIMILARITY_METRIC: str = Field(default="cosine", env="VECTOR_SIMILARITY_METRIC")

    # Rate Limiting Configuration
    # Core requirements from reqs.md
    RATE_LIMIT_PER_USER_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_USER_PER_MINUTE")
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = Field(default=1000, env="RATE_LIMIT_GLOBAL_PER_MINUTE")

    # Burst capacity for each type (allows short bursts above the per-minute rate)
    RATE_LIMIT_USER_BURST_CAPACITY: int = Field(default=10, env="RATE_LIMIT_USER_BURST_CAPACITY")
    RATE_LIMIT_GLOBAL_BURST_CAPACITY: int = Field(default=50, env="RATE_LIMIT_GLOBAL_BURST_CAPACITY")

    # Optional: Additional rate limiting dimensions for more granular control
    RATE_LIMIT_PER_IP_PER_MINUTE: int = Field(default=120, env="RATE_LIMIT_PER_IP_PER_MINUTE")
    RATE_LIMIT_IP_BURST_CAPACITY: int = Field(default=20, env="RATE_LIMIT_IP_BURST_CAPACITY")

    RATE_LIMIT_PER_SESSION_PER_MINUTE: int = Field(default=30, env="RATE_LIMIT_PER_SESSION_PER_MINUTE")
    RATE_LIMIT_SESSION_BURST_CAPACITY: int = Field(default=5, env="RATE_LIMIT_SESSION_BURST_CAPACITY")

    # Cache Configuration
    CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.85, env="CACHE_SIMILARITY_THRESHOLD"
    )
    CACHE_TTL_HOURS: int = Field(default=24, env="CACHE_TTL_HOURS")
    CACHE_MAX_ENTRIES: int = Field(default=10000, env="CACHE_MAX_ENTRIES")


    # Authentication Configuration
    AUTH_SESSION_TIMEOUT_MINUTES: int = Field(
        default=30, env="AUTH_SESSION_TIMEOUT_MINUTES"
    )
    OTP_EXPIRY_MINUTES: int = Field(default=5, env="OTP_EXPIRY_MINUTES")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")

    # SMS Provider Configuration
    SMS_PROVIDER: str = Field(default="twilio", env="SMS_PROVIDER")
    TWILIO_ACCOUNT_SID: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    TWILIO_FROM_NUMBER: Optional[str] = Field(default=None, env="TWILIO_FROM_NUMBER")

    # Email Provider Configuration
    EMAIL_PROVIDER: str = Field(default="sendgrid", env="EMAIL_PROVIDER")
    SENDGRID_API_KEY: Optional[str] = Field(default=None, env="SENDGRID_API_KEY")
    SENDGRID_FROM_EMAIL: Optional[str] = Field(default=None, env="SENDGRID_FROM_EMAIL")

    # SMTP Configuration (alternative to SendGrid)
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    SMTP_USE_TLS: bool = Field(default=True, env="SMTP_USE_TLS")

    # Model Configuration
    RELEVANCE_MODEL: str = Field(default="gpt-3.5-turbo", env="RELEVANCE_MODEL")
    SIMPLE_QUERY_MODEL: str = Field(default="gpt-3.5-turbo", env="SIMPLE_QUERY_MODEL")
    COMPLEX_QUERY_MODEL: str = Field(default="gpt-4", env="COMPLEX_QUERY_MODEL")
    CLARIFICATION_MODEL: str = Field(default="gpt-3.5-turbo", env="CLARIFICATION_MODEL")
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-ada-002", env="EMBEDDING_MODEL"
    )

    # Model Routing Configuration
    COMPLEXITY_THRESHOLD_SIMPLE: float = Field(
        default=0.3, env="COMPLEXITY_THRESHOLD_SIMPLE"
    )
    COMPLEXITY_THRESHOLD_COMPLEX: float = Field(
        default=0.7, env="COMPLEXITY_THRESHOLD_COMPLEX"
    )
    RELEVANCE_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6, env="RELEVANCE_CONFIDENCE_THRESHOLD"
    )
    MAX_CLARIFICATION_ATTEMPTS: int = Field(default=3, env="MAX_CLARIFICATION_ATTEMPTS")

    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["pdf", "txt", "docx", "md"], env="ALLOWED_FILE_TYPES"
    )
    UPLOAD_DIR: str = Field(default="uploads", env="UPLOAD_DIR")

    # Processing Configuration
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    MAX_CHUNKS_PER_DOCUMENT: int = Field(default=1000, env="MAX_CHUNKS_PER_DOCUMENT")

    # Error Handling Configuration
    FALLBACK_ERROR_MESSAGE: str = Field(
        default="I'm having trouble right now. Please contact support at support@company.com",
        env="FALLBACK_ERROR_MESSAGE",
    )
    MAX_RETRY_ATTEMPTS: int = Field(default=3, env="MAX_RETRY_ATTEMPTS")
    RETRY_DELAY_SECONDS: float = Field(default=1.0, env="RETRY_DELAY_SECONDS")

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    LOG_MAX_SIZE_MB: int = Field(default=100, env="LOG_MAX_SIZE_MB")
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = Field(
        default=[], env="BACKEND_CORS_ORIGINS"
    )

    # Security Configuration
    BCRYPT_ROUNDS: int = Field(default=12, env="BCRYPT_ROUNDS")
    CSRF_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    HEALTH_CHECK_TIMEOUT: int = Field(default=30, env="HEALTH_CHECK_TIMEOUT")

    # MCP Server Configuration
    MCP_REQUEST_TIMEOUT: int = Field(default=30, env="MCP_REQUEST_TIMEOUT")
    MCP_MAX_RETRIES: int = Field(default=3, env="MCP_MAX_RETRIES")
    MCP_RETRY_DELAY: float = Field(default=1.0, env="MCP_RETRY_DELAY")
    MCP_HEALTH_CHECK_INTERVAL: int = Field(default=60, env="MCP_HEALTH_CHECK_INTERVAL")
    MCP_CONFIG_FILE: Optional[str] = Field(default=None, env="MCP_CONFIG_FILE")

        # LLM Provider Configuration
    DEFAULT_MODEL_PROVIDER: ModelProvider = Field(default=ModelProvider.OPENAI, env="DEFAULT_MODEL_PROVIDER")
    
    # API Keys for different providers
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    
    # Model Type Mappings - JSON string that gets parsed
    MODEL_TYPE_CONFIGS: str = Field(
        default='{}', 
        env="MODEL_TYPE_CONFIGS",
        description="JSON string mapping model types to provider/model combinations"
    )
    
    @field_validator('MODEL_TYPE_CONFIGS')
    @classmethod
    def validate_model_configs(cls, v: str) -> str:
        """Validate that MODEL_TYPE_CONFIGS is valid JSON"""
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError as e:
            raise ValueError(f"MODEL_TYPE_CONFIGS must be valid JSON: {e}")
    
    def get_model_type_configs(self) -> Dict[str, Dict[str, Any]]:
        """Parse and return model type configurations"""
        if not self.MODEL_TYPE_CONFIGS or self.MODEL_TYPE_CONFIGS == '{}':
            return self._get_default_model_configs()
        
        try:
            configs = json.loads(self.MODEL_TYPE_CONFIGS)
            # Validate the structure
            for model_type, config in configs.items():
                if model_type not in [mt.value for mt in ModelType]:
                    raise ValueError(f"Unknown model type: {model_type}")
                if 'provider' not in config or 'model' not in config:
                    raise ValueError(f"Model config for {model_type} must have 'provider' and 'model'")
            return configs
        except Exception as e:
            logger.warning(f"Failed to parse MODEL_TYPE_CONFIGS, using defaults: {e}")
            return self._get_default_model_configs()
    
    def _get_default_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Default model configurations if none provided"""
        return {
            ModelType.RELEVANCE.value: {
                "provider": self.DEFAULT_MODEL_PROVIDER.value,
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "temperature": 0.1
            },
            ModelType.SIMPLE_QUERY.value: {
                "provider": self.DEFAULT_MODEL_PROVIDER.value,
                "model": "gpt-3.5-turbo",
                "max_tokens": 500,
                "temperature": 0.3
            },
            ModelType.COMPLEX_QUERY.value: {
                "provider": self.DEFAULT_MODEL_PROVIDER.value,
                "model": "gpt-4",
                "max_tokens": 2000,
                "temperature": 0.7
            },
            ModelType.CLARIFICATION.value: {
                "provider": self.DEFAULT_MODEL_PROVIDER.value,
                "model": "gpt-3.5-turbo",
                "max_tokens": 300,
                "temperature": 0.5
            },
            ModelType.EMBEDDING.value: {
                "provider": self.DEFAULT_MODEL_PROVIDER.value,
                "model": "text-embedding-ada-002",
                "dimensions": 1536
            }
        }

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production", "test"]
        if v.lower() not in allowed_environments:
            raise ValueError(f"ENVIRONMENT must be one of: {allowed_environments}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {allowed_levels}")
        return v.upper()

    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format setting."""
        allowed_formats = ["json", "text"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"LOG_FORMAT must be one of: {allowed_formats}")
        return v.lower()

    @field_validator("VECTOR_DB_TYPE")
    @classmethod
    def validate_vector_db_type(cls, v: str) -> str:
        """Validate vector database type."""
        allowed_types = ["pinecone", "chroma", "weaviate", "pgvector"]
        if v.lower() not in allowed_types:
            raise ValueError(f"VECTOR_DB_TYPE must be one of: {allowed_types}")
        return v.lower()

    @field_validator("SMS_PROVIDER")
    @classmethod
    def validate_sms_provider(cls, v: str) -> str:
        """Validate SMS provider setting."""
        allowed_providers = ["twilio", "aws_sns", "custom"]
        if v.lower() not in allowed_providers:
            raise ValueError(f"SMS_PROVIDER must be one of: {allowed_providers}")
        return v.lower()

    @field_validator("EMAIL_PROVIDER")
    @classmethod
    def validate_email_provider(cls, v: str) -> str:
        """Validate email provider setting."""
        allowed_providers = ["sendgrid", "smtp", "aws_ses", "custom"]
        if v.lower() not in allowed_providers:
            raise ValueError(f"EMAIL_PROVIDER must be one of: {allowed_providers}")
        return v.lower()

    @field_validator("ALLOWED_FILE_TYPES", mode="before")
    @classmethod
    def validate_file_types(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate and parse allowed file types."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return [ext.lower() for ext in v]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate and parse CORS origins."""
        if isinstance(v, str) and v:
            return [origin.strip() for origin in v.split(",")]
        return v or []

    @field_validator("CACHE_SIMILARITY_THRESHOLD")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("CACHE_SIMILARITY_THRESHOLD must be between 0 and 1")
        return v

    @field_validator("COMPLEXITY_THRESHOLD_SIMPLE", "COMPLEXITY_THRESHOLD_COMPLEX")
    @classmethod
    def validate_complexity_thresholds(cls, v: float) -> float:
        """Validate complexity thresholds are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Complexity thresholds must be between 0 and 1")
        return v

    @field_validator("RELEVANCE_CONFIDENCE_THRESHOLD")
    @classmethod
    def validate_relevance_threshold(cls, v: float) -> float:
        """Validate relevance confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("RELEVANCE_CONFIDENCE_THRESHOLD must be between 0 and 1")
        return v

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self.ENVIRONMENT == "test"

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "pool_timeout": self.DATABASE_POOL_TIMEOUT,
            "pool_recycle": self.DATABASE_POOL_RECYCLE,
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary."""
        return {
            "url": self.REDIS_URL,
            "password": self.REDIS_PASSWORD,
            "db": self.REDIS_DB,
            "max_connections": self.REDIS_POOL_SIZE,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": self.REDIS_SOCKET_CONNECT_TIMEOUT,
        }

    def get_llm_provider_config(self) -> Dict[str, Dict[str, Any]]:
        """Get LLM provider configuration dictionary."""
        return {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "enabled": bool(self.OPENAI_API_KEY),
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "enabled": bool(self.ANTHROPIC_API_KEY),
            },
            "huggingface": {
                "api_key": self.HUGGINGFACE_API_KEY,
                "enabled": bool(self.HUGGINGFACE_API_KEY),
            },
            "azure_openai": {
                "endpoint": self.AZURE_OPENAI_ENDPOINT,
                "api_key": self.AZURE_OPENAI_API_KEY,
                "api_version": self.AZURE_OPENAI_API_VERSION,
                "enabled": bool(
                    self.AZURE_OPENAI_ENDPOINT and self.AZURE_OPENAI_API_KEY
                ),
            },
        }

    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration dictionary."""
        config = {"type": self.VECTOR_DB_TYPE}

        if self.VECTOR_DB_TYPE == "pinecone":
            config.update(
                {
                    "api_key": self.PINECONE_API_KEY,
                    "environment": self.PINECONE_ENVIRONMENT,
                    "index_name": self.PINECONE_INDEX_NAME,
                }
            )

        return config


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
