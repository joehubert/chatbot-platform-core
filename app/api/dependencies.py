"""
API dependencies for the chatbot platform.

This module provides common dependencies used across API endpoints including
database sessions, authentication, rate limiting, and other shared functionality.
"""

from app.core.config import get_settings
from app.services.cache import SemanticCacheService
from app.services.cache import ConversationCacheService
from app.services.model_factory import ModelFactory
from app.services.model_router import ModelRouter
from app.services.rate_limiting import RateLimitService
from app.services.vector_db import create_vector_db_service, VectorDBService
from app.services.knowledge_base import create_knowledge_base_service, KnowledgeBaseService
from app.services.document_processor import DocumentProcessor

from typing import Optional, Annotated
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import redis
from datetime import datetime, timedelta

from app.core.database import get_sync_db
from app.core.config import get_settings
from app.services.auth_service import AuthService
from app.services.rate_limiting import RateLimitService
from app.schemas.auth import UserProfile as User
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Security scheme for bearer token authentication
security = HTTPBearer(auto_error=False)

# Redis connection for rate limiting and caching
redis_client = redis.Redis.from_url(settings.REDIS_URL)


def get_db() -> Session:
    """
    Dependency to get database session.

    Provides a database session that is automatically closed after use.
    This is the primary way to access the database in API endpoints.
    """
    db = get_sync_db()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """
    Dependency to get Redis client.

    Provides access to Redis for caching, rate limiting, and session storage.
    """
    return redis_client


async def get_current_user(
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[User]:
    """
    Get the current authenticated user from the request.

    This dependency extracts the bearer token from the Authorization header
    and validates it to return the associated user. Returns None if no valid
    authentication is provided.

    Args:
        db: Database session
        credentials: Bearer token credentials from the Authorization header

    Returns:
        User object if authenticated, None otherwise

    Raises:
        HTTPException: If token is invalid or expired
    """
    if not credentials:
        return None

    try:
        auth_service = AuthService(db)
        user = auth_service.get_user_from_token(credentials.credentials)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user

    except Exception as e:
        logger.error(f"Error validating user token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get the current authenticated and active user.

    This dependency ensures that not only is the user authenticated,
    but their account is also active and not suspended.

    Args:
        current_user: The authenticated user from get_current_user

    Returns:
        Active user object

    Raises:
        HTTPException: If user is not authenticated or not active
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive"
        )

    return current_user


async def check_rate_limit(
    request: Request, redis_client: redis.Redis = Depends(get_redis)
) -> bool:
    """
    Check if the request is within rate limits.

    Implements token bucket rate limiting based on IP address and
    optional user identification. Supports both per-user and global limits.

    Args:
        request: FastAPI request object
        redis_client: Redis client for rate limit storage

    Returns:
        True if request is allowed

    Raises:
        HTTPException: If rate limit is exceeded (HTTP 429)
    """
    try:
        rate_limit_service = RateLimitService(redis_client)

        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Check user-specific rate limit if user is identified
        user_id = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = str(request.state.user.id)

        # Check rate limits
        is_allowed, remaining, reset_time = await rate_limit_service.check_rate_limit(
            identifier=client_ip, user_id=user_id, endpoint=str(request.url.path)
        )

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_ip} (user: {user_id})")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(
                        max(0, (reset_time - datetime.now()).total_seconds())
                    ),
                },
            )

        # Add rate limit info to response headers (will be added by middleware)
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = reset_time

        return True

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking rate limit: {str(e)}")
        # Allow request to proceed if rate limiting fails
        return True


async def check_admin_permissions(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Check if the current user has administrator permissions.

    This dependency ensures the user is authenticated, active, and has
    admin privileges for accessing administrative endpoints.

    Args:
        current_user: The authenticated and active user

    Returns:
        User object with admin permissions

    Raises:
        HTTPException: If user lacks admin permissions
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions required",
        )

    return current_user


async def validate_session_id(session_id: str, db: Session = Depends(get_db)) -> str:
    """
    Validate that a session ID exists and is active.

    This dependency checks that the provided session ID corresponds
    to an existing, non-expired session in the database.

    Args:
        session_id: The session ID to validate
        db: Database session

    Returns:
        The validated session ID

    Raises:
        HTTPException: If session is invalid or expired
    """
    try:
        from app.services.session_service import SessionService

        session_service = SessionService(db)
        session = session_service.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        if session_service.is_session_expired(session):
            raise HTTPException(
                status_code=status.HTTP_410_GONE, detail="Session has expired"
            )

        return session_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating session ID: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating session",
        )


def get_pagination_params(
    skip: int = 0, limit: int = 20, max_limit: int = 100
) -> tuple[int, int]:
    """
    Get and validate pagination parameters.

    This dependency provides standardized pagination parameters
    with validation to prevent excessive data retrieval.

    Args:
        skip: Number of records to skip (offset)
        limit: Maximum number of records to return
        max_limit: Maximum allowed limit value

    Returns:
        Tuple of (skip, limit) with validated values

    Raises:
        HTTPException: If pagination parameters are invalid
    """
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative",
        )

    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit parameter must be positive",
        )

    if limit > max_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Limit parameter cannot exceed {max_limit}",
        )

    return skip, limit


async def check_content_type(
    request: Request, allowed_types: list[str] = ["application/json"]
) -> bool:
    """
    Check if the request content type is allowed.

    This dependency validates that the request has an acceptable
    Content-Type header for endpoints that require specific formats.

    Args:
        request: FastAPI request object
        allowed_types: List of allowed content types

    Returns:
        True if content type is allowed

    Raises:
        HTTPException: If content type is not allowed
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip()

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type. Allowed types: {', '.join(allowed_types)}",
        )

    return True


def get_client_info(request: Request) -> dict:
    """
    Extract client information from the request.

    This dependency extracts useful client information such as
    IP address, user agent, and other metadata for logging and analytics.

    Args:
        request: FastAPI request object

    Returns:
        Dictionary containing client information
    """
    return {
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", ""),
        "referer": request.headers.get("referer", ""),
        "accept_language": request.headers.get("accept-language", ""),
        "forwarded_for": request.headers.get("x-forwarded-for", ""),
        "real_ip": request.headers.get("x-real-ip", ""),
        "request_time": datetime.now().isoformat(),
    }

async def get_semantic_cache_service() -> SemanticCacheService:
    """Dependency injection for semantic cache service"""
    settings = get_settings()
    
    # Create cache service with proper configuration
    cache_service = SemanticCacheService(
        redis_url=settings.REDIS_URL,
        ttl_hours=settings.CACHE_TTL_HOURS,
        similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD
    )
    
    # Initialize if not already done
    if not cache_service._initialized:
        await cache_service.initialize()
    
    return cache_service


async def get_vector_db_service() -> VectorDBService:
    """Dependency injection for vector database service"""
    settings = get_settings()
    
    # Use the factory function with proper configuration
    vector_service = create_vector_db_service(
        db_type=settings.VECTOR_DB_TYPE,
        connection_params={
            "api_key": settings.VECTOR_DB_API_KEY,
            "environment": settings.VECTOR_DB_ENVIRONMENT,
            "host": settings.VECTOR_DB_HOST,
            "port": settings.VECTOR_DB_PORT,
        },
        index_name=settings.VECTOR_DB_INDEX_NAME,
        dimension=settings.EMBEDDING_DIMENSION,
        similarity_metric=settings.VECTOR_SIMILARITY_METRIC
    )
    
    # Initialize if not already done
    if not vector_service._initialized:
        await vector_service.initialize()
    
    return vector_service


async def get_document_processor() -> DocumentProcessor:
    """Dependency injection for document processor"""
    settings = get_settings()
    
    # Create document processor with configuration
    doc_processor = DocumentProcessor(
        max_file_size_mb=settings.MAX_FILE_SIZE_MB,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        allowed_file_types=settings.ALLOWED_FILE_TYPES
    )
    
    # Initialize if not already done
    if not doc_processor._initialized:
        await doc_processor.initialize()
    
    return doc_processor


async def get_knowledge_base_service() -> KnowledgeBaseService:
    """Dependency injection for knowledge base service"""
    # Get the required dependencies
    vector_service = await get_vector_db_service()
    doc_processor = await get_document_processor()
    
    # Use the factory function
    knowledge_service = create_knowledge_base_service(
        vector_db_service=vector_service,
        document_processor=doc_processor,
        default_expiration_days=365
    )
    
    # Initialize if not already done
    if not knowledge_service._initialized:
        await knowledge_service.initialize()
    
    return knowledge_service


async def get_model_factory() -> ModelFactory:
    """Dependency injection for model factory"""
    settings = get_settings()

    # Create model factory with configuration
    model_factory = ModelFactory(
        default_provider=settings.DEFAULT_LLM_PROVIDER,
        model_configs=settings.MODEL_CONFIGS
    )
    
    # Initialize if not already done
    if not model_factory._initialized:
        await model_factory.initialize()

    return model_factory


async def get_model_router() -> ModelRouter:
    """Dependency injection for model router"""
    model_factory = await get_model_factory()

    # Create model router with model factory (fix parameter name)
    model_router = ModelRouter(model_service=model_factory)  # Changed from llm_factory to model_service

    # Initialize if not already done
    if not model_router._initialized:
        await model_router.initialize()
    
    return model_router


async def get_rate_limit_service() -> RateLimitService:
    """Dependency injection for rate limiting service"""
    settings = get_settings()
    
    # Create rate limiting service with configuration
    rate_limit_service = RateLimitService(
        redis_url=settings.REDIS_URL,
        per_user_per_minute=settings.RATE_LIMIT_PER_USER_PER_MINUTE,
        global_per_minute=settings.RATE_LIMIT_GLOBAL_PER_MINUTE
    )
    
    # Initialize if not already done
    if not rate_limit_service._initialized:
        await rate_limit_service.initialize()
    
    return rate_limit_service