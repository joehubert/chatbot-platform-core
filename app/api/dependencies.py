"""
API dependencies for the chatbot platform.

This module provides common dependencies used across API endpoints including
database sessions, authentication, rate limiting, and other shared functionality.
"""

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
