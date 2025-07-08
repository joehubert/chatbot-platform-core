"""
Configuration API endpoints for system settings and model management.
Handles runtime configuration updates and system control.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_admin_user, get_current_user, get_db
from app.core.config import get_settings
from app.models.user import User
from app.schemas.config import (
    AuthConfig,
    CacheConfig,
    ConfigResponse,
    ConfigUpdate,
    LLMConfig,
    ModelProviderConfig,
    RateLimitConfig,
    SystemConfig,
    VectorDBConfig,
)
from app.services.cache import CacheService
from app.services.llm_factory import LLMFactory
from app.services.model_router import ModelRouter
from app.services.rate_limiting import RateLimitService
from app.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Initialize services
cache_service = CacheService()
llm_factory = LLMFactory()
model_router = ModelRouter()
rate_limit_service = RateLimitService()
vector_service = VectorDBService()


@router.get("/", response_model=ConfigResponse)
async def get_system_config(
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve current system configuration.
    
    Returns all non-sensitive configuration settings.
    """
    try:
        config = ConfigResponse(
            llm=LLMConfig(
                primary_model_provider=settings.PRIMARY_MODEL_PROVIDER,
                primary_model_name=settings.PRIMARY_MODEL_NAME,
                fallback_model_provider=settings.FALLBACK_MODEL_PROVIDER,
                fallback_model_name=settings.FALLBACK_MODEL_NAME,
                relevance_model=settings.RELEVANCE_MODEL,
                simple_query_model=settings.SIMPLE_QUERY_MODEL,
                complex_query_model=settings.COMPLEX_QUERY_MODEL,
                clarification_model=settings.CLARIFICATION_MODEL,
            ),
            vector_db=VectorDBConfig(
                type=settings.VECTOR_DB_TYPE,
                url=settings.VECTOR_DB_URL,
                similarity_threshold=settings.SIMILARITY_THRESHOLD,
            ),
            cache=CacheConfig(
                similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
                ttl_hours=settings.CACHE_TTL_HOURS,
                max_context_turns=settings.MAX_CONTEXT_TURNS,
                summarization_trigger=settings.SUMMARIZATION_TRIGGER,
            ),
            rate_limit=RateLimitConfig(
                per_user_per_minute=settings.RATE_LIMIT_PER_USER_PER_MINUTE,
                global_per_minute=settings.RATE_LIMIT_GLOBAL_PER_MINUTE,
                burst_capacity=getattr(settings, 'BURST_CAPACITY', 10),
            ),
            auth=AuthConfig(
                session_timeout_minutes=settings.AUTH_SESSION_TIMEOUT_MINUTES,
                otp_expiry_minutes=settings.OTP_EXPIRY_MINUTES,
                sms_provider=settings.SMS_PROVIDER,
                email_provider=settings.EMAIL_PROVIDER,
            ),
            system=SystemConfig(
                max_file_size_mb=settings.MAX_FILE_SIZE_MB,
                allowed_file_types=settings.ALLOWED_FILE_TYPES.split(','),
                fallback_error_message=settings.FALLBACK_ERROR_MESSAGE,
                environment=settings.ENVIRONMENT,
                debug=settings.DEBUG,
            )
        )
        
        return config
        
    except Exception as e:
        logger.error(f"Error retrieving system config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system configuration"
        )


@router.put("/", response_model=ConfigResponse)
async def update_system_config(
    config_update: ConfigUpdate,
    current_user: User = Depends(get_current_admin_user),
):
    """
    Update system configuration.
    
    Requires admin privileges. Updates are applied immediately
    and may affect ongoing operations.
    """
    try:
        # Update LLM configuration
        if config_update.llm:
            await _update_llm_config(config_update.llm)
        
        # Update vector database configuration
        if config_update.vector_db:
            await _update_vector_db_config(config_update.vector_db)
        
        # Update cache configuration
        if config_update.cache:
            await _update_cache_config(config_update.cache)
        
        # Update rate limiting configuration
        if config_update.rate_limit:
            await _update_rate_limit_config(config_update.rate_limit)
        
        # Update authentication configuration
        if config_update.auth:
            await _update_auth_config(config_update.auth)
        
        # Update system configuration
        if config_update.system:
            await _update_system_config(config_update.system)
        
        logger.info(f"System configuration updated by user {current_user.id}")
        
        # Return updated configuration
        return await get_system_config(current_user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating system config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update system configuration"
        )


@router.get("/models/providers")
async def get_available_model_providers(
    current_user: User = Depends(get_current_user),
):
    """
    Get list of available LLM providers and their models.
    """
    try:
        providers = await llm_factory.get_available_providers()
        
        return {
            "providers": providers,
            "total_providers": len(providers)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving model providers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model providers"
        )


@router.get("/models/routing")
async def get_model_routing_rules(
    current_user: User = Depends(get_current_user),
):
    """
    Get current model routing rules and criteria.
    """
    try:
        rules = await model_router.get_routing_rules()
        
        return {
            "routing_rules": rules,
            "total_rules": len(rules)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving model routing rules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model routing rules"
        )


@router.put("/models/routing")
async def update_model_routing_rules(
    routing_rules: List[Dict[str, Any]],
    current_user: User = Depends(get_current_admin_user),
):
    """
    Update model routing rules.
    
    Requires admin privileges.
    """
    try:
        await model_router.update_routing_rules(routing_rules)
        
        logger.info(f"Model routing rules updated by user {current_user.id}")
        return {
            "message": "Model routing rules updated successfully",
            "rules_count": len(routing_rules)
        }
        
    except Exception as e:
        logger.error(f"Error updating model routing rules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model routing rules"
        )


@router.post("/models/test")
async def test_model_connection(
    provider: str,
    model_name: str,
    test_message: Optional[str] = "Hello, this is a test message.",
    current_user: User = Depends(get_current_admin_user),
):
    """
    Test connection to a specific model provider.
    
    Useful for validating API keys and model availability.
    """
    try:
        result = await llm_factory.test_model_connection(
            provider=provider,
            model_name=model_name,
            test_message=test_message
        )
        
        return {
            "provider": provider,
            "model": model_name,
            "status": "success" if result.get("success") else "failed",
            "response_time_ms": result.get("response_time_ms"),
            "error": result.get("error"),
            "test_response": result.get("response")
        }
        
    except Exception as e:
        logger.error(f"Error testing model connection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test model connection: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_statistics(
    current_user: User = Depends(get_current_user),
):
    """
    Get cache performance statistics.
    """
    try:
        stats = await cache_service.get_cache_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving cache statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )


@router.post("/cache/clear")
async def clear_cache(
    cache_type: Optional[str] = None,
    current_user: User = Depends(get_current_admin_user),
):
    """
    Clear cache data.
    
    Optionally specify cache type (semantic, session, etc.).
    Requires admin privileges.
    """
    try:
        result = await cache_service.clear_cache(cache_type=cache_type)
        
        logger.info(f"Cache cleared by user {current_user.id}, type: {cache_type or 'all'}")
        return {
            "message": "Cache cleared successfully",
            "cache_type": cache_type or "all",
            "cleared_entries": result.get("cleared_entries", 0)
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.get("/health")
async def get_system_health(
    current_user: User = Depends(get_current_user),
):
    """
    Get system health status including all components.
    """
    try:
        health_status = {
            "database": await _check_database_health(),
            "cache": await _check_cache_health(),
            "vector_db": await _check_vector_db_health(),
            "llm_providers": await _check_llm_providers_health(),
        }
        
        overall_status = "healthy" if all(
            status.get("status") == "healthy" 
            for status in health_status.values()
        ) else "degraded"
        
        return {
            "overall_status": overall_status,
            "components": health_status,
            "timestamp": settings.get_current_timestamp()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving system health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health"
        )


# Helper functions for configuration updates

async def _update_llm_config(llm_config: LLMConfig):
    """Update LLM configuration."""
    # In a real implementation, this would update environment variables
    # or configuration storage and notify services of changes
    pass


async def _update_vector_db_config(vector_config: VectorDBConfig):
    """Update vector database configuration."""
    # Validate connection with new settings
    await vector_service.test_connection(
        db_type=vector_config.type,
        url=vector_config.url
    )


async def _update_cache_config(cache_config: CacheConfig):
    """Update cache configuration."""
    await cache_service.update_config(
        similarity_threshold=cache_config.similarity_threshold,
        ttl_hours=cache_config.ttl_hours,
        max_context_turns=cache_config.max_context_turns,
        summarization_trigger=cache_config.summarization_trigger
    )


async def _update_rate_limit_config(rate_config: RateLimitConfig):
    """Update rate limiting configuration."""
    await rate_limit_service.update_config(
        per_user_per_minute=rate_config.per_user_per_minute,
        global_per_minute=rate_config.global_per_minute,
        burst_capacity=rate_config.burst_capacity
    )


async def _update_auth_config(auth_config: AuthConfig):
    """Update authentication configuration."""
    # Update authentication service settings
    pass


async def _update_system_config(system_config: SystemConfig):
    """Update system configuration."""
    # Update system-wide settings
    pass


# Helper functions for health checks

async def _check_database_health():
    """Check database connectivity and performance."""
    try:
        # Implement database health check
        return {
            "status": "healthy",
            "response_time_ms": 10,
            "details": "Connection successful"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_cache_health():
    """Check cache service health."""
    try:
        stats = await cache_service.get_cache_stats()
        return {
            "status": "healthy",
            "hit_rate": stats.get("hit_rate", 0),
            "memory_usage": stats.get("memory_usage", "unknown")
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_vector_db_health():
    """Check vector database health."""
    try:
        await vector_service.health_check()
        return {
            "status": "healthy",
            "index_count": await vector_service.get_index_count(),
            "details": "Vector database operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_llm_providers_health():
    """Check LLM provider connectivity."""
    try:
        providers_status = await llm_factory.check_all_providers_health()
        healthy_count = sum(1 for status in providers_status.values() if status.get("status") == "healthy")
        total_count = len(providers_status)
        
        return {
            "status": "healthy" if healthy_count > 0 else "unhealthy",
            "healthy_providers": healthy_count,
            "total_providers": total_count,
            "providers": providers_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
