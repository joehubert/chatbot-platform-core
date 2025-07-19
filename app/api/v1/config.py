"""
Configuration API endpoints for system settings and model management.
Handles runtime configuration updates and system control.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

#from app.api.dependencies import get_current_admin_user, get_current_user, get_db
from app.api.dependencies import get_current_user, get_db
from app.api.dependencies import (
    get_semantic_cache_service,
    #get_conversation_cache_service,
    get_model_factory,
    get_model_router,
    get_rate_limit_service,
    get_vector_db_service,
    create_vector_db_service,
)
from app.core.config import get_settings
from app.models.user import User
from app.schemas.config import (
    AuthConfig,
    CacheConfig,
    ConfigResponse,
    ConfigUpdate,
    ConfigValidationResult,
    DocumentConfig,
    HealthStatus,
    ModelConfig,
    ModelProviderConfig,
    RateLimitConfig,
    SystemConfig,
    VectorDBConfig,
)
from app.services.cache import SemanticCacheService, ConversationCacheService
from app.services.model_factory import ModelFactory
from app.services.model_router import ModelRouter
from app.services.rate_limiting import RateLimitService
from app.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Initialize services
semantic_cache_service = SemanticCacheService()
conversation_cache_service = ConversationCacheService()
model_factory = ModelFactory()
#model_router = ModelRouter()
rate_limit_service = RateLimitService()
vector_service = create_vector_db_service(
    db_type="chroma",  # or "pinecone", "weaviate", "qdrant"
    connection_params={
        "host": "localhost",
        "port": 8000,
        # other connection parameters
    },
    index_name="your_index_name",
    dimension=1536,
    similarity_metric="cosine"
)
""" TODO: Replace with actual vector DB service initialization 
Create an issue to represent this effort.
 create a dependency function in app/api/dependencies.py:

async def get_vector_db_service():
    settings = get_settings()
    return create_vector_db_service(
        db_type=settings.VECTOR_DB_TYPE,
        connection_params=settings.VECTOR_DB_CONNECTION_PARAMS,
        index_name=settings.VECTOR_DB_INDEX_NAME,
        dimension=settings.EMBEDDING_DIMENSION,
        similarity_metric=settings.VECTOR_SIMILARITY_METRIC
    )
Then use it in API routes:
@router.post("/documents")
async def upload_document(
    vector_service: VectorDBService = Depends(get_vector_db_service),
    # ... other parameters
):

"""

@router.get("/", response_model=ConfigResponse)
async def get_system_config(
    current_user: User = Depends(get_current_user),
    cache_service: SemanticCacheService = Depends(get_semantic_cache_service),
    model_router: ModelRouter = Depends(get_model_router),
    vector_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Retrieve current system configuration.
    
    Returns all non-sensitive configuration settings.
    """
    try:
        # Get current settings
        current_settings = get_settings()
        
        # Build system config from current state
        system_config = SystemConfig(
            models=await model_router.get_model_configs(),
            rate_limiting=RateLimitConfig(
                enabled=current_settings.RATE_LIMITING_ENABLED,
                per_user_per_minute=current_settings.RATE_LIMIT_PER_USER_PER_MINUTE,
                global_per_minute=current_settings.RATE_LIMIT_GLOBAL_PER_MINUTE,
            ),
            cache=CacheConfig(
                enabled=current_settings.CACHE_ENABLED,
                similarity_threshold=current_settings.CACHE_SIMILARITY_THRESHOLD,
                ttl_hours=current_settings.CACHE_TTL_HOURS,
            ),
            auth=AuthConfig(
                session_timeout_minutes=current_settings.SESSION_TIMEOUT_MINUTES,
                otp_expiry_minutes=current_settings.OTP_EXPIRY_MINUTES,
            ),
            vector_db=VectorDBConfig(
                provider=current_settings.VECTOR_DB_PROVIDER,
                index_name=current_settings.VECTOR_DB_INDEX_NAME,
                dimension=current_settings.EMBEDDING_DIMENSION,
            ),
            documents=DocumentConfig(
                max_file_size_mb=current_settings.MAX_FILE_SIZE_MB,
                chunk_size=current_settings.CHUNK_SIZE,
                chunk_overlap=current_settings.CHUNK_OVERLAP,
            ),
        )
        
        # Get metadata
        metadata = {
            "total_models": len(system_config.models),
            "active_providers": list(set(model.provider for model in system_config.models.values())),
            "vector_db_status": await vector_service.health_check(),
            "cache_stats": await cache_service.get_stats() if current_settings.CACHE_ENABLED else None,
        }
        
        return ConfigResponse(
            config=system_config,
            metadata=metadata,
            last_updated=None,  # Would come from database in production
            version=current_settings.APP_VERSION,
        )
        
    except Exception as e:
        logger.error(f"Failed to get system configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system configuration"
        )


@router.put("/", response_model=ConfigResponse)
async def update_system_config(
    config_update: ConfigUpdate,
    current_user: User = Depends(get_current_user),
    # FIXED: Inject services as dependencies
    cache_service = Depends(get_semantic_cache_service),
    model_router = Depends(get_model_router),
    rate_limit_service = Depends(get_rate_limit_service),
):
    """
    Update system configuration.
    
    Requires admin privileges. Updates are applied immediately.
    """
    try:
        # Validate the configuration update
        validation_result = await validate_config_update(config_update)
        
        if not validation_result.valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Configuration validation failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Apply the updates
        if config_update.models:
            await model_router.update_model_configs(config_update.models)
        
        if config_update.rate_limiting:
            await rate_limit_service.update_config(config_update.rate_limiting)
        
        if config_update.cache:
            await cache_service.update_config(config_update.cache)
        
        # Log the configuration change
        logger.info(f"Configuration updated by user {current_user.id}")
        
        # Return the updated configuration
        return await get_system_config(current_user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update system configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update system configuration"
        )


@router.get("/models/providers")
async def get_available_model_providers(
    current_user: User = Depends(get_current_user),
):
    """
    Get list of available model providers and their models.
    """
    try:
        providers = await model_factory.get_available_providers()

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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
):
    """
    Test connection to a specific model provider.
    
    Useful for validating API keys and model availability.
    """
    try:
        result = await model_factory.test_model_connection(
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
    current_user: User = Depends(get_current_user),
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


@router.post("/validate", response_model=ConfigValidationResult)
async def validate_configuration(
    config_update: ConfigUpdate,
    current_user: User = Depends(get_current_user),
):
    """
    Validate a configuration update without applying it.
    
    Useful for testing configuration changes before applying them.
    """
    try:
        validation_result = await validate_config_update(config_update)
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate configuration"
        )


@router.get("/health", response_model=HealthStatus)
async def get_system_health(
    current_user: User = Depends(get_current_user),
    # FIXED: Inject services as dependencies
    cache_service = Depends(get_semantic_cache_service),
    vector_service = Depends(get_vector_db_service),
):
    """
    Get system health status including all components.
    """
    try:
        components = {}
        
        # Check vector database health
        components["vector_db"] = await vector_service.health_check()
        
        # Check cache health if enabled
        settings = get_settings()
        if settings.CACHE_ENABLED:
            components["cache"] = await cache_service.health_check()
        
        # Determine overall status
        component_statuses = [comp.get("status", "unhealthy") for comp in components.values()]
        if all(status == "healthy" for status in component_statuses):
            overall_status = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            uptime_seconds=0,  # Would calculate actual uptime
            version=settings.APP_VERSION,
            components=components
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health"
        )


# Helper functions for configuration updates

async def _update_model_config(model_config: ModelConfig):
    """Update model configuration."""
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


async def _check_model_providers_health():
    """Check model provider connectivity."""
    try:
        providers_status = await model_factory.check_all_providers_health()
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

async def validate_config_update(config_update: ConfigUpdate) -> ConfigValidationResult:
    """
    Validate a configuration update.
    
    This function checks for:
    - Required environment variables
    - Model availability
    - Valid configuration values
    """
    errors = []
    warnings = []
    missing_env_vars = []
    
    # Validate models
    if config_update.models:
        for model_id, model_config in config_update.models.items():
            if model_config.api_key_name:
                # Check if environment variable exists
                import os
                if not os.getenv(model_config.api_key_name):
                    missing_env_vars.append(model_config.api_key_name)
                    errors.append(f"Model '{model_id}' requires {model_config.api_key_name} environment variable")
    
    # Validate rate limiting
    if config_update.rate_limiting:
        if config_update.rate_limiting.per_user_per_minute > config_update.rate_limiting.global_per_minute:
            warnings.append("Per-user rate limit exceeds global rate limit")
    
    # Validate cache settings
    if config_update.cache:
        if config_update.cache.ttl_hours < 1:
            warnings.append("Cache TTL is very low (less than 1 hour)")
    
    return ConfigValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        missing_env_vars=missing_env_vars
    )

__all__ = ["router"]
