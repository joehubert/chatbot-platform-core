"""
Health check API endpoint for the chatbot platform.
Provides comprehensive health monitoring including dependencies.

TODO:
- list the routes in the header
"""

from typing import Dict, List, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_async_db
from app.core.redis import get_redis_client
from app.services.vector_db import VectorDBService
from app.services.model_factory import ModelFactory
from app.utils.logging import get_logger, log_api_call
from app.utils.exceptions import DatabaseError, CacheError, VectorDBError, ModelError
import asyncio
import time
from datetime import datetime, timezone

router = APIRouter()
logger = get_logger(__name__)


@router.get("/", tags=["health"])
@log_api_call(logger, "/", "GET")
async def health_check(db: Session = Depends(get_async_db)):
    """
    Comprehensive health check endpoint.

    Returns:
        dict: Health status of the system and all dependencies
    """
    start_time = time.time()

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "uptime": time.time() - start_time,
        "checks": {},
    }

    overall_healthy = True

    # Check database
    try:
        db_status = await check_database_health(db)
        health_status["checks"]["database"] = db_status
        if db_status["status"] != "healthy":
            overall_healthy = False
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        overall_healthy = False

    # Check Redis
    try:
        redis_status = await check_redis_health()
        health_status["checks"]["redis"] = redis_status
        if redis_status["status"] != "healthy":
            overall_healthy = False
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        overall_healthy = False

    # Check Vector Database
    try:
        vector_db_status = await check_vector_db_health()
        health_status["checks"]["vector_db"] = vector_db_status
        if vector_db_status["status"] != "healthy":
            overall_healthy = False
    except Exception as e:
        logger.error(f"Vector DB health check failed: {str(e)}")
        health_status["checks"]["vector_db"] = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        overall_healthy = False

    # Check Model Providers
    try:
        model_status = await check_model_providers_health()
        health_status["checks"]["model_providers"] = model_status
        # Model providers are not critical for basic health
        # overall_healthy remains true even if some providers are down
    except Exception as e:
        logger.error(f"Model providers health check failed: {str(e)}")
        health_status["checks"]["model_providers"] = {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Set overall status
    if not overall_healthy:
        health_status["status"] = "unhealthy"
    elif any(
        check.get("status") == "degraded" for check in health_status["checks"].values()
    ):
        health_status["status"] = "degraded"

    health_status["duration_ms"] = round((time.time() - start_time) * 1000, 2)

    # Return appropriate HTTP status code
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


@router.get("/health/ready", tags=["health"])
@log_api_call(logger, "/health/ready", "GET")
async def readiness_check(db: Session = Depends(get_async_db)):
    """
    Readiness check endpoint for Kubernetes.

    Returns:
        dict: Simple ready/not ready status
    """
    try:
        # Quick database check
        db.execute("SELECT 1")

        # Quick Redis check
        redis_client = await get_redis_client()
        await redis_client.ping()

        return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.get("/health/live", tags=["health"])
@log_api_call(logger, "/health/live", "GET")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes.

    Returns:
        dict: Simple alive status
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


async def check_database_health(db: Session) -> Dict[str, Any]:
    """Check database connectivity and performance."""
    start_time = time.time()

    try:
        # Test basic connectivity
        result = db.execute("SELECT 1 as test")
        row = result.fetchone()

        if not row or row[0] != 1:
            raise DatabaseError("Database query returned unexpected result")

        # Test a more complex query to check performance
        db.execute("SELECT COUNT(*) FROM conversations LIMIT 1")

        duration_ms = round((time.time() - start_time) * 1000, 2)

        status = "healthy"
        if duration_ms > 1000:  # > 1 second is concerning
            status = "degraded"

        return {
            "status": status,
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity and performance."""
    start_time = time.time()

    try:
        redis_client = await get_redis_client()

        # Test basic connectivity
        pong = await redis_client.ping()
        if not pong:
            raise CacheError("Redis ping failed")

        # Test read/write operations
        test_key = "health_check_test"
        test_value = "test_value"

        await redis_client.set(test_key, test_value, ex=60)  # Expire in 60 seconds
        retrieved_value = await redis_client.get(test_key)

        if retrieved_value != test_value:
            raise CacheError("Redis read/write test failed")

        # Clean up
        await redis_client.delete(test_key)

        duration_ms = round((time.time() - start_time) * 1000, 2)

        status = "healthy"
        if duration_ms > 500:  # > 500ms is concerning for Redis
            status = "degraded"

        return {
            "status": status,
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Redis health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def check_vector_db_health() -> Dict[str, Any]:
    """Check vector database connectivity and performance."""
    start_time = time.time()

    try:
        vector_db = VectorDBService()

        # Test basic connectivity - this will vary based on the vector DB implementation
        # For now, we'll just check if the service initializes properly
        await vector_db.health_check()

        duration_ms = round((time.time() - start_time) * 1000, 2)

        status = "healthy"
        if duration_ms > 2000:  # > 2 seconds is concerning
            status = "degraded"

        return {
            "status": status,
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Vector DB health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def check_model_providers_health() -> Dict[str, Any]:
    """Check model provider connectivity and availability."""
    start_time = time.time()

    try:
        model_factory = ModelFactory()
        provider_statuses = {}

        # Get configured providers
        providers = ["openai", "anthropic", "huggingface"]  # Add others as configured

        # Check each provider asynchronously
        tasks = []
        for provider in providers:
            task = check_model_provider_health(model_factory, provider)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, provider in enumerate(providers):
            if isinstance(results[i], Exception):
                provider_statuses[provider] = {
                    "status": "unhealthy",
                    "error": str(results[i]),
                }
            else:
                provider_statuses[provider] = results[i]

        # Determine overall status
        healthy_count = sum(
            1 for status in provider_statuses.values() if status["status"] == "healthy"
        )
        total_count = len(provider_statuses)

        if healthy_count == 0:
            overall_status = "unhealthy"
        elif healthy_count < total_count:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        duration_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": overall_status,
            "providers": provider_statuses,
            "healthy_providers": healthy_count,
            "total_providers": total_count,
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Model providers health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def check_model_provider_health(
    model_factory: ModelFactory, provider: str
) -> Dict[str, Any]:
    """Check health of a specific model provider."""
    start_time = time.time()

    try:
        # Try to get a client for the provider
        client = await model_factory.get_client(provider)

        # Perform a minimal health check (implementation depends on provider)
        await client.health_check()

        duration_ms = round((time.time() - start_time) * 1000, 2)

        return {"status": "healthy", "response_time_ms": duration_ms}

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {"status": "unhealthy", "error": str(e), "response_time_ms": duration_ms}


@router.get("/health/dependencies", tags=["health"])
@log_api_call(logger, "/health/dependencies", "GET")
async def dependencies_health():
    """
    Detailed health check for all dependencies.

    Returns:
        dict: Detailed status of each dependency
    """
    dependencies = {}

    # Check all dependencies in parallel
    tasks = {
        "database": check_database_health_standalone(),
        "redis": check_redis_health(),
        "vector_db": check_vector_db_health(),
        "model_providers": check_model_providers_health(),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for i, (dep_name, _) in enumerate(tasks.items()):
        if isinstance(results[i], Exception):
            dependencies[dep_name] = {
                "status": "unhealthy",
                "error": str(results[i]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            dependencies[dep_name] = results[i]

    return {
        "dependencies": dependencies,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def check_database_health_standalone() -> Dict[str, Any]:
    """Standalone database health check without dependency injection."""
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        return await check_database_health(db)
    finally:
        db.close()
