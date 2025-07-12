"""
Main FastAPI application for the Turnkey AI Chatbot platform.
Handles application initialization, middleware setup, and routing.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time

from app.core.config import settings
from app.core.database import init_db, db_manager
from app.core.redis import init_redis, redis_manager
from app.core.security import SecurityHeaders
from app.utils.logging import setup_logging

setup_logging()
from app.api import api_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if settings.LOG_FORMAT == 'text' 
           else '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"in {process_time:.3f}s"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request failed: {str(e)} in {process_time:.3f}s")
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        security_headers = SecurityHeaders.get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Turnkey AI Chatbot platform...")
    
    try:
        # Initialize database and Redis
        logger.info("Initializing database and Redis connections...")
        await init_db()
        logger.info("Database and Redis connections established")
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Turnkey AI Chatbot platform...")
        
        try:
            # Close database connections
            await db_manager.close()
            logger.info("Database connections closed")
            
            # Close Redis connections
            await redis_manager.close()
            logger.info("Redis connections closed")
            
            logger.info("Application shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app with lifespan
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Enterprise-grade chatbot platform with RAG capabilities and intelligent routing",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Add routes
    setup_routes(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Setup application middleware.
    
    Args:
        app: FastAPI application instance
    """
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request logging middleware
    if settings.DEBUG or settings.LOG_LEVEL == "DEBUG":
        app.add_middleware(RequestLoggingMiddleware)
    
    # CORS middleware
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Trusted host middleware (for production)
    if settings.is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual allowed hosts in production
        )
    
    # GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()} - {request.method} {request.url}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": True,
                "message": "Validation error",
                "details": exc.errors(),
                "status_code": 422,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {str(exc)} - {request.method} {request.url}", exc_info=True)
        
        if settings.DEBUG:
            error_detail = str(exc)
        else:
            error_detail = "Internal server error"
        
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": error_detail,
                "status_code": 500,
                "path": str(request.url.path)
            }
        )


def setup_routes(app: FastAPI) -> None:
    """
    Setup application routes.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Turnkey AI Chatbot Platform",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "healthy"
        }
    
    @app.get("/health")
    async def health_check():
        """
        Health check endpoint.
        
        Returns:
            Health status of the application and its dependencies
        """
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "checks": {}
        }
        
        # Check database health
        try:
            db_healthy = await db_manager.test_connection()
            health_status["checks"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "details": "Connection successful" if db_healthy else "Connection failed"
            }
        except Exception as e:
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "details": str(e)
            }
        
        # Check Redis health
        try:
            redis_healthy = await redis_manager.test_connection()
            health_status["checks"]["redis"] = {
                "status": "healthy" if redis_healthy else "unhealthy",
                "details": "Connection successful" if redis_healthy else "Connection failed"
            }
        except Exception as e:
            health_status["checks"]["redis"] = {
                "status": "unhealthy",
                "details": str(e)
            }
        
        # Determine overall health
        all_healthy = all(
            check["status"] == "healthy" 
            for check in health_status["checks"].values()
        )
        
        if not all_healthy:
            health_status["status"] = "unhealthy"
        
        # Return appropriate status code
        status_code = 200 if all_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
    
    @app.get("/metrics")
    async def metrics():
        """
        Basic metrics endpoint.
        
        Returns:
            Application metrics
        """
        if not settings.ENABLE_METRICS:
            return JSONResponse(
                status_code=404,
                content={"error": "Metrics not enabled"}
            )
        
        # Basic metrics (can be extended with Prometheus metrics)
        metrics_data = {
            "timestamp": time.time(),
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime": "available in future version",
            "memory_usage": "available in future version",
            "request_count": "available in future version",
        }
        
        return metrics_data
    
    # API routes will be added in future implementations
    app.include_router(api_router, prefix=settings.API_V1_STR)


# Create the application instance
app = create_application()


def main():
    """Main entry point for running the application."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
