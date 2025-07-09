"""
Logging utilities for the chatbot platform.
Provides structured logging with correlation IDs and performance tracking.
"""

import logging
import logging.config
import sys
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Optional
from contextvars import ContextVar

from app.core.config import settings

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'structured': {
            'format': '%(asctime)s [%(levelname)s] %(name)s [%(correlation_id)s]: %(message)s'
        },
        'json': {
            '()': 'app.utils.logging.JSONFormatter',
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'structured',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'error': {
            'level': 'ERROR',
            'formatter': 'structured',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'json',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'error', 'file'],
            'level': 'DEBUG' if settings.DEBUG else 'INFO',
            'propagate': False
        },
        'app': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG' if settings.DEBUG else 'INFO',
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
        'fastapi': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        # Add correlation ID to record
        record.correlation_id = correlation_id.get() or 'none'
        
        # Create log data
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': record.correlation_id,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class CorrelationFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get() or 'none'
        return True


def setup_logging(level: Optional[str] = None):
    """Setup logging configuration."""
    if level:
        LOGGING_CONFIG['loggers']['']['level'] = level
        LOGGING_CONFIG['loggers']['app']['level'] = level
    
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Add correlation filter to all handlers
    correlation_filter = CorrelationFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(correlation_filter)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def set_correlation_id(cid: str):
    """Set the correlation ID for the current context."""
    correlation_id.set(cid)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


@contextmanager
def correlation_context(cid: Optional[str] = None):
    """Context manager for correlation ID."""
    if cid is None:
        cid = generate_correlation_id()
    
    token = correlation_id.set(cid)
    try:
        yield cid
    finally:
        correlation_id.reset(token)


def log_performance(logger: logging.Logger, operation: str):
    """Decorator to log performance metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation} completed successfully",
                    extra={
                        'operation': operation,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'success'
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation} failed: {str(e)}",
                    extra={
                        'operation': operation,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation} completed successfully",
                    extra={
                        'operation': operation,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'success'
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation} failed: {str(e)}",
                    extra={
                        'operation': operation,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    
    return decorator


def log_api_call(logger: logging.Logger, endpoint: str, method: str):
    """Decorator to log API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            cid = get_correlation_id() or generate_correlation_id()
            set_correlation_id(cid)
            
            logger.info(
                f"API call started: {method} {endpoint}",
                extra={
                    'endpoint': endpoint,
                    'method': method,
                    'correlation_id': cid,
                    'event': 'api_call_start'
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"API call completed: {method} {endpoint}",
                    extra={
                        'endpoint': endpoint,
                        'method': method,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'success',
                        'event': 'api_call_complete'
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"API call failed: {method} {endpoint} - {str(e)}",
                    extra={
                        'endpoint': endpoint,
                        'method': method,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'error',
                        'error': str(e),
                        'event': 'api_call_error'
                    }
                )
                
                raise
        
        return wrapper
    
    return decorator


def log_database_operation(logger: logging.Logger, operation: str, table: str):
    """Decorator to log database operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            logger.debug(
                f"Database operation started: {operation} on {table}",
                extra={
                    'operation': operation,
                    'table': table,
                    'event': 'db_operation_start'
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.debug(
                    f"Database operation completed: {operation} on {table}",
                    extra={
                        'operation': operation,
                        'table': table,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'success',
                        'event': 'db_operation_complete'
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Database operation failed: {operation} on {table} - {str(e)}",
                    extra={
                        'operation': operation,
                        'table': table,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'error',
                        'error': str(e),
                        'event': 'db_operation_error'
                    }
                )
                
                raise
        
        return wrapper
    
    return decorator


def log_llm_call(logger: logging.Logger, provider: str, model: str):
    """Decorator to log LLM API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            logger.info(
                f"LLM call started: {provider}/{model}",
                extra={
                    'provider': provider,
                    'model': model,
                    'event': 'llm_call_start'
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract token usage if available
                token_usage = getattr(result, 'token_usage', None)
                extra_data = {
                    'provider': provider,
                    'model': model,
                    'duration_ms': round(duration * 1000, 2),
                    'status': 'success',
                    'event': 'llm_call_complete'
                }
                
                if token_usage:
                    extra_data.update({
                        'tokens_used': token_usage.get('total_tokens', 0),
                        'input_tokens': token_usage.get('prompt_tokens', 0),
                        'output_tokens': token_usage.get('completion_tokens', 0)
                    })
                
                logger.info(
                    f"LLM call completed: {provider}/{model}",
                    extra=extra_data
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"LLM call failed: {provider}/{model} - {str(e)}",
                    extra={
                        'provider': provider,
                        'model': model,
                        'duration_ms': round(duration * 1000, 2),
                        'status': 'error',
                        'error': str(e),
                        'event': 'llm_call_error'
                    }
                )
                
                raise
        
        return wrapper
    
    return decorator


# Initialize logging
setup_logging()

# Export commonly used logger
logger = get_logger('app')
