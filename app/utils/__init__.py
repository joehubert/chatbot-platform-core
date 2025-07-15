"""
Utility modules for the chatbot platform.
"""

from .logging import get_logger, setup_logging, correlation_context, log_performance
from .exceptions import (
    ChatbotException, ValidationError, NotFoundError, UnauthorizedError,
    ForbiddenError, RateLimitError, AuthenticationError, ModelError,
    KnowledgeBaseError, VectorDBError, CacheError, DatabaseError,
    MCPError, ConfigurationError, SessionError, ErrorCode
)
from .monitoring import (
    metrics_collector, system_monitor, alert_manager, health_checker,
    PerformanceMonitor, monitor_performance, get_metrics_summary,
    record_api_metrics, record_model_metrics, record_database_metrics
)

__all__ = [
    # Logging
    'get_logger',
    'setup_logging', 
    'correlation_context',
    'log_performance',
    
    # Exceptions
    'ChatbotException',
    'ValidationError',
    'NotFoundError', 
    'UnauthorizedError',
    'ForbiddenError',
    'RateLimitError',
    'AuthenticationError',
    'ModelError',
    'KnowledgeBaseError',
    'VectorDBError',
    'CacheError',
    'DatabaseError',
    'MCPError',
    'ConfigurationError',
    'SessionError',
    'ErrorCode',
    
    # Monitoring
    'metrics_collector',
    'system_monitor',
    'alert_manager',
    'health_checker',
    'PerformanceMonitor',
    'monitor_performance',
    'get_metrics_summary',
    'record_api_metrics',
    'record_model_metrics',
    'record_database_metrics'
]
