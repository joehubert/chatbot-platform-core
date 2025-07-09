"""
Custom exceptions for the chatbot platform.
Provides structured error handling and response formatting.
"""

from typing import Any, Dict, Optional
from enum import Enum


class ErrorCode(Enum):
    """Enumeration of error codes for the chatbot platform."""
    
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    
    # Authentication errors
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    OTP_REQUIRED = "OTP_REQUIRED"
    OTP_INVALID = "OTP_INVALID"
    OTP_EXPIRED = "OTP_EXPIRED"
    
    # LLM and processing errors
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_QUOTA_EXCEEDED = "LLM_QUOTA_EXCEEDED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    INAPPROPRIATE_CONTENT = "INAPPROPRIATE_CONTENT"
    
    # Knowledge base errors
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_PROCESSING_FAILED = "DOCUMENT_PROCESSING_FAILED"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    KNOWLEDGE_BASE_UNAVAILABLE = "KNOWLEDGE_BASE_UNAVAILABLE"
    
    # Vector database errors
    VECTOR_DB_UNAVAILABLE = "VECTOR_DB_UNAVAILABLE"
    VECTOR_DB_OPERATION_FAILED = "VECTOR_DB_OPERATION_FAILED"
    EMBEDDING_GENERATION_FAILED = "EMBEDDING_GENERATION_FAILED"
    
    # Cache errors
    CACHE_UNAVAILABLE = "CACHE_UNAVAILABLE"
    CACHE_OPERATION_FAILED = "CACHE_OPERATION_FAILED"
    
    # Database errors
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    DATABASE_OPERATION_FAILED = "DATABASE_OPERATION_FAILED"
    DUPLICATE_ENTRY = "DUPLICATE_ENTRY"
    
    # MCP errors
    MCP_SERVER_UNAVAILABLE = "MCP_SERVER_UNAVAILABLE"
    MCP_OPERATION_FAILED = "MCP_OPERATION_FAILED"
    MCP_AUTHENTICATION_FAILED = "MCP_AUTHENTICATION_FAILED"
    
    # Configuration errors
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_API_KEY = "INVALID_API_KEY"
    
    # Session errors
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    INVALID_SESSION = "INVALID_SESSION"


class ChatbotException(Exception):
    """Base exception class for chatbot platform errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "details": self.details
            }
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"
    
    def __repr__(self) -> str:
        return f"ChatbotException(message='{self.message}', error_code={self.error_code}, status_code={self.status_code})"


class ValidationError(ChatbotException):
    """Exception for validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            status_code=400
        )


class NotFoundError(ChatbotException):
    """Exception for resource not found errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.NOT_FOUND,
            details=details,
            status_code=404
        )


class UnauthorizedError(ChatbotException):
    """Exception for unauthorized access errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED,
            details=details,
            status_code=401
        )


class ForbiddenError(ChatbotException):
    """Exception for forbidden access errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.FORBIDDEN,
            details=details,
            status_code=403
        )


class RateLimitError(ChatbotException):
    """Exception for rate limiting errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMITED,
            details=details,
            status_code=429
        )


class AuthenticationError(ChatbotException):
    """Exception for authentication errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=401
        )


class LLMError(ChatbotException):
    """Exception for LLM-related errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        status_code = 503 if error_code == ErrorCode.LLM_UNAVAILABLE else 500
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code
        )


class KnowledgeBaseError(ChatbotException):
    """Exception for knowledge base errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        status_code = 400 if error_code in [ErrorCode.UNSUPPORTED_FILE_TYPE, ErrorCode.FILE_TOO_LARGE] else 500
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code
        )


class VectorDBError(ChatbotException):
    """Exception for vector database errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=503
        )


class CacheError(ChatbotException):
    """Exception for cache errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=503
        )


class DatabaseError(ChatbotException):
    """Exception for database errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        status_code = 409 if error_code == ErrorCode.DUPLICATE_ENTRY else 500
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code
        )


class MCPError(ChatbotException):
    """Exception for MCP-related errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=503
        )


class ConfigurationError(ChatbotException):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=500
        )


class SessionError(ChatbotException):
    """Exception for session-related errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        status_code = 401 if error_code in [ErrorCode.SESSION_EXPIRED, ErrorCode.INVALID_SESSION] else 404
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code
        )


# Convenience functions for creating specific errors
def validation_error(message: str, field: Optional[str] = None, value: Optional[Any] = None) -> ValidationError:
    """Create a validation error with optional field and value details."""
    details = {}
    if field:
        details['field'] = field
    if value is not None:
        details['value'] = value
    return ValidationError(message, details)


def not_found_error(resource_type: str, resource_id: Optional[str] = None) -> NotFoundError:
    """Create a not found error for a specific resource."""
    message = f"{resource_type} not found"
    if resource_id:
        message += f" with ID: {resource_id}"
    
    details = {'resource_type': resource_type}
    if resource_id:
        details['resource_id'] = resource_id
    
    return NotFoundError(message, details)


def unauthorized_error(reason: str = "Authentication required") -> UnauthorizedError:
    """Create an unauthorized error."""
    return UnauthorizedError(reason)


def forbidden_error(reason: str = "Access forbidden") -> ForbiddenError:
    """Create a forbidden error."""
    return ForbiddenError(reason)


def rate_limit_error(limit: int, window: str = "minute") -> RateLimitError:
    """Create a rate limit error."""
    message = f"Rate limit exceeded: {limit} requests per {window}"
    details = {'limit': limit, 'window': window}
    return RateLimitError(message, details)


def llm_unavailable_error(provider: str, model: str) -> LLMError:
    """Create an LLM unavailable error."""
    message = f"LLM service unavailable: {provider}/{model}"
    details = {'provider': provider, 'model': model}
    return LLMError(message, ErrorCode.LLM_UNAVAILABLE, details)


def document_processing_error(filename: str, error: str) -> KnowledgeBaseError:
    """Create a document processing error."""
    message = f"Failed to process document: {filename}"
    details = {'filename': filename, 'error': error}
    return KnowledgeBaseError(message, ErrorCode.DOCUMENT_PROCESSING_FAILED, details)


def vector_db_error(operation: str, error: str) -> VectorDBError:
    """Create a vector database error."""
    message = f"Vector database operation failed: {operation}"
    details = {'operation': operation, 'error': error}
    return VectorDBError(message, ErrorCode.VECTOR_DB_OPERATION_FAILED, details)


def cache_error(operation: str, error: str) -> CacheError:
    """Create a cache error."""
    message = f"Cache operation failed: {operation}"
    details = {'operation': operation, 'error': error}
    return CacheError(message, ErrorCode.CACHE_OPERATION_FAILED, details)


def database_error(operation: str, table: str, error: str) -> DatabaseError:
    """Create a database error."""
    message = f"Database operation failed: {operation} on {table}"
    details = {'operation': operation, 'table': table, 'error': error}
    return DatabaseError(message, ErrorCode.DATABASE_OPERATION_FAILED, details)


def mcp_error(server: str, operation: str, error: str) -> MCPError:
    """Create an MCP error."""
    message = f"MCP operation failed: {operation} on {server}"
    details = {'server': server, 'operation': operation, 'error': error}
    return MCPError(message, ErrorCode.MCP_OPERATION_FAILED, details)


def configuration_error(setting: str, value: Optional[str] = None) -> ConfigurationError:
    """Create a configuration error."""
    message = f"Invalid configuration: {setting}"
    details = {'setting': setting}
    if value:
        details['value'] = value
    return ConfigurationError(message, ErrorCode.INVALID_CONFIGURATION, details)


def session_expired_error(session_id: str) -> SessionError:
    """Create a session expired error."""
    message = "Session has expired"
    details = {'session_id': session_id}
    return SessionError(message, ErrorCode.SESSION_EXPIRED, details)


def invalid_token_error(token_type: str = "access") -> AuthenticationError:
    """Create an invalid token error."""
    message = f"Invalid {token_type} token"
    details = {'token_type': token_type}
    return AuthenticationError(message, ErrorCode.INVALID_TOKEN, details)


def otp_required_error(methods: list) -> AuthenticationError:
    """Create an OTP required error."""
    message = "One-time password required"
    details = {'available_methods': methods}
    return AuthenticationError(message, ErrorCode.OTP_REQUIRED, details)


def otp_invalid_error() -> AuthenticationError:
    """Create an OTP invalid error."""
    return AuthenticationError("Invalid one-time password", ErrorCode.OTP_INVALID)


def otp_expired_error() -> AuthenticationError:
    """Create an OTP expired error."""
    return AuthenticationError("One-time password has expired", ErrorCode.OTP_EXPIRED)
