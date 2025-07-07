"""
Security utilities for the Turnkey AI Chatbot platform.
Handles authentication, authorization, password hashing, and token management.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request
import hashlib
import hmac
import re
from email_validator import validate_email, EmailNotValidError

from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"

# Security bearer for JWT
security_bearer = HTTPBearer(auto_error=False)


class SecurityManager:
    """Manages authentication and authorization operations."""
    
    def __init__(self):
        """Initialize security manager."""
        self.pwd_context = pwd_context
        self.secret_key = settings.SECRET_KEY
        self.algorithm = ALGORITHM
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Previously hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def generate_otp(self, length: int = 6) -> str:
        """
        Generate a numeric OTP (One-Time Password).
        
        Args:
            length: Length of the OTP (default: 6)
            
        Returns:
            Numeric OTP string
        """
        return ''.join([str(secrets.randbelow(10)) for _ in range(length)])
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.
        
        Args:
            length: Length of the token in bytes (default: 32)
            
        Returns:
            URL-safe base64 encoded token
        """
        return secrets.token_urlsafe(length)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Data to encode in the token
            expires_delta: Token expiration time delta
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"JWT encoding error: {e}")
            raise SecurityError("Failed to create access token")
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT refresh token.
        
        Args:
            data: Data to encode in the token
            expires_delta: Token expiration time delta
            
        Returns:
            Encoded JWT refresh token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({"exp": expire, "type": "refresh"})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"JWT encoding error: {e}")
            raise SecurityError("Failed to create refresh token")
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            token_type: Expected token type ("access" or "refresh")
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                logger.warning(f"Invalid token type. Expected: {token_type}, Got: {payload.get('type')}")
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                logger.warning("Token has expired")
                return None
            
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def hash_otp(self, otp: str, user_identifier: str) -> str:
        """
        Hash an OTP with user identifier for secure storage.
        
        Args:
            otp: One-time password
            user_identifier: User's email or phone number
            
        Returns:
            Hashed OTP string
        """
        combined = f"{otp}:{user_identifier}:{self.secret_key}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def verify_otp_hash(self, otp: str, user_identifier: str, hashed_otp: str) -> bool:
        """
        Verify an OTP against its hash.
        
        Args:
            otp: One-time password to verify
            user_identifier: User's email or phone number
            hashed_otp: Previously hashed OTP
            
        Returns:
            True if OTP matches, False otherwise
        """
        computed_hash = self.hash_otp(otp, user_identifier)
        return hmac.compare_digest(computed_hash, hashed_otp)


class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """
        Validate phone number format (basic validation).
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Remove all non-digit characters
        cleaned = re.sub(r'\D', '', phone)
        
        # Check if it's between 10 and 15 digits (international format)
        return 10 <= len(cleaned) <= 15
    
    @staticmethod
    def sanitize_input(input_string: str, max_length: int = 1000) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            input_string: Input to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized input string
        """
        if not isinstance(input_string, str):
            return ""
        
        # Truncate to max length
        sanitized = input_string[:max_length]
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Strip leading/trailing whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        
        return len(issues) == 0, issues


class RateLimitValidator:
    """Rate limiting validation utilities."""
    
    @staticmethod
    def get_client_ip(request: Request) -> str:
        """
        Extract client IP address from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in case of multiple proxies
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    @staticmethod
    def generate_rate_limit_key(identifier: str, endpoint: str) -> str:
        """
        Generate a rate limit key for Redis.
        
        Args:
            identifier: User identifier (IP, user_id, etc.)
            endpoint: API endpoint identifier
            
        Returns:
            Rate limit key string
        """
        return f"rate_limit:{endpoint}:{identifier}"


class CSRFProtection:
    """CSRF protection utilities."""
    
    def __init__(self):
        """Initialize CSRF protection."""
        self.secret_key = settings.CSRF_SECRET_KEY
    
    def generate_csrf_token(self, session_id: str) -> str:
        """
        Generate CSRF token for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CSRF token string
        """
        timestamp = str(int(datetime.utcnow().timestamp()))
        message = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def verify_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """
        Verify CSRF token.
        
        Args:
            token: CSRF token to verify
            session_id: Session identifier
            max_age: Maximum token age in seconds
            
        Returns:
            True if valid, False otherwise
        """
        try:
            timestamp_str, signature = token.split(":", 1)
            timestamp = int(timestamp_str)
            
            # Check token age
            current_time = int(datetime.utcnow().timestamp())
            if current_time - timestamp > max_age:
                return False
            
            # Verify signature
            message = f"{session_id}:{timestamp_str}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, IndexError):
            return False


# Global security manager instance
security_manager = SecurityManager()

# Global input validator instance
input_validator = InputValidator()

# Global rate limit validator instance
rate_limit_validator = RateLimitValidator()

# Global CSRF protection instance
csrf_protection = CSRFProtection()


# Authentication dependency functions
async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract and validate user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User data from token or None if invalid
    """
    if not credentials:
        return None
    
    try:
        payload = security_manager.verify_token(credentials.credentials, "access")
        if not payload:
            return None
        
        return payload
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return None


async def require_authentication(
    credentials: Optional[HTTPAuthorizationCredentials] = None
) -> Dict[str, Any]:
    """
    Require valid authentication token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User data from token
        
    Raises:
        HTTPException: If authentication fails
    """
    user = await get_current_user_from_token(credentials)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


class SecurityError(Exception):
    """Custom security-related error."""
    pass


class AuthenticationError(SecurityError):
    """Authentication-specific error."""
    pass


class AuthorizationError(SecurityError):
    """Authorization-specific error."""
    pass


def create_api_key(identifier: str, permissions: list = None) -> str:
    """
    Create an API key for external integrations.
    
    Args:
        identifier: Unique identifier for the API key
        permissions: List of permissions for the key
        
    Returns:
        Generated API key
    """
    if permissions is None:
        permissions = []
    
    # Create a payload with identifier and permissions
    payload = {
        "identifier": identifier,
        "permissions": permissions,
        "created_at": datetime.utcnow().isoformat(),
        "type": "api_key"
    }
    
    # Create a long-lived token (1 year)
    expires_delta = timedelta(days=365)
    api_key = security_manager.create_access_token(payload, expires_delta)
    
    return api_key


def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Verify an API key and extract its payload.
    
    Args:
        api_key: API key to verify
        
    Returns:
        API key payload or None if invalid
    """
    payload = security_manager.verify_token(api_key, "access")
    
    if not payload or payload.get("type") != "api_key":
        return None
    
    return payload


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash sensitive data with salt.
    
    Args:
        data: Data to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple of (hashed_data, salt_used)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    combined = f"{data}:{salt}:{settings.SECRET_KEY}"
    hashed = hashlib.sha256(combined.encode()).hexdigest()
    
    return hashed, salt


def verify_sensitive_data(data: str, hashed_data: str, salt: str) -> bool:
    """
    Verify sensitive data against its hash.
    
    Args:
        data: Original data
        hashed_data: Previously hashed data
        salt: Salt used in hashing
        
    Returns:
        True if data matches, False otherwise
    """
    computed_hash, _ = hash_sensitive_data(data, salt)
    return hmac.compare_digest(computed_hash, hashed_data)


class SessionManager:
    """Manages user sessions and session security."""
    
    def __init__(self):
        """Initialize session manager."""
        self.session_timeout = timedelta(minutes=settings.AUTH_SESSION_TIMEOUT_MINUTES)
    
    def create_session_id(self) -> str:
        """Create a new session ID."""
        return secrets.token_urlsafe(32)
    
    def create_session_data(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """
        Create session data dictionary.
        
        Args:
            user_id: User identifier
            **kwargs: Additional session data
            
        Returns:
            Session data dictionary
        """
        now = datetime.utcnow()
        
        session_data = {
            "user_id": user_id,
            "created_at": now.isoformat(),
            "last_accessed": now.isoformat(),
            "expires_at": (now + self.session_timeout).isoformat(),
            **kwargs
        }
        
        return session_data
    
    def is_session_valid(self, session_data: Dict[str, Any]) -> bool:
        """
        Check if session is still valid.
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            True if session is valid, False otherwise
        """
        try:
            expires_at = datetime.fromisoformat(session_data.get("expires_at", ""))
            return datetime.utcnow() < expires_at
        except (ValueError, TypeError):
            return False
    
    def refresh_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refresh session expiration time.
        
        Args:
            session_data: Current session data
            
        Returns:
            Updated session data
        """
        now = datetime.utcnow()
        session_data["last_accessed"] = now.isoformat()
        session_data["expires_at"] = (now + self.session_timeout).isoformat()
        
        return session_data


# Global session manager instance
session_manager = SessionManager()


class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """
        Get recommended security headers.
        
        Returns:
            Dictionary of security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),
        }


class DataEncryption:
    """Data encryption utilities for sensitive information."""
    
    def __init__(self):
        """Initialize encryption with secret key."""
        self.key = settings.SECRET_KEY.encode()[:32]  # Use first 32 bytes
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data (simplified implementation).
        Note: For production, use proper encryption libraries like cryptography.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as hex string
        """
        # This is a simplified implementation for demonstration
        # In production, use proper encryption like AES with cryptography library
        import base64
        
        # Simple XOR encryption (NOT suitable for production)
        encrypted = bytearray()
        key_bytes = self.key
        
        for i, byte in enumerate(data.encode()):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return base64.b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data as hex string
            
        Returns:
            Decrypted data
        """
        import base64
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = bytearray()
            key_bytes = self.key
            
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
            
            return decrypted.decode()
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise SecurityError("Failed to decrypt data")


# Global encryption instance
data_encryption = DataEncryption()


def secure_compare(a: str, b: str) -> bool:
    """
    Timing-safe string comparison.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal, False otherwise
    """
    return hmac.compare_digest(a, b)


def generate_nonce(length: int = 16) -> str:
    """
    Generate a cryptographic nonce.
    
    Args:
        length: Length of nonce in bytes
        
    Returns:
        Base64-encoded nonce
    """
    return secrets.token_urlsafe(length)
