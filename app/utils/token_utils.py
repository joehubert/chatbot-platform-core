"""
Token Utilities for Authentication System

Provides secure token generation, hashing, and validation utilities
for the authentication system.
"""

import hashlib
import secrets
import string
import base64
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TokenUtils:
    """
    Utility class for token operations in the authentication system.
    
    Handles:
    - Secure token generation
    - Token hashing for safe storage
    - Token validation
    """
    
    @staticmethod
    def generate_secure_token(length: int = 6) -> str:
        """
        Generate a cryptographically secure random token.
        
        Args:
            length: Length of the token (default: 6)
            
        Returns:
            Secure random token string
        """
        # Use digits for OTP tokens (easier to read/type)
        characters = string.digits
        
        # Generate secure random token
        token = ''.join(secrets.choice(characters) for _ in range(length))
        
        logger.debug(f"Generated secure token of length {length}")
        return token
    
    @staticmethod
    def generate_alphanumeric_token(length: int = 12) -> str:
        """
        Generate a secure alphanumeric token.
        
        Args:
            length: Length of the token (default: 12)
            
        Returns:
            Secure alphanumeric token string
        """
        # Use letters and digits (excluding confusing characters)
        characters = string.ascii_uppercase + string.digits
        characters = characters.replace('0', '').replace('O', '').replace('1', '').replace('I', '')
        
        token = ''.join(secrets.choice(characters) for _ in range(length))
        
        logger.debug(f"Generated alphanumeric token of length {length}")
        return token
    
    @staticmethod
    def hash_token(token: str) -> str:
        """
        Hash a token for secure storage in database.
        
        Args:
            token: Plain text token to hash
            
        Returns:
            SHA-256 hash of the token
        """
        # Create SHA-256 hash
        token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
        
        logger.debug("Token hashed for storage")
        return token_hash
    
    @staticmethod
    def verify_token_format(token: str, expected_length: int = 6) -> bool:
        """
        Verify that a token has the expected format.
        
        Args:
            token: Token to verify
            expected_length: Expected token length
            
        Returns:
            True if token format is valid
        """
        if not token:
            return False
        
        if len(token) != expected_length:
            return False
        
        # Check if token contains only digits (for OTP)
        if not token.isdigit():
            return False
        
        return True
    
    @staticmethod
    def is_token_expired(created_at: datetime, expiry_minutes: int) -> bool:
        """
        Check if a token has expired.
        
        Args:
            created_at: When the token was created
            expiry_minutes: Token expiry time in minutes
            
        Returns:
            True if token has expired
        """
        expiry_time = created_at + timedelta(minutes=expiry_minutes)
        return datetime.utcnow() > expiry_time
    
    @staticmethod
    def generate_session_id() -> str:
        """
        Generate a unique session identifier.
        
        Returns:
            Base64-encoded random session ID
        """
        # Generate 32 random bytes
        random_bytes = secrets.token_bytes(32)
        
        # Encode as base64 and remove padding
        session_id = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
        
        logger.debug("Generated session ID")
        return session_id
    
    @staticmethod
    def generate_api_key(prefix: str = "cbp") -> str:
        """
        Generate an API key for external integrations.
        
        Args:
            prefix: Prefix for the API key (default: "cbp" for chatbot platform)
            
        Returns:
            API key string
        """
        # Generate 32 random bytes
        random_bytes = secrets.token_bytes(32)
        
        # Encode as hex
        key_part = random_bytes.hex()
        
        # Combine with prefix
        api_key = f"{prefix}_{key_part}"
        
        logger.debug(f"Generated API key with prefix {prefix}")
        return api_key
    
    @staticmethod
    def mask_token(token: str, visible_chars: int = 2) -> str:
        """
        Mask a token for logging purposes.
        
        Args:
            token: Token to mask
            visible_chars: Number of characters to show at start and end
            
        Returns:
            Masked token string
        """
        if not token:
            return "***"
        
        if len(token) <= visible_chars * 2:
            return "*" * len(token)
        
        start = token[:visible_chars]
        end = token[-visible_chars:]
        middle = "*" * (len(token) - visible_chars * 2)
        
        return f"{start}{middle}{end}"
    
    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """
        Compare two strings in constant time to prevent timing attacks.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            True if strings are equal
        """
        return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))
    
    @staticmethod
    def generate_backup_codes(count: int = 8, length: int = 8) -> list[str]:
        """
        Generate backup authentication codes.
        
        Args:
            count: Number of backup codes to generate
            length: Length of each backup code
            
        Returns:
            List of backup codes
        """
        codes = []
        
        for _ in range(count):
            code = TokenUtils.generate_alphanumeric_token(length)
            codes.append(code)
        
        logger.debug(f"Generated {count} backup codes")
        return codes
    
    @staticmethod
    def format_token_for_display(token: str) -> str:
        """
        Format a token for user-friendly display.
        
        Args:
            token: Token to format
            
        Returns:
            Formatted token string
        """
        if not token:
            return ""
        
        # For 6-digit tokens, format as XXX-XXX
        if len(token) == 6 and token.isdigit():
            return f"{token[:3]}-{token[3:]}"
        
        # For longer tokens, add spaces every 4 characters
        if len(token) > 4:
            formatted = ""
            for i in range(0, len(token), 4):
                if formatted:
                    formatted += " "
                formatted += token[i:i+4]
            return formatted
        
        return token
    
    @staticmethod
    def validate_token_strength(token: str) -> dict:
        """
        Validate the strength of a generated token.
        
        Args:
            token: Token to validate
            
        Returns:
            Dict with validation results
        """
        results = {
            "valid": True,
            "length": len(token),
            "has_digits": any(c.isdigit() for c in token),
            "has_letters": any(c.isalpha() for c in token),
            "has_uppercase": any(c.isupper() for c in token),
            "has_lowercase": any(c.islower() for c in token),
            "entropy_bits": 0,
            "warnings": []
        }
        
        # Calculate entropy
        if token:
            # Estimate character set size
            charset_size = 0
            if results["has_digits"]:
                charset_size += 10
            if results["has_uppercase"]:
                charset_size += 26
            if results["has_lowercase"]:
                charset_size += 26
            
            if charset_size > 0:
                import math
                results["entropy_bits"] = len(token) * math.log2(charset_size)
        
        # Check for common issues
        if len(token) < 6:
            results["warnings"].append("Token is shorter than recommended minimum (6 characters)")
            results["valid"] = False
        
        if not results["has_digits"] and not results["has_letters"]:
            results["warnings"].append("Token contains no alphanumeric characters")
            results["valid"] = False
        
        if results["entropy_bits"] < 20:
            results["warnings"].append("Token has low entropy (less than 20 bits)")
        
        return results


# Convenience functions for common operations
def generate_otp() -> str:
    """Generate a standard 6-digit OTP."""
    return TokenUtils.generate_secure_token(6)


def hash_otp(otp: str) -> str:
    """Hash an OTP for storage."""
    return TokenUtils.hash_token(otp)


def verify_otp_format(otp: str) -> bool:
    """Verify OTP format."""
    return TokenUtils.verify_token_format(otp, 6)


def generate_session() -> str:
    """Generate a session ID."""
    return TokenUtils.generate_session_id()


def mask_otp(otp: str) -> str:
    """Mask an OTP for logging."""
    return TokenUtils.mask_token(otp, 1)


# Token validation constants
MIN_TOKEN_LENGTH = 6
MAX_TOKEN_LENGTH = 64
OTP_LENGTH = 6
SESSION_ID_LENGTH = 43  # Base64 encoded 32 bytes without padding
