"""
AuthToken model for the Turnkey AI Chatbot platform.

This module defines the AuthToken model that represents one-time
authentication tokens used for user verification via SMS or email.
"""

import uuid
import secrets
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Index, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped
from .base import Base

if TYPE_CHECKING:
    from .user import User


class AuthToken(Base):
    """
    AuthToken model representing one-time authentication tokens.

    These tokens are generated for user authentication via SMS or email
    and have a limited lifespan for security. Each token can only be
    used once and expires after a configured time period.

    Attributes:
        id: Unique identifier for the token
        user_id: Foreign key to the user this token belongs to
        token: The actual token string (hashed for security)
        session_id: Session ID this token is associated with
        expires_at: Timestamp when the token expires
        used: Whether the token has been used
        used_at: Timestamp when token was used (if applicable)
        created_at: Timestamp when token was created
        delivery_method: How the token was delivered (sms/email)
        delivery_address: Where the token was sent
        attempts: Number of verification attempts made
        max_attempts: Maximum allowed verification attempts
        user: Reference to the associated user
    """

    __tablename__ = "auth_tokens"

    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the token",
    )

    # Foreign key to user
    user_id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the user this token belongs to",
    )

    # Token data
    token: Mapped[str] = Column(
        String(64),  # SHA-256 hash length
        nullable=False,
        unique=True,
        index=True,
        comment="The hashed token string for security",
    )

    session_id: Mapped[str] = Column(
        String(255),
        nullable=False,
        index=True,
        comment="Session ID this token is associated with",
    )

    # Timestamps
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Timestamp when token was created",
    )

    expires_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Timestamp when the token expires",
    )

    used_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when token was used (if applicable)",
    )

    # Token status
    used: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the token has been used",
    )

    # Delivery information
    delivery_method: Mapped[str] = Column(
        String(10),  # 'sms' or 'email'
        nullable=False,
        comment="How the token was delivered (sms/email)",
    )

    delivery_address: Mapped[str] = Column(
        String(255),
        nullable=False,
        comment="Where the token was sent (phone number or email)",
    )

    # Security measures
    attempts: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of verification attempts made",
    )

    max_attempts: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=3,
        comment="Maximum allowed verification attempts",
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User", back_populates="auth_tokens", lazy="select"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_auth_token_user_id", "user_id"),
        Index("idx_auth_token_token", "token"),
        Index("idx_auth_token_session_id", "session_id"),
        Index("idx_auth_token_expires_at", "expires_at"),
        Index("idx_auth_token_used", "used"),
        Index("idx_auth_token_delivery_method", "delivery_method"),
        Index("idx_auth_token_user_session", "user_id", "session_id"),
        Index("idx_auth_token_expires_used", "expires_at", "used"),
    )

    def __repr__(self) -> str:
        """String representation of the auth token."""
        return (
            f"<AuthToken(id={self.id}, user_id={self.user_id}, "
            f"delivery_method={self.delivery_method}, "
            f"expires_at={self.expires_at}, used={self.used})>"
        )

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.utcnow() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        """
        Check if the token is valid for use.

        Returns:
            True if token is not used, not expired, and under attempt limit
        """
        return (
            not self.used and not self.is_expired and self.attempts < self.max_attempts
        )

    @property
    def seconds_until_expiry(self) -> int:
        """
        Get seconds remaining until expiry.

        Returns:
            Seconds until expiry, 0 if already expired
        """
        if self.is_expired:
            return 0

        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))

    @property
    def minutes_until_expiry(self) -> int:
        """
        Get minutes remaining until expiry.

        Returns:
            Minutes until expiry, 0 if already expired
        """
        return self.seconds_until_expiry // 60

    @property
    def attempts_remaining(self) -> int:
        """Get number of verification attempts remaining."""
        return max(0, self.max_attempts - self.attempts)

    @property
    def is_sms_delivery(self) -> bool:
        """Check if token was delivered via SMS."""
        return self.delivery_method == "sms"

    @property
    def is_email_delivery(self) -> bool:
        """Check if token was delivered via email."""
        return self.delivery_method == "email"

    def mark_used(self) -> None:
        """Mark the token as used."""
        self.used = True
        self.used_at = datetime.utcnow()

    def increment_attempts(self) -> bool:
        """
        Increment the verification attempts counter.

        Returns:
            True if increment was successful, False if max attempts exceeded
        """
        if self.attempts >= self.max_attempts:
            return False

        self.attempts += 1
        return True

    def can_attempt_verification(self) -> bool:
        """
        Check if verification can be attempted.

        Returns:
            True if token is valid and attempts are available
        """
        return self.is_valid and self.attempts < self.max_attempts

    @classmethod
    def generate_token_string(cls, length: int = 6) -> str:
        """
        Generate a random token string.

        Args:
            length: Length of the token (default 6 digits)

        Returns:
            Random numeric token string
        """
        # Generate a cryptographically secure random token
        # For user-facing tokens, we use digits for ease of entry
        import string
        return "".join(secrets.choice(string.digits) for _ in range(length))

    @classmethod
    def hash_token(cls, token: str) -> str:
        """
        Hash a token for secure storage.

        Args:
            token: Plain text token to hash

        Returns:
            SHA-256 hash of the token
        """
        import hashlib

        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def create_token(
        cls,
        user_id: uuid.UUID,
        session_id: str,
        delivery_method: str,
        delivery_address: str,
        expiry_minutes: int = 5,
    ) -> tuple["AuthToken", str]:
        """
        Create a new authentication token.

        Args:
            user_id: ID of the user this token is for
            session_id: Session ID to associate with
            delivery_method: How to deliver the token ('sms' or 'email')
            delivery_address: Where to send the token
            expiry_minutes: Token expiry time in minutes

        Returns:
            Tuple of (AuthToken instance, plain text token)
        """
        # Generate plain text token
        plain_token = cls.generate_token_string()

        # Create expiry time
        expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)

        # Create token instance
        auth_token = cls(
            user_id=user_id,
            token=cls.hash_token(plain_token),
            session_id=session_id,
            expires_at=expires_at,
            delivery_method=delivery_method,
            delivery_address=delivery_address,
        )

        return auth_token, plain_token

    def verify_token(self, provided_token: str) -> bool:
        """
        Verify a provided token against this stored token.

        Args:
            provided_token: Token provided by user

        Returns:
            True if token matches and is valid
        """
        # Check if token is valid for verification
        if not self.can_attempt_verification():
            return False

        # Increment attempts
        self.increment_attempts()

        # Hash the provided token and compare
        hashed_provided = self.hash_token(provided_token)

        if hashed_provided == self.token:
            self.mark_used()
            return True

        return False

    def extend_expiry(self, additional_minutes: int) -> None:
        """
        Extend the token expiry time.

        Args:
            additional_minutes: Minutes to add to expiry time
        """
        if not self.used:
            self.expires_at += timedelta(minutes=additional_minutes)

    def invalidate(self) -> None:
        """Invalidate the token by marking it as used."""
        self.used = True
        self.used_at = datetime.utcnow()

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """
        Convert token to dictionary representation.

        Args:
            include_sensitive: Whether to include sensitive information

        Returns:
            Dictionary representation of the token
        """
        data = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "used": self.used,
            "delivery_method": self.delivery_method,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "is_expired": self.is_expired,
            "is_valid": self.is_valid,
            "seconds_until_expiry": self.seconds_until_expiry,
        }

        if include_sensitive:
            data.update(
                {
                    "delivery_address": self.delivery_address,
                    "used_at": self.used_at.isoformat() if self.used_at else None,
                }
            )

        return data

