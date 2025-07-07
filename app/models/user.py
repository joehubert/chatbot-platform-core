"""
User model for the Turnkey AI Chatbot platform.

This module defines the User model that represents users who interact
with the chatbot and may require authentication for certain features.
"""

import uuid
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Index, Text, or_
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, Session
from .base import Base

if TYPE_CHECKING:
    from .auth_token import AuthToken


class User(Base):
    """
    User model representing users who interact with the chatbot.

    Users are identified by their mobile number or email address
    and are created when authentication is required. This model
    tracks user preferences, authentication history, and interaction
    patterns for analytics and personalization.

    Attributes:
        id: Unique identifier for the user
        mobile_number: User's mobile phone number (for SMS auth)
        email: User's email address (for email auth)
        created_at: Timestamp when user was first created
        last_authenticated: Timestamp of most recent successful authentication
        authentication_count: Total number of successful authentications
        is_active: Whether the user account is active
        preferred_contact_method: Preferred method for authentication (sms/email)
        timezone: User's timezone for scheduling and timestamps
        language_preference: Preferred language code (e.g., 'en', 'es')
        metadata: Additional user metadata stored as JSON
        notes: Administrative notes about the user
        last_seen: Timestamp when user was last active
        auth_tokens: Related authentication tokens
    """

    __tablename__ = "users"

    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the user",
    )

    # Contact information (at least one must be provided)
    mobile_number: Mapped[Optional[str]] = Column(
        String(20),
        nullable=True,
        unique=True,
        index=True,
        comment="User's mobile phone number for SMS authentication",
    )

    email: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        comment="User's email address for email authentication",
    )

    # Timestamps
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
        comment="Timestamp when user was first created",
    )

    last_authenticated: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of most recent successful authentication",
    )

    last_seen: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when user was last active",
    )

    # Authentication tracking
    authentication_count: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of successful authentications",
    )

    # User status
    is_active: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether the user account is active",
    )

    # Preferences
    preferred_contact_method: Mapped[Optional[str]] = Column(
        String(10),  # 'sms' or 'email'
        nullable=True,
        comment="Preferred method for authentication (sms/email)",
    )

    timezone: Mapped[Optional[str]] = Column(
        String(50),
        nullable=True,
        comment="User's timezone for scheduling and timestamps",
    )

    language_preference: Mapped[Optional[str]] = Column(
        String(10), nullable=True, comment="Preferred language code (e.g., 'en', 'es')"
    )

    # Additional data
    metadata: Mapped[Optional[str]] = Column(
        Text, nullable=True, comment="Additional user metadata stored as JSON"
    )

    notes: Mapped[Optional[str]] = Column(
        Text, nullable=True, comment="Administrative notes about the user"
    )

    # Relationships
    auth_tokens: Mapped[List["AuthToken"]] = relationship(
        "AuthToken", back_populates="user", cascade="all, delete-orphan", lazy="dynamic"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_user_mobile_number", "mobile_number"),
        Index("idx_user_email", "email"),
        Index("idx_user_created_at", "created_at"),
        Index("idx_user_last_authenticated", "last_authenticated"),
        Index("idx_user_is_active", "is_active"),
        Index("idx_user_preferred_contact", "preferred_contact_method"),
    )

    def __repr__(self) -> str:
        """String representation of the user."""
        identifier = self.email or self.mobile_number or "no-contact"
        return (
            f"<User(id={self.id}, identifier='{identifier}', "
            f"created_at={self.created_at}, is_active={self.is_active})>"
        )

    @property
    def primary_identifier(self) -> Optional[str]:
        """
        Get the primary identifier for the user.

        Returns:
            Email if available, otherwise mobile number
        """
        return self.email or self.mobile_number

    @property
    def has_contact_info(self) -> bool:
        """Check if user has any contact information."""
        return bool(self.email or self.mobile_number)

    @property
    def can_authenticate_sms(self) -> bool:
        """Check if user can authenticate via SMS."""
        return bool(self.mobile_number)

    @property
    def can_authenticate_email(self) -> bool:
        """Check if user can authenticate via email."""
        return bool(self.email)

    @property
    def days_since_creation(self) -> int:
        """Calculate days since user was created."""
        delta = datetime.utcnow() - self.created_at
        return delta.days

    @property
    def days_since_last_auth(self) -> Optional[int]:
        """
        Calculate days since last authentication.

        Returns:
            Days since last auth, or None if never authenticated
        """
        if self.last_authenticated is None:
            return None

        delta = datetime.utcnow() - self.last_authenticated
        return delta.days

    @property
    def is_new_user(self) -> bool:
        """Check if user was created recently (within 7 days)."""
        return self.days_since_creation <= 7

    @property
    def has_been_authenticated(self) -> bool:
        """Check if user has ever been authenticated."""
        return self.authentication_count > 0

    def record_authentication(self) -> None:
        """Record a successful authentication."""
        self.last_authenticated = datetime.utcnow()
        self.authentication_count += 1
        self.last_seen = datetime.utcnow()

    def update_last_seen(self) -> None:
        """Update the last seen timestamp."""
        self.last_seen = datetime.utcnow()

    def deactivate(self) -> None:
        """Deactivate the user account."""
        self.is_active = False

    def activate(self) -> None:
        """Activate the user account."""
        self.is_active = True

    def set_preferred_contact_method(self, method: str) -> None:
        """
        Set the preferred contact method.

        Args:
            method: Either 'sms' or 'email'

        Raises:
            ValueError: If method is invalid or user doesn't have that contact info
        """
        if method not in ["sms", "email"]:
            raise ValueError("Contact method must be 'sms' or 'email'")

        if method == "sms" and not self.mobile_number:
            raise ValueError("Cannot set SMS as preferred method without mobile number")

        if method == "email" and not self.email:
            raise ValueError(
                "Cannot set email as preferred method without email address"
            )

        self.preferred_contact_method = method

    def get_contact_for_method(self, method: str) -> Optional[str]:
        """
        Get contact information for a specific method.

        Args:
            method: Either 'sms' or 'email'

        Returns:
            Contact information for the method, or None if not available
        """
        if method == "sms":
            return self.mobile_number
        elif method == "email":
            return self.email
        return None

    def get_available_auth_methods(self) -> List[str]:
        """
        Get list of available authentication methods for this user.

        Returns:
            List of available methods ('sms', 'email')
        """
        methods = []
        if self.mobile_number:
            methods.append("sms")
        if self.email:
            methods.append("email")
        return methods

    def get_preferred_auth_method(self) -> Optional[str]:
        """
        Get the preferred authentication method.

        Returns:
            Preferred method if set and available, otherwise first available method
        """
        available_methods = self.get_available_auth_methods()

        if not available_methods:
            return None

        if (
            self.preferred_contact_method
            and self.preferred_contact_method in available_methods
        ):
            return self.preferred_contact_method

        # Default preference: email over SMS
        if "email" in available_methods:
            return "email"
        elif "sms" in available_methods:
            return "sms"

        return None

    def clean_expired_tokens(self) -> int:
        """
        Remove expired authentication tokens for this user.

        Returns:
            Number of tokens removed
        """
        from .auth_token import AuthToken
        from sqlalchemy.orm import object_session

        session = object_session(self)
        if not session:
            return 0

        expired_tokens = (
            session.query(AuthToken)
            .filter(
                AuthToken.user_id == self.id, AuthToken.expires_at <= datetime.utcnow()
            )
            .all()
        )

        count = len(expired_tokens)
        for token in expired_tokens:
            session.delete(token)

        return count

    @classmethod
    def find_by_contact(cls, session: Session, contact: str) -> Optional["User"]:
        """
        Find a user by email or mobile number.

        Args:
            session: The database session to use for the query.
            contact: Email address or mobile number

        Returns:
            User if found, None otherwise
        """
        return session.query(cls).filter(
            or_(cls.email == contact, cls.mobile_number == contact)
        ).first()

    def validate_contact_info(self) -> bool:
        """
        Validate that user has at least one contact method.

        Returns:
            True if user has valid contact information
        """
        return self.has_contact_info

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """
        Convert user to dictionary representation.

        Args:
            include_sensitive: Whether to include sensitive information

        Returns:
            Dictionary representation of the user
        """
        data = {
            "id": str(self.id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "is_active": self.is_active,
            "authentication_count": self.authentication_count,
            "preferred_contact_method": self.preferred_contact_method,
            "language_preference": self.language_preference,
            "timezone": self.timezone,
        }

        if include_sensitive:
            data.update(
                {
                    "email": self.email,
                    "mobile_number": self.mobile_number,
                    "last_authenticated": (
                        self.last_authenticated.isoformat()
                        if self.last_authenticated
                        else None
                    ),
                }
            )

        return data
