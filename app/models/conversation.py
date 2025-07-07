"""
Conversation model for the Turnkey AI Chatbot platform.

This module defines the Conversation model that represents a complete
conversation session between a user and the chatbot, including all
related messages and metadata.
"""

import uuid
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Integer,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped
from .base import Base

if TYPE_CHECKING:
    from .message import Message


class Conversation(Base):
    """
    Conversation model representing a chat session.

    A conversation tracks a complete interaction session between a user
    and the chatbot, including session management, resolution tracking,
    and authentication status.

    Attributes:
        id: Unique identifier for the conversation
        session_id: Session identifier for tracking user sessions
        user_identifier: Optional user identifier (email/phone)
        started_at: Timestamp when conversation began
        ended_at: Timestamp when conversation ended (if completed)
        resolved: Whether the conversation was successfully resolved
        resolution_attempts: Number of attempts made to resolve the query
        authenticated: Whether user was authenticated during conversation
        context_data: Additional context information (JSON)
        user_agent: User agent string from the client
        ip_address: Client IP address (for analytics/security)
        messages: Related messages in this conversation
    """

    __tablename__ = "conversations"

    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the conversation",
    )

    # Session management
    session_id: Mapped[str] = Column(
        String(255),
        nullable=False,
        index=True,
        comment="Session identifier for tracking user sessions",
    )

    # User identification (optional)
    user_identifier: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Optional user identifier (email or phone number)",
    )

    # Timestamps
    started_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Timestamp when conversation began",
    )

    ended_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when conversation ended",
    )

    # Resolution tracking
    resolved: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the conversation was successfully resolved",
    )

    resolution_attempts: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of attempts made to resolve the query",
    )

    # Authentication status
    authenticated: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether user was authenticated during conversation",
    )

    # Context and metadata
    context_data: Mapped[Optional[str]] = Column(
        Text, nullable=True, comment="Additional context information stored as JSON"
    )

    user_agent: Mapped[Optional[str]] = Column(
        String(512), nullable=True, comment="User agent string from the client"
    )

    ip_address: Mapped[Optional[str]] = Column(
        String(45),  # IPv6 max length
        nullable=True,
        comment="Client IP address for analytics and security",
    )

    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="Message.timestamp.asc()",
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_conversation_session_id", "session_id"),
        Index("idx_conversation_user_identifier", "user_identifier"),
        Index("idx_conversation_started_at", "started_at"),
        Index("idx_conversation_resolved", "resolved"),
        Index("idx_conversation_authenticated", "authenticated"),
        Index("idx_conversation_session_started", "session_id", "started_at"),
    )

    def __repr__(self) -> str:
        """String representation of the conversation."""
        return (
            f"<Conversation(id={self.id}, session_id='{self.session_id}', "
            f"started_at={self.started_at}, resolved={self.resolved})>"
        )

    @property
    def duration_seconds(self) -> Optional[int]:
        """
        Calculate conversation duration in seconds.

        Returns:
            Duration in seconds if conversation has ended, None otherwise
        """
        if self.ended_at and self.started_at:
            return int((self.ended_at - self.started_at).total_seconds())
        return None

    @property
    def is_active(self) -> bool:
        """
        Check if conversation is currently active.

        Returns:
            True if conversation is ongoing, False if ended
        """
        return self.ended_at is None

    @property
    def message_count(self) -> int:
        """
        Get the total number of messages in this conversation.

        Returns:
            Total count of messages
        """
        return self.messages.count()

    def mark_resolved(self, resolved: bool = True) -> None:
        """
        Mark the conversation as resolved or unresolved.

        Args:
            resolved: Whether the conversation is resolved
        """
        self.resolved = resolved
        if resolved and not self.ended_at:
            self.ended_at = datetime.utcnow()

    def increment_resolution_attempts(self) -> None:
        """Increment the resolution attempts counter."""
        self.resolution_attempts += 1

    def mark_authenticated(self) -> None:
        """Mark the conversation as authenticated."""
        self.authenticated = True

    def end_conversation(self) -> None:
        """Mark the conversation as ended."""
        if not self.ended_at:
            self.ended_at = datetime.utcnow()
