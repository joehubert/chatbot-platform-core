"""
Message model for the Turnkey AI Chatbot platform.

This module defines the Message model that represents individual messages
within a conversation, tracking content, metadata, and processing information.
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Integer,
    Text,
    ForeignKey,
    Index,
    Enum as SQLEnum,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship, Mapped
from .base import Base

if TYPE_CHECKING:
    from .conversation import Conversation


class MessageRole(str, Enum):
    """
    Enumeration of possible message roles.

    Values:
        USER: Message from the user
        ASSISTANT: Message from the chatbot assistant
        SYSTEM: System-generated message (e.g., notifications, errors)
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(Base):
    """
    Message model representing individual messages in conversations.

    Each message tracks the content, sender role, processing metadata,
    and performance metrics for analytics and optimization.

    Attributes:
        id: Unique identifier for the message
        conversation_id: Foreign key to the parent conversation
        content: The actual message content/text
        role: Role of the message sender (user/assistant/system)
        timestamp: When the message was created
        model_used: LLM model used to generate response (for assistant messages)
        cached: Whether the response was served from cache
        processing_time_ms: Time taken to process/generate the message
        tokens_used: Number of tokens consumed (for LLM calls)
        confidence_score: Confidence score for the response (0.0-1.0)
        requires_clarification: Whether the message needs clarification
        metadata: Additional metadata stored as JSON
        conversation: Reference to the parent conversation
    """

    __tablename__ = "messages"

    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the message",
    )

    # Foreign key to conversation
    conversation_id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the parent conversation",
    )

    # Message content
    content: Mapped[str] = Column(
        Text, nullable=False, comment="The actual message content/text"
    )

    # Message role
    role: Mapped[MessageRole] = Column(
        SQLEnum(MessageRole),
        nullable=False,
        index=True,
        comment="Role of the message sender (user/assistant/system)",
    )

    # Timestamp
    timestamp: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
        comment="When the message was created",
    )

    # Processing metadata
    model_used: Mapped[Optional[str]] = Column(
        String(100),
        nullable=True,
        comment="LLM model used to generate response (for assistant messages)",
    )

    cached: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the response was served from cache",
    )

    processing_time_ms: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True,
        comment="Time taken to process/generate the message in milliseconds",
    )

    # Token usage and costs
    tokens_used: Mapped[Optional[int]] = Column(
        Integer, nullable=True, comment="Number of tokens consumed for LLM calls"
    )

    # Quality metrics
    confidence_score: Mapped[Optional[float]] = Column(
        Float,
        nullable=True,
        comment="Confidence score for the response (0.0-1.0)",
    )

    requires_clarification: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the message needs clarification",
    )

    # Additional data
    data: Mapped[Optional[dict]] = Column(
        JSON, nullable=True, comment="Additional data stored as JSON"
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages", lazy="select"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_message_conversation_id", "conversation_id"),
        Index("idx_message_role", "role"),
        Index("idx_message_timestamp", "timestamp"),
        Index("idx_message_cached", "cached"),
        Index("idx_message_model_used", "model_used"),
        Index("idx_message_conversation_timestamp", "conversation_id", "timestamp"),
        Index("idx_message_role_timestamp", "role", "timestamp"),
    )

    def __repr__(self) -> str:
        """String representation of the message."""
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return (
            f"<Message(id={self.id}, role={self.role.value}, "
            f"content='{content_preview}', timestamp={self.timestamp})>"
        )

    @property
    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.role == MessageRole.USER

    @property
    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message."""
        return self.role == MessageRole.ASSISTANT

    @property
    def is_system_message(self) -> bool:
        """Check if this is a system message."""
        return self.role == MessageRole.SYSTEM

    @property
    def processing_time_seconds(self) -> Optional[float]:
        """
        Get processing time in seconds.

        Returns:
            Processing time in seconds, or None if not recorded
        """
        if self.processing_time_ms is not None:
            return self.processing_time_ms / 1000.0
        return None

    @property
    def content_length(self) -> int:
        """Get the character count of the message content."""
        return len(self.content) if self.content else 0

    @property
    def word_count(self) -> int:
        """Get the approximate word count of the message content."""
        if not self.content:
            return 0
        return len(self.content.split())

    def mark_as_cached(self) -> None:
        """Mark this message as served from cache."""
        self.cached = True

    def set_processing_time(self, milliseconds: int) -> None:
        """
        Set the processing time for this message.

        Args:
            milliseconds: Processing time in milliseconds
        """
        self.processing_time_ms = max(0, milliseconds)

    def set_model_usage(self, model_name: str, tokens: Optional[int] = None) -> None:
        """
        Set the model and token usage information.

        Args:
            model_name: Name of the LLM model used
            tokens: Number of tokens consumed (optional)
        """
        self.model_used = model_name
        if tokens is not None:
            self.tokens_used = max(0, tokens)

    def set_confidence_score(self, score: float) -> None:
        """
        Set the confidence score for this message.

        Args:
            score: Confidence score between 0.0 and 1.0
        """
        self.confidence_score = max(0.0, min(1.0, score))

    def mark_needs_clarification(self) -> None:
        """Mark this message as needing clarification."""
        self.requires_clarification = True

    def is_recent(self, minutes: int = 5) -> bool:
        """
        Check if the message was created recently.

        Args:
            minutes: Number of minutes to consider as recent

        Returns:
            True if message was created within the specified minutes
        """
        from datetime import timedelta

        threshold = datetime.utcnow() - timedelta(minutes=minutes)
        return self.timestamp >= threshold
