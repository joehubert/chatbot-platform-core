"""
Conversation Manager Service

Handles conversation recording, metrics collection, and conversation lifecycle management.
Implements the requirements from the core platform specifications for conversation recording
and analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.core.database import get_async_db
from app.core.config import settings
from app.utils.conversation_utils import ConversationSummary, ConversationMetrics

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation lifecycle, recording, and analytics.

    This service handles:
    - Conversation creation and management
    - Message recording and storage
    - Conversation metrics collection
    - Resolution tracking
    - Conversation summarization for cache
    """

    def __init__(self, db: Session):
        self.db = db
        self.max_conversation_age = timedelta(hours=settings.CONVERSATION_MAX_AGE_HOURS)
        self.resolution_timeout = timedelta(minutes=settings.RESOLUTION_TIMEOUT_MINUTES)

    async def create_conversation(
        self,
        session_id: str,
        user_identifier: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """
        Create a new conversation record.

        Args:
            session_id: Session identifier
            user_identifier: Optional user identifier (mobile/email)
            context: Optional context information (page_url, user_agent, etc.)

        Returns:
            Conversation: Created conversation object
        """
        try:
            conversation = Conversation(
                id=uuid4(),
                session_id=session_id,
                user_identifier=user_identifier,
                started_at=datetime.utcnow(),
                context=context or {},
                resolved=False,
                resolution_attempts=0,
                authenticated=False,
            )

            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)

            logger.info(
                f"Created conversation {conversation.id} for session {session_id}"
            )
            return conversation

        except SQLAlchemyError as e:
            logger.error(f"Failed to create conversation: {str(e)}")
            self.db.rollback()
            raise

    async def get_conversation(
        self, session_id: str, create_if_not_exists: bool = True
    ) -> Optional[Conversation]:
        """
        Get or create a conversation by session ID.

        Args:
            session_id: Session identifier
            create_if_not_exists: Whether to create if not found

        Returns:
            Conversation: Found or created conversation
        """
        try:
            # Look for active conversation
            conversation = (
                self.db.query(Conversation)
                .filter(
                    and_(
                        Conversation.session_id == session_id,
                        Conversation.ended_at.is_(None),
                    )
                )
                .first()
            )

            if conversation:
                # Check if conversation is too old
                if (
                    datetime.utcnow() - conversation.started_at
                    > self.max_conversation_age
                ):
                    await self.end_conversation(conversation.id)
                    conversation = None

            if not conversation and create_if_not_exists:
                conversation = await self.create_conversation(session_id)

            return conversation

        except SQLAlchemyError as e:
            logger.error(f"Failed to get conversation: {str(e)}")
            raise

    async def add_message(
        self,
        conversation_id: UUID,
        content: str,
        role: str,
        model_used: Optional[str] = None,
        cached: bool = False,
        processing_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation ID
            content: Message content
            role: Message role (user, assistant, system)
            model_used: LLM model used for response
            cached: Whether response was cached
            processing_time_ms: Processing time in milliseconds
            metadata: Additional metadata

        Returns:
            Message: Created message object
        """
        try:
            message = Message(
                id=uuid4(),
                conversation_id=conversation_id,
                content=content,
                role=role,
                timestamp=datetime.utcnow(),
                model_used=model_used,
                cached=cached,
                processing_time_ms=processing_time_ms or 0,
                metadata=metadata or {},
            )

            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)

            logger.debug(
                f"Added message {message.id} to conversation {conversation_id}"
            )
            return message

        except SQLAlchemyError as e:
            logger.error(f"Failed to add message: {str(e)}")
            self.db.rollback()
            raise

    async def increment_resolution_attempts(self, conversation_id: UUID) -> None:
        """
        Increment the resolution attempts counter for a conversation.

        Args:
            conversation_id: Conversation ID
        """
        try:
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )

            if conversation:
                conversation.resolution_attempts += 1
                self.db.commit()

                logger.debug(
                    f"Incremented resolution attempts for conversation {conversation_id}"
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to increment resolution attempts: {str(e)}")
            self.db.rollback()
            raise

    async def mark_conversation_resolved(
        self, conversation_id: UUID, resolved: bool = True
    ) -> None:
        """
        Mark a conversation as resolved or unresolved.

        Args:
            conversation_id: Conversation ID
            resolved: Whether the conversation is resolved
        """
        try:
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )

            if conversation:
                conversation.resolved = resolved
                if resolved:
                    conversation.resolved_at = datetime.utcnow()
                else:
                    conversation.resolved_at = None

                self.db.commit()

                logger.info(
                    f"Marked conversation {conversation_id} as {'resolved' if resolved else 'unresolved'}"
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to mark conversation resolved: {str(e)}")
            self.db.rollback()
            raise

    async def mark_conversation_authenticated(self, conversation_id: UUID) -> None:
        """
        Mark a conversation as authenticated.

        Args:
            conversation_id: Conversation ID
        """
        try:
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )

            if conversation:
                conversation.authenticated = True
                self.db.commit()

                logger.info(f"Marked conversation {conversation_id} as authenticated")

        except SQLAlchemyError as e:
            logger.error(f"Failed to mark conversation authenticated: {str(e)}")
            self.db.rollback()
            raise

    async def end_conversation(self, conversation_id: UUID) -> None:
        """
        End a conversation by setting the ended_at timestamp.

        Args:
            conversation_id: Conversation ID
        """
        try:
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )

            if conversation and not conversation.ended_at:
                conversation.ended_at = datetime.utcnow()
                self.db.commit()

                logger.info(f"Ended conversation {conversation_id}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to end conversation: {str(e)}")
            self.db.rollback()
            raise

    async def get_conversation_summary(
        self, conversation_id: UUID
    ) -> ConversationSummary:
        """
        Generate a summary of a conversation for caching purposes.

        Args:
            conversation_id: Conversation ID

        Returns:
            ConversationSummary: Summary object
        """
        try:
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )

            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            messages = (
                self.db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp)
                .all()
            )

            user_messages = [m for m in messages if m.role == "user"]
            assistant_messages = [m for m in messages if m.role == "assistant"]

            return ConversationSummary(
                conversation_id=conversation_id,
                session_id=conversation.session_id,
                user_messages=[m.content for m in user_messages],
                assistant_messages=[m.content for m in assistant_messages],
                resolved=conversation.resolved,
                resolution_attempts=conversation.resolution_attempts,
                authenticated=conversation.authenticated,
                started_at=conversation.started_at,
                ended_at=conversation.ended_at,
                total_messages=len(messages),
                avg_processing_time_ms=sum(m.processing_time_ms for m in messages)
                / len(messages)
                if messages
                else 0,
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to generate conversation summary: {str(e)}")
            raise

    async def get_conversation_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> ConversationMetrics:
        """
        Get conversation metrics for analytics.

        Args:
            start_date: Start date for metrics (default: 24 hours ago)
            end_date: End date for metrics (default: now)
            limit: Maximum number of conversations to analyze

        Returns:
            ConversationMetrics: Metrics object
        """
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=1)
            if not end_date:
                end_date = datetime.utcnow()

            # Base query for time range
            base_query = self.db.query(Conversation).filter(
                and_(
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date,
                )
            )

            # Total conversations
            total_conversations = base_query.count()

            # Resolved conversations
            resolved_conversations = base_query.filter(
                Conversation.resolved == True
            ).count()

            # Resolution rate
            resolution_rate = (
                (resolved_conversations / total_conversations)
                if total_conversations > 0
                else 0
            )

            # Average resolution attempts
            avg_resolution_attempts = (
                self.db.query(func.avg(Conversation.resolution_attempts))
                .filter(
                    and_(
                        Conversation.started_at >= start_date,
                        Conversation.started_at <= end_date,
                    )
                )
                .scalar()
                or 0
            )

            # Average conversation duration
            avg_duration_query = (
                self.db.query(
                    func.avg(
                        func.extract(
                            "epoch", Conversation.ended_at - Conversation.started_at
                        )
                    )
                )
                .filter(
                    and_(
                        Conversation.started_at >= start_date,
                        Conversation.started_at <= end_date,
                        Conversation.ended_at.isnot(None),
                    )
                )
                .scalar()
            )

            avg_duration_seconds = avg_duration_query or 0

            # Message statistics
            message_stats = (
                self.db.query(
                    func.count(Message.id).label("total_messages"),
                    func.avg(Message.processing_time_ms).label("avg_processing_time"),
                    func.count(Message.id)
                    .filter(Message.cached == True)
                    .label("cached_messages"),
                )
                .join(Conversation)
                .filter(
                    and_(
                        Conversation.started_at >= start_date,
                        Conversation.started_at <= end_date,
                    )
                )
                .first()
            )

            total_messages = message_stats.total_messages or 0
            avg_processing_time = message_stats.avg_processing_time or 0
            cached_messages = message_stats.cached_messages or 0
            cache_hit_rate = (
                (cached_messages / total_messages) if total_messages > 0 else 0
            )

            return ConversationMetrics(
                total_conversations=total_conversations,
                resolved_conversations=resolved_conversations,
                resolution_rate=resolution_rate,
                avg_resolution_attempts=avg_resolution_attempts,
                avg_duration_seconds=avg_duration_seconds,
                total_messages=total_messages,
                avg_processing_time_ms=avg_processing_time,
                cache_hit_rate=cache_hit_rate,
                start_date=start_date,
                end_date=end_date,
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to get conversation metrics: {str(e)}")
            raise

    async def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """
        Clean up old conversations and their messages.

        Args:
            days_old: Number of days old to consider for cleanup

        Returns:
            int: Number of conversations deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # Find old conversations
            old_conversations = (
                self.db.query(Conversation)
                .filter(Conversation.started_at < cutoff_date)
                .all()
            )

            deleted_count = 0
            for conversation in old_conversations:
                # Delete messages first
                self.db.query(Message).filter(
                    Message.conversation_id == conversation.id
                ).delete()

                # Delete conversation
                self.db.delete(conversation)
                deleted_count += 1

            self.db.commit()

            logger.info(f"Cleaned up {deleted_count} old conversations")
            return deleted_count

        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old conversations: {str(e)}")
            self.db.rollback()
            raise

    async def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        try:
            conversation = await self.get_conversation(
                session_id, create_if_not_exists=False
            )

            if not conversation:
                return []

            messages = (
                self.db.query(Message)
                .filter(Message.conversation_id == conversation.id)
                .order_by(Message.timestamp.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": str(message.id),
                    "content": message.content,
                    "role": message.role,
                    "timestamp": message.timestamp.isoformat(),
                    "model_used": message.model_used,
                    "cached": message.cached,
                    "processing_time_ms": message.processing_time_ms,
                }
                for message in reversed(messages)
            ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            raise

    async def get_active_conversation_count(self) -> int:
        """
        Get count of active (not ended) conversations.

        Returns:
            int: Number of active conversations
        """
        try:
            return (
                self.db.query(Conversation)
                .filter(Conversation.ended_at.is_(None))
                .count()
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to get active conversation count: {str(e)}")
            raise

    async def get_conversations_by_user(
        self, user_identifier: str, limit: int = 10
    ) -> List[Conversation]:
        """
        Get conversations for a specific user.

        Args:
            user_identifier: User identifier (mobile/email)
            limit: Maximum number of conversations to return

        Returns:
            List of conversations
        """
        try:
            return (
                self.db.query(Conversation)
                .filter(Conversation.user_identifier == user_identifier)
                .order_by(Conversation.started_at.desc())
                .limit(limit)
                .all()
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to get conversations by user: {str(e)}")
            raise


# Dependency injection helper
async def get_conversation_manager(db: Session = None) -> ConversationManager:
    """
    Get a ConversationManager instance with database dependency.

    Args:
        db: Database session (will be injected by FastAPI)

    Returns:
        ConversationManager: Initialized conversation manager
    """
    if db is None:
        db = next(get_async_db())

    return ConversationManager(db)
