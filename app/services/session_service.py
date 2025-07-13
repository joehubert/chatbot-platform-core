"""
Session Manager Service

Handles session management, context preservation, and conversation flow control.
Implements session timeout, context management, and conversation summarization as per
the core platform requirements.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

import redis
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.redis import get_redis_client
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.conversation_manager import ConversationManager
from app.utils.conversation_utils import ConversationContext, SessionData

logger = logging.getLogger(__name__)


class SessionService:
    """
    Manages user sessions, context preservation, and conversation flow.

    This service handles:
    - Session creation and management
    - Context preservation across requests
    - Conversation summarization for context management
    - Session timeout and cleanup
    - Authentication state management
    """

    def __init__(self, redis_client: redis.Redis = None, db: Session = None):
        self.redis_client = redis_client or get_redis_client()
        self.db = db
        self.session_timeout = timedelta(minutes=settings.AUTH_SESSION_TIMEOUT_MINUTES)
        self.context_max_turns = settings.MAX_CONTEXT_TURNS
        self.summarization_trigger = settings.SUMMARIZATION_TRIGGER_TURNS

        # Redis key prefixes
        self.session_prefix = "session:"
        self.context_prefix = "context:"
        self.auth_prefix = "auth:"

    async def create_session(
        self,
        user_identifier: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new session.

        Args:
            user_identifier: Optional user identifier
            initial_context: Initial context data

        Returns:
            str: Session ID
        """
        session_id = str(uuid4())

        session_data = SessionData(
            session_id=session_id,
            user_identifier=user_identifier,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            authenticated=False,
            context=initial_context or {},
            conversation_count=0,
            total_messages=0,
        )

        # Store session data in Redis
        await self._store_session_data(session_id, session_data)

        logger.info(f"Created session {session_id} for user {user_identifier}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data by session ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionData or None if not found/expired
        """
        try:
            session_key = f"{self.session_prefix}{session_id}"
            session_data_json = await self.redis_client.get(session_key)

            if not session_data_json:
                return None

            session_data = SessionData.from_json(session_data_json)

            # Check if session is expired
            if datetime.utcnow() - session_data.last_activity > self.session_timeout:
                await self.end_session(session_id)
                return None

            return session_data

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {str(e)}")
            return None

    async def update_session_activity(self, session_id: str) -> None:
        """
        Update session last activity timestamp.

        Args:
            session_id: Session identifier
        """
        try:
            session_data = await self.get_session(session_id)
            if session_data:
                session_data.last_activity = datetime.utcnow()
                await self._store_session_data(session_id, session_data)

        except Exception as e:
            logger.error(f"Failed to update session activity: {str(e)}")

    async def mark_session_authenticated(
        self, session_id: str, user_identifier: str
    ) -> None:
        """
        Mark a session as authenticated.

        Args:
            session_id: Session identifier
            user_identifier: User identifier (mobile/email)
        """
        try:
            session_data = await self.get_session(session_id)
            if session_data:
                session_data.authenticated = True
                session_data.user_identifier = user_identifier
                session_data.authenticated_at = datetime.utcnow()
                await self._store_session_data(session_id, session_data)

                # Store auth state separately for quick lookup
                auth_key = f"{self.auth_prefix}{session_id}"
                auth_data = {
                    "user_identifier": user_identifier,
                    "authenticated_at": datetime.utcnow().isoformat(),
                }
                await self.redis_client.setex(
                    auth_key,
                    int(self.session_timeout.total_seconds()),
                    json.dumps(auth_data),
                )

                logger.info(
                    f"Marked session {session_id} as authenticated for user {user_identifier}"
                )

        except Exception as e:
            logger.error(f"Failed to mark session authenticated: {str(e)}")

    async def is_session_authenticated(self, session_id: str) -> bool:
        """
        Check if a session is authenticated.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if authenticated
        """
        try:
            # Quick check from auth cache
            auth_key = f"{self.auth_prefix}{session_id}"
            auth_data = await self.redis_client.get(auth_key)

            if auth_data:
                return True

            # Fallback to session data
            session_data = await self.get_session(session_id)
            return session_data.authenticated if session_data else False

        except Exception as e:
            logger.error(f"Failed to check session authentication: {str(e)}")
            return False

    async def get_conversation_context(
        self, session_id: str, conversation_manager: ConversationManager
    ) -> ConversationContext:
        """
        Get conversation context for a session.

        Args:
            session_id: Session identifier
            conversation_manager: Conversation manager instance

        Returns:
            ConversationContext: Context object
        """
        try:
            context_key = f"{self.context_prefix}{session_id}"
            context_data = await self.redis_client.get(context_key)

            if context_data:
                return ConversationContext.from_json(context_data)

            # Build context from conversation history
            conversation = await conversation_manager.get_conversation(
                session_id, create_if_not_exists=False
            )

            if not conversation:
                return ConversationContext(
                    session_id=session_id,
                    messages=[],
                    entities={},
                    summary="",
                    turn_count=0,
                    last_updated=datetime.utcnow(),
                )

            # Get recent messages
            messages = (
                self.db.query(Message)
                .filter(Message.conversation_id == conversation.id)
                .order_by(Message.timestamp.desc())
                .limit(self.context_max_turns * 2)
                .all()
            )

            context = ConversationContext(
                session_id=session_id,
                messages=[
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                    }
                    for msg in reversed(messages)
                ],
                entities={},
                summary="",
                turn_count=len(messages),
                last_updated=datetime.utcnow(),
            )

            # Store context for future use
            await self._store_context(session_id, context)

            return context

        except Exception as e:
            logger.error(f"Failed to get conversation context: {str(e)}")
            return ConversationContext(
                session_id=session_id,
                messages=[],
                entities={},
                summary="",
                turn_count=0,
                last_updated=datetime.utcnow(),
            )

    async def update_conversation_context(
        self,
        session_id: str,
        new_message: Dict[str, Any],
        entities: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update conversation context with new message.

        Args:
            session_id: Session identifier
            new_message: New message to add
            entities: Extracted entities to update
        """
        try:
            context_key = f"{self.context_prefix}{session_id}"
            context_data = await self.redis_client.get(context_key)

            if context_data:
                context = ConversationContext.from_json(context_data)
            else:
                context = ConversationContext(
                    session_id=session_id,
                    messages=[],
                    entities={},
                    summary="",
                    turn_count=0,
                    last_updated=datetime.utcnow(),
                )

            # Add new message
            context.messages.append(new_message)
            context.turn_count += 1
            context.last_updated = datetime.utcnow()

            # Update entities if provided
            if entities:
                context.entities.update(entities)

            # Check if we need to summarize
            if context.turn_count >= self.summarization_trigger:
                await self._summarize_context(context)

            # Keep only recent messages
            if len(context.messages) > self.context_max_turns:
                context.messages = context.messages[-self.context_max_turns :]

            await self._store_context(session_id, context)

        except Exception as e:
            logger.error(f"Failed to update conversation context: {str(e)}")

    async def _summarize_context(self, context: ConversationContext) -> None:
        """
        Summarize conversation context to compress older messages.

        Args:
            context: Conversation context to summarize
        """
        try:
            # Simple summarization - in production, this would use an LLM
            # For now, we'll create a basic summary

            if not context.messages:
                return

            # Count message types
            user_messages = [msg for msg in context.messages if msg["role"] == "user"]
            assistant_messages = [
                msg for msg in context.messages if msg["role"] == "assistant"
            ]

            # Create summary
            summary_parts = []

            if user_messages:
                recent_user_queries = [
                    msg["content"][:50] + "..."
                    if len(msg["content"]) > 50
                    else msg["content"]
                    for msg in user_messages[-3:]
                ]
                summary_parts.append(
                    f"Recent user queries: {'; '.join(recent_user_queries)}"
                )

            if assistant_messages:
                summary_parts.append(f"Provided {len(assistant_messages)} responses")

            if context.entities:
                summary_parts.append(f"Discussed: {', '.join(context.entities.keys())}")

            context.summary = (
                " | ".join(summary_parts) if summary_parts else "No significant content"
            )

            # Keep only the most recent messages after summarization
            recent_count = max(4, self.context_max_turns // 2)
            context.messages = context.messages[-recent_count:]

            logger.debug(f"Summarized context for session {context.session_id}")

        except Exception as e:
            logger.error(f"Failed to summarize context: {str(e)}")

    async def clear_session_context(self, session_id: str) -> None:
        """
        Clear conversation context for a session.

        Args:
            session_id: Session identifier
        """
        try:
            context_key = f"{self.context_prefix}{session_id}"
            await self.redis_client.delete(context_key)

            logger.debug(f"Cleared context for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to clear session context: {str(e)}")

    async def end_session(self, session_id: str) -> None:
        """
        End a session and clean up associated data.

        Args:
            session_id: Session identifier
        """
        try:
            # Clean up session data
            session_key = f"{self.session_prefix}{session_id}"
            context_key = f"{self.context_prefix}{session_id}"
            auth_key = f"{self.auth_prefix}{session_id}"

            # Delete all session-related keys
            await self.redis_client.delete(session_key, context_key, auth_key)

            logger.info(f"Ended session {session_id}")

        except Exception as e:
            logger.error(f"Failed to end session: {str(e)}")

    async def extend_session(self, session_id: str) -> bool:
        """
        Extend session timeout.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session was extended
        """
        try:
            session_data = await self.get_session(session_id)
            if session_data:
                session_data.last_activity = datetime.utcnow()
                await self._store_session_data(session_id, session_data)

                # Extend auth timeout if authenticated
                if session_data.authenticated:
                    auth_key = f"{self.auth_prefix}{session_id}"
                    await self.redis_client.expire(
                        auth_key, int(self.session_timeout.total_seconds())
                    )

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to extend session: {str(e)}")
            return False

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get session statistics.

        Args:
            session_id: Session identifier

        Returns:
            Dict with session statistics
        """
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return {}

            context = await self.get_conversation_context(session_id, None)

            duration = datetime.utcnow() - session_data.created_at

            return {
                "session_id": session_id,
                "user_identifier": session_data.user_identifier,
                "created_at": session_data.created_at.isoformat(),
                "last_activity": session_data.last_activity.isoformat(),
                "duration_seconds": int(duration.total_seconds()),
                "authenticated": session_data.authenticated,
                "conversation_count": session_data.conversation_count,
                "total_messages": session_data.total_messages,
                "context_turn_count": context.turn_count,
                "context_has_summary": bool(context.summary),
                "entities_count": len(context.entities) if context.entities else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return {}

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            int: Number of sessions cleaned up
        """
        try:
            # Get all session keys
            session_keys = await self.redis_client.keys(f"{self.session_prefix}*")

            cleaned_count = 0
            for session_key in session_keys:
                try:
                    session_data_json = await self.redis_client.get(session_key)
                    if session_data_json:
                        session_data = SessionData.from_json(session_data_json)

                        # Check if expired
                        if (
                            datetime.utcnow() - session_data.last_activity
                            > self.session_timeout
                        ):
                            session_id = session_data.session_id
                            await self.end_session(session_id)
                            cleaned_count += 1

                except Exception as e:
                    logger.error(
                        f"Error processing session key {session_key}: {str(e)}"
                    )
                    continue

            logger.info(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return 0

    async def get_active_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            int: Number of active sessions
        """
        try:
            session_keys = await self.redis_client.keys(f"{self.session_prefix}*")
            return len(session_keys)

        except Exception as e:
            logger.error(f"Failed to get active session count: {str(e)}")
            return 0

    async def _store_session_data(
        self, session_id: str, session_data: SessionData
    ) -> None:
        """
        Store session data in Redis.

        Args:
            session_id: Session identifier
            session_data: Session data object
        """
        session_key = f"{self.session_prefix}{session_id}"
        await self.redis_client.setex(
            session_key,
            int(self.session_timeout.total_seconds()),
            session_data.to_json(),
        )

    async def _store_context(
        self, session_id: str, context: ConversationContext
    ) -> None:
        """
        Store conversation context in Redis.

        Args:
            session_id: Session identifier
            context: Conversation context object
        """
        context_key = f"{self.context_prefix}{session_id}"
        await self.redis_client.setex(
            context_key, int(self.session_timeout.total_seconds()), context.to_json()
        )


# Dependency injection helper
async def get_session_service(
    redis_client: redis.Redis = None, db: Session = None
) -> SessionService:
    """
    Get a SessionService instance with dependencies.

    Args:
        redis_client: Redis client (will be injected)
        db: Database session (will be injected)

    Returns:
        SessionService: Initialized session service
    """
    return SessionService(redis_client, db)


# Add sync wrapper methods for API compatibility
def get_session_sync(self, session_id: str) -> Optional[SessionData]:
    """Synchronous wrapper for get_session."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running in async context - need to use create_task
            task = asyncio.create_task(self.get_session(session_id))
            return asyncio.run_coroutine_threadsafe(task, loop).result()
        else:
            return asyncio.run(self.get_session(session_id))
    except Exception as e:
        logger.error(f"Error in get_session_sync: {str(e)}")
        return None


def is_session_authenticated_sync(self, session_id: str) -> bool:
    """Synchronous wrapper for is_session_authenticated."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.create_task(self.is_session_authenticated(session_id))
            return asyncio.run_coroutine_threadsafe(task, loop).result()
        else:
            return asyncio.run(self.is_session_authenticated(session_id))
    except Exception as e:
        logger.error(f"Error in is_session_authenticated_sync: {str(e)}")
        return False


def authenticate_session_sync(
    self, session_id: str, user_identifier: str
) -> Optional[SessionData]:
    """Synchronous wrapper for authenticate_session."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.create_task(
                self.mark_session_authenticated(session_id, user_identifier)
            )
            asyncio.run_coroutine_threadsafe(task, loop).result()
        else:
            asyncio.run(self.mark_session_authenticated(session_id, user_identifier))

        return self.get_session_sync(session_id)
    except Exception as e:
        logger.error(f"Error in authenticate_session_sync: {str(e)}")
        return None


def is_session_expired(self, session: SessionData) -> bool:
    """Check if a session is expired based on SessionData object."""
    try:
        return datetime.utcnow() - session.last_activity > self.session_timeout
    except Exception as e:
        logger.error(f"Error checking session expiry: {str(e)}")
        return True  # Assume expired on error for security


def get_session_expiry(self, session_id: str) -> Optional[datetime]:
    """Get session expiration time."""
    try:
        session = self.get_session_sync(session_id)
        if not session:
            return None
        return session.last_activity + self.session_timeout
    except Exception as e:
        logger.error(f"Error getting session expiry: {str(e)}")
        return None


def delete_session_sync(self, session_id: str) -> bool:
    """Synchronous wrapper for delete_session."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.create_task(self.end_session(session_id))
            asyncio.run_coroutine_threadsafe(task, loop).result()
        else:
            asyncio.run(self.end_session(session_id))
        return True
    except Exception as e:
        logger.error(f"Error in delete_session_sync: {str(e)}")
        return False
