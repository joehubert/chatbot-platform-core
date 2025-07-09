"""
Tests for conversation models.
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from uuid import uuid4

from app.models.conversation import Conversation, Message
from app.models.user import User


class TestConversationModel:
    """Test cases for Conversation model."""

    @pytest.mark.asyncio
    async def test_create_conversation(
        self, 
        db_session: AsyncSession,
        sample_user: User
    ):
        """Test creating a new conversation."""
        conversation = Conversation(
            id=uuid4(),
            session_id="test_session_456",
            user_identifier=sample_user.email,
            user_id=sample_user.id,
            resolved=False,
            resolution_attempts=0,
            authenticated=False
        )
        
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        assert conversation.id is not None
        assert conversation.session_id == "test_session_456"
        assert conversation.user_identifier == sample_user.email
        assert conversation.user_id == sample_user.id
        assert conversation.resolved is False
        assert conversation.resolution_attempts == 0
        assert conversation.authenticated is False
        assert conversation.started_at is not None
        assert conversation.ended_at is None

    @pytest.mark.asyncio
    async def test_conversation_with_messages(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test conversation with associated messages."""
        # Create messages
        message1 = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Hello, I need help",
            role="user",
            cached=False,
            processing_time_ms=100
        )
        
        message2 = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="How can I help you?",
            role="assistant",
            model_used="gpt-3.5-turbo",
            cached=False,
            processing_time_ms=1500
        )
        
        db_session.add_all([message1, message2])
        await db_session.commit()
        
        # Refresh conversation to load messages
        await db_session.refresh(sample_conversation)
        
        assert len(sample_conversation.messages) == 2
        assert sample_conversation.messages[0].content == "Hello, I need help"
        assert sample_conversation.messages[1].content == "How can I help you?"

    @pytest.mark.asyncio
    async def test_conversation_user_relationship(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation,
        sample_user: User
    ):
        """Test conversation-user relationship."""
        # Refresh conversation to load user
        await db_session.refresh(sample_conversation)
        
        assert sample_conversation.user is not None
        assert sample_conversation.user.id == sample_user.id
        assert sample_conversation.user.email == sample_user.email

    @pytest.mark.asyncio
    async def test_conversation_without_user(
        self, 
        db_session: AsyncSession
    ):
        """Test conversation without associated user."""
        conversation = Conversation(
            id=uuid4(),
            session_id="anonymous_session",
            user_identifier="anonymous@example.com",
            user_id=None,
            resolved=False,
            resolution_attempts=0,
            authenticated=False
        )
        
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        assert conversation.user_id is None
        assert conversation.user is None
        assert conversation.user_identifier == "anonymous@example.com"

    @pytest.mark.asyncio
    async def test_conversation_resolution_tracking(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test conversation resolution tracking."""
        # Initially not resolved
        assert sample_conversation.resolved is False
        assert sample_conversation.resolution_attempts == 0
        
        # Increment resolution attempts
        sample_conversation.resolution_attempts += 1
        await db_session.commit()
        
        # Mark as resolved
        sample_conversation.resolved = True
        sample_conversation.ended_at = datetime.utcnow()
        await db_session.commit()
        
        await db_session.refresh(sample_conversation)
        
        assert sample_conversation.resolved is True
        assert sample_conversation.resolution_attempts == 1
        assert sample_conversation.ended_at is not None

    @pytest.mark.asyncio
    async def test_conversation_authentication_tracking(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test conversation authentication tracking."""
        # Initially not authenticated
        assert sample_conversation.authenticated is False
        
        # Mark as authenticated
        sample_conversation.authenticated = True
        await db_session.commit()
        
        await db_session.refresh(sample_conversation)
        
        assert sample_conversation.authenticated is True

    @pytest.mark.asyncio
    async def test_conversation_query_by_session(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test querying conversation by session ID."""
        stmt = select(Conversation).where(
            Conversation.session_id == sample_conversation.session_id
        )
        result = await db_session.execute(stmt)
        found_conversation = result.scalar_one()
        
        assert found_conversation.id == sample_conversation.id
        assert found_conversation.session_id == sample_conversation.session_id

    @pytest.mark.asyncio
    async def test_conversation_query_by_user(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation,
        sample_user: User
    ):
        """Test querying conversations by user."""
        stmt = select(Conversation).where(
            Conversation.user_id == sample_user.id
        )
        result = await db_session.execute(stmt)
        conversations = result.scalars().all()
        
        assert len(conversations) >= 1
        assert sample_conversation.id in [c.id for c in conversations]

    @pytest.mark.asyncio
    async def test_conversation_timestamps(
        self, 
        db_session: AsyncSession,
        sample_user: User
    ):
        """Test conversation timestamp handling."""
        start_time = datetime.utcnow()
        
        conversation = Conversation(
            id=uuid4(),
            session_id="timestamp_test_session",
            user_identifier=sample_user.email,
            user_id=sample_user.id,
            resolved=False,
            resolution_attempts=0,
            authenticated=False
        )
        
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        # Check that started_at is set and recent
        assert conversation.started_at is not None
        assert conversation.started_at >= start_time
        assert conversation.ended_at is None
        
        # End the conversation
        end_time = datetime.utcnow()
        conversation.ended_at = end_time
        await db_session.commit()
        
        await db_session.refresh(conversation)
        assert conversation.ended_at == end_time

    @pytest.mark.asyncio
    async def test_conversation_cascade_delete(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test that deleting conversation cascades to messages."""
        # Create a message
        message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Test message for cascade",
            role="user",
            cached=False,
            processing_time_ms=100
        )
        
        db_session.add(message)
        await db_session.commit()
        
        # Verify message exists
        stmt = select(Message).where(Message.conversation_id == sample_conversation.id)
        result = await db_session.execute(stmt)
        messages = result.scalars().all()
        assert len(messages) == 1
        
        # Delete conversation
        await db_session.delete(sample_conversation)
        await db_session.commit()
        
        # Verify messages are also deleted
        stmt = select(Message).where(Message.conversation_id == sample_conversation.id)
        result = await db_session.execute(stmt)
        messages = result.scalars().all()
        assert len(messages) == 0


class TestMessageModel:
    """Test cases for Message model."""

    @pytest.mark.asyncio
    async def test_create_message(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test creating a new message."""
        message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Test message content",
            role="user",
            cached=False,
            processing_time_ms=250
        )
        
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.id is not None
        assert message.conversation_id == sample_conversation.id
        assert message.content == "Test message content"
        assert message.role == "user"
        assert message.cached is False
        assert message.processing_time_ms == 250
        assert message.timestamp is not None
        assert message.model_used is None

    @pytest.mark.asyncio
    async def test_create_assistant_message(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test creating an assistant message with model info."""
        message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Assistant response",
            role="assistant",
            model_used="gpt-4",
            cached=False,
            processing_time_ms=1800
        )
        
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.role == "assistant"
        assert message.model_used == "gpt-4"
        assert message.content == "Assistant response"
        assert message.processing_time_ms == 1800

    @pytest.mark.asyncio
    async def test_create_cached_message(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test creating a cached message."""
        message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Cached response",
            role="assistant",
            model_used="gpt-3.5-turbo",
            cached=True,
            processing_time_ms=50
        )
        
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.cached is True
        assert message.processing_time_ms == 50
        assert message.model_used == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_message_conversation_relationship(
        self, 
        db_session: AsyncSession,
        sample_message: Message,
        sample_conversation: Conversation
    ):
        """Test message-conversation relationship."""
        # Refresh message to load conversation
        await db_session.refresh(sample_message)
        
        assert sample_message.conversation is not None
        assert sample_message.conversation.id == sample_conversation.id
        assert sample_message.conversation.session_id == sample_conversation.session_id

    @pytest.mark.asyncio
    async def test_message_ordering(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test message ordering by timestamp."""
        # Create messages with slight delays
        message1 = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="First message",
            role="user",
            cached=False,
            processing_time_ms=100
        )
        
        db_session.add(message1)
        await db_session.commit()
        
        # Small delay to ensure different timestamps
        import asyncio
        await asyncio.sleep(0.01)
        
        message2 = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Second message",
            role="assistant",
            cached=False,
            processing_time_ms=200
        )
        
        db_session.add(message2)
        await db_session.commit()
        
        # Query messages ordered by timestamp
        stmt = select(Message).where(
            Message.conversation_id == sample_conversation.id
        ).order_by(Message.timestamp)
        result = await db_session.execute(stmt)
        messages = result.scalars().all()
        
        assert len(messages) >= 2
        assert messages[0].content == "First message"
        assert messages[1].content == "Second message"
        assert messages[0].timestamp <= messages[1].timestamp

    @pytest.mark.asyncio
    async def test_message_role_validation(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test message role validation."""
        # Test valid roles
        valid_roles = ["user", "assistant", "system"]
        
        for role in valid_roles:
            message = Message(
                id=uuid4(),
                conversation_id=sample_conversation.id,
                content=f"Test message for {role}",
                role=role,
                cached=False,
                processing_time_ms=100
            )
            
            db_session.add(message)
            await db_session.commit()
            await db_session.refresh(message)
            
            assert message.role == role

    @pytest.mark.asyncio
    async def test_message_content_handling(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test message content handling."""
        # Test with long content
        long_content = "This is a very long message content. " * 100
        
        message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content=long_content,
            role="user",
            cached=False,
            processing_time_ms=100
        )
        
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.content == long_content

    @pytest.mark.asyncio
    async def test_message_processing_time_tracking(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test message processing time tracking."""
        processing_times = [100, 500, 1000, 2500, 5000]
        
        for i, processing_time in enumerate(processing_times):
            message = Message(
                id=uuid4(),
                conversation_id=sample_conversation.id,
                content=f"Message {i}",
                role="assistant",
                model_used="gpt-3.5-turbo",
                cached=False,
                processing_time_ms=processing_time
            )
            
            db_session.add(message)
            await db_session.commit()
            await db_session.refresh(message)
            
            assert message.processing_time_ms == processing_time

    @pytest.mark.asyncio
    async def test_message_query_by_conversation(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test querying messages by conversation."""
        # Create multiple messages
        messages = []
        for i in range(3):
            message = Message(
                id=uuid4(),
                conversation_id=sample_conversation.id,
                content=f"Message {i}",
                role="user" if i % 2 == 0 else "assistant",
                cached=False,
                processing_time_ms=100 * (i + 1)
            )
            messages.append(message)
            db_session.add(message)
        
        await db_session.commit()
        
        # Query messages for the conversation
        stmt = select(Message).where(
            Message.conversation_id == sample_conversation.id
        )
        result = await db_session.execute(stmt)
        found_messages = result.scalars().all()
        
        assert len(found_messages) >= 3
        message_ids = [m.id for m in found_messages]
        for message in messages:
            assert message.id in message_ids

    @pytest.mark.asyncio
    async def test_message_query_by_role(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test querying messages by role."""
        # Create messages with different roles
        user_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="User message",
            role="user",
            cached=False,
            processing_time_ms=100
        )
        
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Assistant message",
            role="assistant",
            model_used="gpt-3.5-turbo",
            cached=False,
            processing_time_ms=200
        )
        
        system_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="System message",
            role="system",
            cached=False,
            processing_time_ms=50
        )
        
        db_session.add_all([user_message, assistant_message, system_message])
        await db_session.commit()
        
        # Query user messages
        stmt = select(Message).where(
            Message.conversation_id == sample_conversation.id,
            Message.role == "user"
        )
        result = await db_session.execute(stmt)
        user_messages = result.scalars().all()
        
        assert len(user_messages) >= 1
        assert all(m.role == "user" for m in user_messages)
        
        # Query assistant messages
        stmt = select(Message).where(
            Message.conversation_id == sample_conversation.id,
            Message.role == "assistant"
        )
        result = await db_session.execute(stmt)
        assistant_messages = result.scalars().all()
        
        assert len(assistant_messages) >= 1
        assert all(m.role == "assistant" for m in assistant_messages)

    @pytest.mark.asyncio
    async def test_message_query_cached_vs_uncached(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test querying cached vs uncached messages."""
        # Create cached and uncached messages
        cached_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Cached message",
            role="assistant",
            model_used="gpt-3.5-turbo",
            cached=True,
            processing_time_ms=50
        )
        
        uncached_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Uncached message",
            role="assistant",
            model_used="gpt-4",
            cached=False,
            processing_time_ms=1500
        )
        
        db_session.add_all([cached_message, uncached_message])
        await db_session.commit()
        
        # Query cached messages
        stmt = select(Message).where(
            Message.conversation_id == sample_conversation.id,
            Message.cached == True
        )
        result = await db_session.execute(stmt)
        cached_messages = result.scalars().all()
        
        assert len(cached_messages) >= 1
        assert all(m.cached is True for m in cached_messages)
        
        # Query uncached messages
        stmt = select(Message).where(
            Message.conversation_id == sample_conversation.id,
            Message.cached == False
        )
        result = await db_session.execute(stmt)
        uncached_messages = result.scalars().all()
        
        assert len(uncached_messages) >= 1
        assert all(m.cached is False for m in uncached_messages)

    @pytest.mark.asyncio
    async def test_message_timestamp_auto_creation(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test that message timestamp is automatically created."""
        start_time = datetime.utcnow()
        
        message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            content="Timestamp test message",
            role="user",
            cached=False,
            processing_time_ms=100
        )
        
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.timestamp is not None
        assert message.timestamp >= start_time
        assert message.timestamp <= datetime.utcnow()


class TestConversationMessageIntegration:
    """Integration tests for Conversation and Message models."""

    @pytest.mark.asyncio
    async def test_conversation_message_cascade(
        self, 
        db_session: AsyncSession,
        sample_user: User
    ):
        """Test cascade behavior between conversation and messages."""
        # Create conversation
        conversation = Conversation(
            id=uuid4(),
            session_id="cascade_test_session",
            user_identifier=sample_user.email,
            user_id=sample_user.id,
            resolved=False,
            resolution_attempts=0,
            authenticated=False
        )
        
        db_session.add(conversation)
        await db_session.commit()
        
        # Create multiple messages
        messages = []
        for i in range(5):
            message = Message(
                id=uuid4(),
                conversation_id=conversation.id,
                content=f"Message {i}",
                role="user" if i % 2 == 0 else "assistant",
                cached=False,
                processing_time_ms=100 * (i + 1)
            )
            messages.append(message)
            db_session.add(message)
        
        await db_session.commit()
        
        # Verify messages exist
        stmt = select(Message).where(Message.conversation_id == conversation.id)
        result = await db_session.execute(stmt)
        found_messages = result.scalars().all()
        assert len(found_messages) == 5
        
        # Delete conversation
        await db_session.delete(conversation)
        await db_session.commit()
        
        # Verify messages are also deleted
        stmt = select(Message).where(Message.conversation_id == conversation.id)
        result = await db_session.execute(stmt)
        found_messages = result.scalars().all()
        assert len(found_messages) == 0

    @pytest.mark.asyncio
    async def test_conversation_metrics_calculation(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test calculation of conversation metrics."""
        # Create messages with different processing times
        processing_times = [100, 500, 1000, 250, 750]
        
        for i, processing_time in enumerate(processing_times):
            message = Message(
                id=uuid4(),
                conversation_id=sample_conversation.id,
                content=f"Message {i}",
                role="assistant",
                model_used="gpt-3.5-turbo",
                cached=i % 2 == 0,  # Alternate cached/uncached
                processing_time_ms=processing_time
            )
            db_session.add(message)
        
        await db_session.commit()
        
        # Calculate metrics
        stmt = select(Message).where(Message.conversation_id == sample_conversation.id)
        result = await db_session.execute(stmt)
        messages = result.scalars().all()
        
        total_processing_time = sum(m.processing_time_ms for m in messages)
        avg_processing_time = total_processing_time / len(messages)
        cached_count = sum(1 for m in messages if m.cached)
        uncached_count = len(messages) - cached_count
        
        assert total_processing_time == sum(processing_times)
        assert avg_processing_time == sum(processing_times) / len(processing_times)
        assert cached_count == 3  # Even indices (0, 2, 4)
        assert uncached_count == 2  # Odd indices (1, 3)

    @pytest.mark.asyncio
    async def test_conversation_timeline(
        self, 
        db_session: AsyncSession,
        sample_conversation: Conversation
    ):
        """Test conversation timeline with message ordering."""
        # Create messages with specific timestamps
        messages_data = [
            ("Hello", "user"),
            ("How can I help?", "assistant"),
            ("I need account info", "user"),
            ("Let me check that", "assistant"),
            ("Here's your info", "assistant")
        ]
        
        messages = []
        for content, role in messages_data:
            message = Message(
                id=uuid4(),
                conversation_id=sample_conversation.id,
                content=content,
                role=role,
                cached=False,
                processing_time_ms=100
            )
            messages.append(message)
            db_session.add(message)
            
            # Small delay to ensure different timestamps
            import asyncio
            await asyncio.sleep(0.01)
        
        await db_session.commit()
        
        # Query messages in chronological order
        stmt = select(Message).where(
            Message.conversation_id == sample_conversation.id
        ).order_by(Message.timestamp)
        result = await db_session.execute(stmt)
        ordered_messages = result.scalars().all()
        
        assert len(ordered_messages) >= 5
        
        # Verify ordering
        for i in range(len(ordered_messages) - 1):
            assert ordered_messages[i].timestamp <= ordered_messages[i + 1].timestamp
