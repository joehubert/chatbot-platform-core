"""
Test configuration and fixtures for the chatbot platform core.
"""
import asyncio
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import redis
from unittest.mock import Mock, AsyncMock
import tempfile
import os
from typing import AsyncGenerator, Generator
from uuid import uuid4

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings
from app.models.conversation import Conversation, Message
from app.models.document import Document
from app.models.user import User
from app.models.auth import AuthToken
from app.services.llm import LLMService
from app.services.vector_db import VectorDBService
from app.services.auth import AuthService
from app.services.sms import SMSService
from app.services.email import EmailService

# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def client(db_session: AsyncSession) -> Generator[TestClient, None, None]:
    """Create a test client with database session override."""
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = Mock()
    mock_redis.get = Mock(return_value=None)
    mock_redis.set = Mock(return_value=True)
    mock_redis.delete = Mock(return_value=1)
    mock_redis.exists = Mock(return_value=False)
    mock_redis.incr = Mock(return_value=1)
    mock_redis.expire = Mock(return_value=True)
    mock_redis.ttl = Mock(return_value=-1)
    return mock_redis


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    mock_service = Mock(spec=LLMService)
    mock_service.generate_response = AsyncMock(return_value={
        "response": "Test response",
        "model_used": "gpt-3.5-turbo",
        "tokens_used": 100,
        "cost": 0.002
    })
    mock_service.check_relevance = AsyncMock(return_value={
        "is_relevant": True,
        "confidence": 0.95
    })
    mock_service.classify_query_complexity = AsyncMock(return_value={
        "complexity": "simple",
        "confidence": 0.8
    })
    return mock_service


@pytest.fixture
def mock_vector_db_service():
    """Mock Vector DB service."""
    mock_service = Mock(spec=VectorDBService)
    mock_service.search_similar = AsyncMock(return_value=[
        {
            "id": "doc1",
            "content": "Test document content",
            "metadata": {"source": "test.pdf"},
            "score": 0.95
        }
    ])
    mock_service.store_document = AsyncMock(return_value="doc_id_123")
    mock_service.delete_document = AsyncMock(return_value=True)
    mock_service.health_check = AsyncMock(return_value=True)
    return mock_service


@pytest.fixture
def mock_auth_service():
    """Mock Auth service."""
    mock_service = Mock(spec=AuthService)
    mock_service.generate_token = AsyncMock(return_value="123456")
    mock_service.verify_token = AsyncMock(return_value=True)
    mock_service.create_session = AsyncMock(return_value="session_123")
    mock_service.validate_session = AsyncMock(return_value=True)
    return mock_service


@pytest.fixture
def mock_sms_service():
    """Mock SMS service."""
    mock_service = Mock(spec=SMSService)
    mock_service.send_token = AsyncMock(return_value=True)
    mock_service.is_available = Mock(return_value=True)
    return mock_service


@pytest.fixture
def mock_email_service():
    """Mock Email service."""
    mock_service = Mock(spec=EmailService)
    mock_service.send_token = AsyncMock(return_value=True)
    mock_service.is_available = Mock(return_value=True)
    return mock_service


@pytest_asyncio.fixture
async def sample_user(db_session: AsyncSession) -> User:
    """Create a sample user for testing."""
    user = User(
        id=uuid4(),
        email="test@example.com",
        mobile_number="+1234567890"
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def sample_conversation(db_session: AsyncSession, sample_user: User) -> Conversation:
    """Create a sample conversation for testing."""
    conversation = Conversation(
        id=uuid4(),
        session_id="test_session_123",
        user_identifier=sample_user.email,
        user_id=sample_user.id,
        resolved=False,
        resolution_attempts=0,
        authenticated=False
    )
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)
    return conversation


@pytest_asyncio.fixture
async def sample_message(db_session: AsyncSession, sample_conversation: Conversation) -> Message:
    """Create a sample message for testing."""
    message = Message(
        id=uuid4(),
        conversation_id=sample_conversation.id,
        content="Test message content",
        role="user",
        cached=False,
        processing_time_ms=500
    )
    db_session.add(message)
    await db_session.commit()
    await db_session.refresh(message)
    return message


@pytest_asyncio.fixture
async def sample_document(db_session: AsyncSession) -> Document:
    """Create a sample document for testing."""
    document = Document(
        id=uuid4(),
        filename="test_document.pdf",
        content_type="application/pdf",
        processed=True,
        chunk_count=5,
        vector_ids=["vec1", "vec2", "vec3"]
    )
    db_session.add(document)
    await db_session.commit()
    await db_session.refresh(document)
    return document


@pytest_asyncio.fixture
async def sample_auth_token(db_session: AsyncSession, sample_user: User) -> AuthToken:
    """Create a sample auth token for testing."""
    token = AuthToken(
        id=uuid4(),
        user_id=sample_user.id,
        token="123456",
        session_id="test_session_123",
        used=False
    )
    db_session.add(token)
    await db_session.commit()
    await db_session.refresh(token)
    return token


@pytest.fixture
def temp_file():
    """Create a temporary file for testing file uploads."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content for file upload testing.")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_file_upload():
    """Mock file upload for testing."""
    class MockFile:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self.content = content
            self.content_type = "application/pdf"
        
        async def read(self):
            return self.content
    
    return MockFile("test.pdf", b"Mock PDF content")


@pytest.fixture
def chat_request_data():
    """Sample chat request data."""
    return {
        "message": "Hello, I need help with my account",
        "session_id": "test_session_123",
        "user_id": None,
        "context": {
            "page_url": "https://example.com/support",
            "user_agent": "Mozilla/5.0 Test Browser"
        }
    }


@pytest.fixture
def auth_request_data():
    """Sample auth request data."""
    return {
        "session_id": "test_session_123",
        "contact_method": "email",
        "contact_value": "test@example.com"
    }


@pytest.fixture
def document_upload_data():
    """Sample document upload data."""
    return {
        "metadata": {
            "category": "support",
            "expiration_date": "2024-12-31T23:59:59Z"
        }
    }


# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    test_settings = {
        "TESTING": True,
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": "redis://localhost:6379/1",
        "RATE_LIMIT_PER_USER_PER_MINUTE": 60,
        "RATE_LIMIT_GLOBAL_PER_MINUTE": 1000,
        "CACHE_SIMILARITY_THRESHOLD": 0.85,
        "CACHE_TTL_HOURS": 24,
        "AUTH_SESSION_TIMEOUT_MINUTES": 30,
        "OTP_EXPIRY_MINUTES": 5,
        "MAX_FILE_SIZE_MB": 50,
        "ALLOWED_FILE_TYPES": "pdf,txt,docx,md",
        "RELEVANCE_MODEL": "gpt-3.5-turbo",
        "SIMPLE_QUERY_MODEL": "gpt-3.5-turbo",
        "COMPLEX_QUERY_MODEL": "gpt-4",
        "CLARIFICATION_MODEL": "gpt-3.5-turbo",
        "FALLBACK_ERROR_MESSAGE": "I'm having trouble right now. Please contact support.",
        "VECTOR_DB_TYPE": "pinecone",
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "PINECONE_API_KEY": "test_key",
        "PINECONE_ENVIRONMENT": "test"
    }
    
    for key, value in test_settings.items():
        monkeypatch.setenv(key, str(value))


class AsyncMockContext:
    """Helper class for async context manager mocking."""
    def __init__(self, return_value):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_async_context():
    """Factory for creating async context manager mocks."""
    return AsyncMockContext
