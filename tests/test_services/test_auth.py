"""
Tests for authentication service.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.services.auth import AuthService
from app.models.user import User
from app.models.auth import AuthToken
from app.models.conversation import Conversation
from app.core.exceptions import AuthenticationError, ValidationError
from app.services.sms import SMSService
from app.services.email import EmailService


class TestAuthService:
    """Test cases for AuthService."""

    @pytest.fixture
    def auth_service(self, db_session: AsyncSession, mock_redis: Mock):
        """Create AuthService instance for testing."""
        with patch('app.core.cache.redis_client', mock_redis):
            return AuthService(db_session=db_session)

    @pytest.mark.asyncio
    async def test_generate_token_success(
        self, 
        auth_service: AuthService, 
        sample_user: User
    ):
        """Test successful token generation."""
        token = await auth_service.generate_token(
            user_id=sample_user.id,
            session_id="test_session_123"
        )
        
        assert len(token) == 6
        assert token.isdigit()

    @pytest.mark.asyncio
    async def test_generate_token_with_expiry(
        self, 
        auth_service: AuthService, 
        sample_user: User
    ):
        """Test token generation with custom expiry."""
        expiry_minutes = 10
        token = await auth_service.generate_token(
            user_id=sample_user.id,
            session_id="test_session_123",
            expiry_minutes=expiry_minutes
        )
        
        assert len(token) == 6
        assert token.isdigit()

    @pytest.mark.asyncio
    async def test_verify_token_success(
        self, 
        auth_service: AuthService, 
        sample_auth_token: AuthToken
    ):
        """Test successful token verification."""
        result = await auth_service.verify_token(
            token=sample_auth_token.token,
            session_id=sample_auth_token.session_id
        )
        
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_token_invalid(
        self, 
        auth_service: AuthService
    ):
        """Test token verification with invalid token."""
        result = await auth_service.verify_token(
            token="invalid_token",
            session_id="test_session_123"
        )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_token_expired(
        self, 
        auth_service: AuthService, 
        db_session: AsyncSession,
        sample_user: User
    ):
        """Test token verification with expired token."""
        # Create expired token
        expired_token = AuthToken(
            id=uuid4(),
            user_id=sample_user.id,
            token="123456",
            session_id="test_session_123",
            used=False,
            expires_at=datetime.utcnow() - timedelta(minutes=1)  # Expired
        )
        db_session.add(expired_token)
        await db_session.commit()
        
        with pytest.raises(AuthenticationError, match="Token expired"):
            await auth_service.verify_token(
                token=expired_token.token,
                session_id=expired_token.session_id
            )

    @pytest.mark.asyncio
    async def test_verify_token_already_used(
        self, 
        auth_service: AuthService, 
        db_session: AsyncSession,
        sample_user: User
    ):
        """Test token verification with already used token."""
        # Create used token
        used_token = AuthToken(
            id=uuid4(),
            user_id=sample_user.id,
            token="123456",
            session_id="test_session_123",
            used=True,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        db_session.add(used_token)
        await db_session.commit()
        
        result = await auth_service.verify_token(
            token=used_token.token,
            session_id=used_token.session_id
        )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_create_session_success(
        self, 
        auth_service: AuthService, 
        sample_user: User
    ):
        """Test successful session creation."""
        session_id = await auth_service.create_session(
            user_id=sample_user.id,
            timeout_minutes=30
        )
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_validate_session_success(
        self, 
        auth_service: AuthService, 
        sample_user: User,
        mock_redis: Mock
    ):
        """Test successful session validation."""
        session_id = "test_session_123"
        
        # Mock Redis session data
        mock_redis.get.return_value = str(sample_user.id)
        
        result = await auth_service.validate_session(session_id)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_session_invalid(
        self, 
        auth_service: AuthService,
        mock_redis: Mock
    ):
        """Test session validation with invalid session."""
        session_id = "invalid_session"
        
        # Mock Redis returning None (no session)
        mock_redis.get.return_value = None
        
        result = await auth_service.validate_session(session_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_session_success(
        self, 
        auth_service: AuthService, 
        sample_user: User,
        mock_redis: Mock
    ):
        """Test successful session extension."""
        session_id = "test_session_123"
        
        # Mock Redis session exists
        mock_redis.get.return_value = str(sample_user.id)
        mock_redis.expire.return_value = True
        
        result = await auth_service.extend_session(
            session_id=session_id,
            timeout_minutes=60
        )
        
        assert result is True
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_session_success(
        self, 
        auth_service: AuthService,
        mock_redis: Mock
    ):
        """Test successful session invalidation."""
        session_id = "test_session_123"
        
        mock_redis.delete.return_value = 1
        
        result = await auth_service.invalidate_session(session_id)
        
        assert result is True
        mock_redis.delete.assert_called_once_with(f"session:{session_id}")

    @pytest.mark.asyncio
    async def test_get_user_by_contact_email(
        self, 
        auth_service: AuthService, 
        sample_user: User
    ):
        """Test getting user by email contact."""
        user = await auth_service.get_user_by_contact(
            contact_method="email",
            contact_value=sample_user.email
        )
        
        assert user is not None
        assert user.email == sample_user.email

    @pytest.mark.asyncio
    async def test_get_user_by_contact_mobile(
        self, 
        auth_service: AuthService, 
        sample_user: User
    ):
        """Test getting user by mobile contact."""
        user = await auth_service.get_user_by_contact(
            contact_method="sms",
            contact_value=sample_user.mobile_number
        )
        
        assert user is not None
        assert user.mobile_number == sample_user.mobile_number

    @pytest.mark.asyncio
    async def test_get_user_by_contact_not_found(
        self, 
        auth_service: AuthService
    ):
        """Test getting user by contact when user doesn't exist."""
        user = await auth_service.get_user_by_contact(
            contact_method="email",
            contact_value="nonexistent@example.com"
        )
        
        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_by_contact_invalid_method(
        self, 
        auth_service: AuthService
    ):
        """Test getting user by invalid contact method."""
        with pytest.raises(ValidationError, match="Invalid contact method"):
            await auth_service.get_user_by_contact(
                contact_method="invalid",
                contact_value="test@example.com"
            )

    @pytest.mark.asyncio
    async def test_create_user_success(
        self, 
        auth_service: AuthService
    ):
        """Test successful user creation."""
        user = await auth_service.create_user(
            email="newuser@example.com",
            mobile_number="+1987654321"
        )
        
        assert user is not None
        assert user.email == "newuser@example.com"
        assert user.mobile_number == "+1987654321"

    @pytest.mark.asyncio
    async def test_create_user_email_only(
        self, 
        auth_service: AuthService
    ):
        """Test user creation with email only."""
        user = await auth_service.create_user(
            email="emailonly@example.com"
        )
        
        assert user is not None
        assert user.email == "emailonly@example.com"
        assert user.mobile_number is None

    @pytest.mark.asyncio
    async def test_create_user_mobile_only(
        self, 
        auth_service: AuthService
    ):
        """Test user creation with mobile only."""
        user = await auth_service.create_user(
            mobile_number="+1555666777"
        )
        
        assert user is not None
        assert user.mobile_number == "+1555666777"
        assert user.email is None

    @pytest.mark.asyncio
    async def test_create_user_no_contact_info(
        self, 
        auth_service: AuthService
    ):
        """Test user creation without contact info."""
        with pytest.raises(ValidationError, match="Either email or mobile_number must be provided"):
            await auth_service.create_user()

    @pytest.mark.asyncio
    async def test_requires_auth_true(
        self, 
        auth_service: AuthService
    ):
        """Test requires_auth returns True for auth-required scenarios."""
        # Mock scenario where MCP server requires auth
        with patch('app.services.mcp.MCPService.requires_auth', return_value=True):
            result = await auth_service.requires_auth(
                query="Check my account balance",
                session_id="test_session_123"
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_requires_auth_false(
        self, 
        auth_service: AuthService
    ):
        """Test requires_auth returns False for non-auth scenarios."""
        # Mock scenario where no auth is required
        with patch('app.services.mcp.MCPService.requires_auth', return_value=False):
            result = await auth_service.requires_auth(
                query="What are your business hours?",
                session_id="test_session_123"
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_get_available_auth_methods_both(
        self, 
        auth_service: AuthService, 
        sample_user: User
    ):
        """Test getting available auth methods when both are available."""
        methods = await auth_service.get_available_auth_methods(sample_user)
        
        assert "email" in methods
        assert "sms" in methods
        assert len(methods) == 2

    @pytest.mark.asyncio
    async def test_get_available_auth_methods_email_only(
        self, 
        auth_service: AuthService, 
        db_session: AsyncSession
    ):
        """Test getting available auth methods for email-only user."""
        email_only_user = User(
            id=uuid4(),
            email="emailonly@example.com",
            mobile_number=None
        )
        db_session.add(email_only_user)
        await db_session.commit()
        
        methods = await auth_service.get_available_auth_methods(email_only_user)
        
        assert "email" in methods
        assert "sms" not in methods
        assert len(methods) == 1

    @pytest.mark.asyncio
    async def test_get_available_auth_methods_mobile_only(
        self, 
        auth_service: AuthService, 
        db_session: AsyncSession
    ):
        """Test getting available auth methods for mobile-only user."""
        mobile_only_user = User(
            id=uuid4(),
            email=None,
            mobile_number="+1555666777"
        )
        db_session.add(mobile_only_user)
        await db_session.commit()
        
        methods = await auth_service.get_available_auth_methods(mobile_only_user)
        
        assert "sms" in methods
        assert "email" not in methods
        assert len(methods) == 1

    @pytest.mark.asyncio
    async def test_request_auth_token_sms_success(
        self, 
        auth_service: AuthService, 
        sample_user: User,
        mock_sms_service: Mock
    ):
        """Test successful SMS auth token request."""
        with patch('app.services.sms.SMSService', return_value=mock_sms_service):
            result = await auth_service.request_auth_token(
                user=sample_user,
                contact_method="sms",
                session_id="test_session_123"
            )
            
            assert result is True
            mock_sms_service.send_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_auth_token_email_success(
        self, 
        auth_service: AuthService, 
        sample_user: User,
        mock_email_service: Mock
    ):
        """Test successful email auth token request."""
        with patch('app.services.email.EmailService', return_value=mock_email_service):
            result = await auth_service.request_auth_token(
                user=sample_user,
                contact_method="email",
                session_id="test_session_123"
            )
            
            assert result is True
            mock_email_service.send_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_auth_token_service_failure(
        self, 
        auth_service: AuthService, 
        sample_user: User,
        mock_sms_service: Mock
    ):
        """Test auth token request when service fails."""
        mock_sms_service.send_token.side_effect = Exception("SMS service error")
        
        with patch('app.services.sms.SMSService', return_value=mock_sms_service):
            result = await auth_service.request_auth_token(
                user=sample_user,
                contact_method="sms",
                session_id="test_session_123"
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_request_auth_token_invalid_method(
        self, 
        auth_service: AuthService, 
        sample_user: User
    ):
        """Test auth token request with invalid method."""
        with pytest.raises(ValidationError, match="Invalid contact method"):
            await auth_service.request_auth_token(
                user=sample_user,
                contact_method="invalid",
                session_id="test_session_123"
            )

    @pytest.mark.asyncio
    async def test_cleanup_expired_tokens(
        self, 
        auth_service: AuthService, 
        db_session: AsyncSession,
        sample_user: User
    ):
        """Test cleanup of expired tokens."""
        # Create expired token
        expired_token = AuthToken(
            id=uuid4(),
            user_id=sample_user.id,
            token="expired_token",
            session_id="test_session_123",
            used=False,
            expires_at=datetime.utcnow() - timedelta(minutes=10)
        )
        db_session.add(expired_token)
        await db_session.commit()
        
        # Run cleanup
        cleaned_count = await auth_service.cleanup_expired_tokens()
        
        assert cleaned_count >= 1

    @pytest.mark.asyncio
    async def test_get_session_user_id(
        self, 
        auth_service: AuthService, 
        sample_user: User,
        mock_redis: Mock
    ):
        """Test getting user ID from session."""
        session_id = "test_session_123"
        
        # Mock Redis session data
        mock_redis.get.return_value = str(sample_user.id)
        
        user_id = await auth_service.get_session_user_id(session_id)
        
        assert user_id == sample_user.id

    @pytest.mark.asyncio
    async def test_get_session_user_id_invalid_session(
        self, 
        auth_service: AuthService,
        mock_redis: Mock
    ):
        """Test getting user ID from invalid session."""
        session_id = "invalid_session"
        
        # Mock Redis returning None
        mock_redis.get.return_value = None
        
        user_id = await auth_service.get_session_user_id(session_id)
        
        assert user_id is None

    @pytest.mark.asyncio
    async def test_is_session_authenticated(
        self, 
        auth_service: AuthService, 
        sample_user: User,
        mock_redis: Mock
    ):
        """Test checking if session is authenticated."""
        session_id = "test_session_123"
        
        # Mock authenticated session
        mock_redis.get.return_value = str(sample_user.id)
        
        result = await auth_service.is_session_authenticated(session_id)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_is_session_authenticated_false(
        self, 
        auth_service: AuthService,
        mock_redis: Mock
    ):
        """Test checking if session is authenticated when not authenticated."""
        session_id = "test_session_123"
        
        # Mock unauthenticated session
        mock_redis.get.return_value = None
        
        result = await auth_service.is_session_authenticated(session_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_mark_token_as_used(
        self, 
        auth_service: AuthService, 
        sample_auth_token: AuthToken,
        db_session: AsyncSession
    ):
        """Test marking token as used."""
        await auth_service.mark_token_as_used(sample_auth_token.id)
        
        # Refresh token from database
        await db_session.refresh(sample_auth_token)
        
        assert sample_auth_token.used is True

    @pytest.mark.asyncio
    async def test_generate_secure_token(
        self, 
        auth_service: AuthService
    ):
        """Test secure token generation."""
        token = auth_service._generate_secure_token(length=8)
        
        assert len(token) == 8
        assert token.isdigit()

    @pytest.mark.asyncio
    async def test_generate_session_id(
        self, 
        auth_service: AuthService
    ):
        """Test session ID generation."""
        session_id = auth_service._generate_session_id()
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id.startswith("session_")

    @pytest.mark.asyncio
    async def test_validate_contact_method_valid(
        self, 
        auth_service: AuthService
    ):
        """Test contact method validation with valid methods."""
        # Should not raise exception
        auth_service._validate_contact_method("email")
        auth_service._validate_contact_method("sms")

    @pytest.mark.asyncio
    async def test_validate_contact_method_invalid(
        self, 
        auth_service: AuthService
    ):
        """Test contact method validation with invalid method."""
        with pytest.raises(ValidationError, match="Invalid contact method"):
            auth_service._validate_contact_method("invalid")


class TestAuthServiceIntegration:
    """Integration tests for AuthService."""

    @pytest.mark.asyncio
    async def test_full_auth_flow_sms(
        self, 
        db_session: AsyncSession,
        mock_redis: Mock,
        mock_sms_service: Mock
    ):
        """Test complete SMS authentication flow."""
        auth_service = AuthService(db_session=db_session)
        
        # Step 1: Create user
        user = await auth_service.create_user(
            email="test@example.com",
            mobile_number="+1234567890"
        )
        
        # Step 2: Request auth token
        with patch('app.services.sms.SMSService', return_value=mock_sms_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            result = await auth_service.request_auth_token(
                user=user,
                contact_method="sms",
                session_id="test_session_123"
            )
            
            assert result is True
        
        # Step 3: Get the generated token from database
        from sqlalchemy import select
        stmt = select(AuthToken).where(AuthToken.user_id == user.id)
        result = await db_session.execute(stmt)
        token_record = result.scalar_one()
        
        # Step 4: Verify token
        with patch('app.core.cache.redis_client', mock_redis):
            verified = await auth_service.verify_token(
                token=token_record.token,
                session_id="test_session_123"
            )
            
            assert verified is True

    @pytest.mark.asyncio
    async def test_full_auth_flow_email(
        self, 
        db_session: AsyncSession,
        mock_redis: Mock,
        mock_email_service: Mock
    ):
        """Test complete email authentication flow."""
        auth_service = AuthService(db_session=db_session)
        
        # Step 1: Create user
        user = await auth_service.create_user(
            email="test@example.com",
            mobile_number="+1234567890"
        )
        
        # Step 2: Request auth token
        with patch('app.services.email.EmailService', return_value=mock_email_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            result = await auth_service.request_auth_token(
                user=user,
                contact_method="email",
                session_id="test_session_123"
            )
            
            assert result is True
        
        # Step 3: Get the generated token from database
        from sqlalchemy import select
        stmt = select(AuthToken).where(AuthToken.user_id == user.id)
        result = await db_session.execute(stmt)
        token_record = result.scalar_one()
        
        # Step 4: Verify token
        with patch('app.core.cache.redis_client', mock_redis):
            verified = await auth_service.verify_token(
                token=token_record.token,
                session_id="test_session_123"
            )
            
            assert verified is True

    @pytest.mark.asyncio
    async def test_session_management_flow(
        self, 
        db_session: AsyncSession,
        mock_redis: Mock
    ):
        """Test complete session management flow."""
        auth_service = AuthService(db_session=db_session)
        
        # Step 1: Create user
        user = await auth_service.create_user(
            email="test@example.com"
        )
        
        with patch('app.core.cache.redis_client', mock_redis):
            # Step 2: Create session
            session_id = await auth_service.create_session(
                user_id=user.id,
                timeout_minutes=30
            )
            
            # Step 3: Validate session
            is_valid = await auth_service.validate_session(session_id)
            assert is_valid is True
            
            # Step 4: Extend session
            extended = await auth_service.extend_session(
                session_id=session_id,
                timeout_minutes=60
            )
            assert extended is True
            
            # Step 5: Invalidate session
            invalidated = await auth_service.invalidate_session(session_id)
            assert invalidated is True
            
            # Step 6: Validate session after invalidation
            is_valid_after = await auth_service.validate_session(session_id)
            assert is_valid_after is False
