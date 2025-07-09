"""
Tests for chat API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
import json

from app.models.conversation import Conversation, Message
from app.models.user import User
from app.models.auth import AuthToken


class TestChatMessage:
    """Test cases for POST /api/v1/chat/message endpoint."""

    def test_chat_message_success(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test successful chat message processing."""
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "session_id" in data
            assert "conversation_id" in data
            assert data["requires_auth"] is False
            assert data["cached"] is False
            assert data["model_used"] == "gpt-3.5-turbo"

    def test_chat_message_with_cache_hit(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test chat message with cache hit."""
        # Mock cache hit
        mock_redis.get.return_value = json.dumps({
            "response": "Cached response",
            "model_used": "gpt-3.5-turbo",
            "cached": True
        })
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Cached response"
            assert data["cached"] is True

    def test_chat_message_requires_auth(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test chat message that requires authentication."""
        # Mock MCP server requiring auth
        mock_llm_service.generate_response.side_effect = Exception("Auth required")
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis), \
             patch('app.services.auth.AuthService.requires_auth', return_value=True):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["requires_auth"] is True
            assert "auth_methods" in data
            assert "sms" in data["auth_methods"] or "email" in data["auth_methods"]

    def test_chat_message_rate_limited(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_redis: Mock
    ):
        """Test chat message rate limiting."""
        # Mock rate limit exceeded
        mock_redis.incr.return_value = 61  # Over limit of 60
        
        with patch('app.core.cache.redis_client', mock_redis):
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 429
            data = response.json()
            assert "rate limit" in data["detail"].lower()

    def test_chat_message_irrelevant_query(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test chat message with irrelevant query."""
        # Mock irrelevant query
        mock_llm_service.check_relevance.return_value = {
            "is_relevant": False,
            "confidence": 0.9
        }
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "not able to help" in data["response"].lower() or \
                   "outside my scope" in data["response"].lower()

    def test_chat_message_invalid_input(self, client: TestClient):
        """Test chat message with invalid input."""
        invalid_data = {
            "message": "",  # Empty message
            "session_id": "test_session"
        }
        
        response = client.post("/api/v1/chat/message", json=invalid_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_chat_message_missing_message(self, client: TestClient):
        """Test chat message with missing message field."""
        invalid_data = {
            "session_id": "test_session"
        }
        
        response = client.post("/api/v1/chat/message", json=invalid_data)
        
        assert response.status_code == 422

    def test_chat_message_llm_failure(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test chat message when LLM service fails."""
        # Mock LLM service failure
        mock_llm_service.generate_response.side_effect = Exception("LLM service unavailable")
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "trouble" in data["response"].lower() or \
                   "contact support" in data["response"].lower()

    def test_chat_message_with_context(
        self, 
        client: TestClient, 
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test chat message with context information."""
        request_data = {
            "message": "Help me with my order",
            "session_id": "test_session",
            "context": {
                "page_url": "https://example.com/orders",
                "user_agent": "Mozilla/5.0"
            }
        }
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "session_id" in data


class TestChatAuthRequest:
    """Test cases for POST /api/v1/chat/auth/request endpoint."""

    def test_auth_request_sms_success(
        self, 
        client: TestClient, 
        mock_auth_service: Mock,
        mock_sms_service: Mock
    ):
        """Test successful SMS auth request."""
        request_data = {
            "session_id": "test_session_123",
            "contact_method": "sms",
            "contact_value": "+1234567890"
        }
        
        with patch('app.services.auth.AuthService', return_value=mock_auth_service), \
             patch('app.services.sms.SMSService', return_value=mock_sms_service):
            
            response = client.post("/api/v1/chat/auth/request", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "token_sent" in data
            assert data["method"] == "sms"

    def test_auth_request_email_success(
        self, 
        client: TestClient, 
        mock_auth_service: Mock,
        mock_email_service: Mock
    ):
        """Test successful email auth request."""
        request_data = {
            "session_id": "test_session_123",
            "contact_method": "email",
            "contact_value": "test@example.com"
        }
        
        with patch('app.services.auth.AuthService', return_value=mock_auth_service), \
             patch('app.services.email.EmailService', return_value=mock_email_service):
            
            response = client.post("/api/v1/chat/auth/request", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "token_sent" in data
            assert data["method"] == "email"

    def test_auth_request_invalid_method(self, client: TestClient):
        """Test auth request with invalid method."""
        request_data = {
            "session_id": "test_session_123",
            "contact_method": "invalid",
            "contact_value": "test@example.com"
        }
        
        response = client.post("/api/v1/chat/auth/request", json=request_data)
        
        assert response.status_code == 422

    def test_auth_request_invalid_email(self, client: TestClient):
        """Test auth request with invalid email."""
        request_data = {
            "session_id": "test_session_123",
            "contact_method": "email",
            "contact_value": "invalid-email"
        }
        
        response = client.post("/api/v1/chat/auth/request", json=request_data)
        
        assert response.status_code == 422

    def test_auth_request_service_unavailable(
        self, 
        client: TestClient, 
        mock_auth_service: Mock,
        mock_sms_service: Mock
    ):
        """Test auth request when service is unavailable."""
        mock_sms_service.send_token.side_effect = Exception("SMS service unavailable")
        
        request_data = {
            "session_id": "test_session_123",
            "contact_method": "sms",
            "contact_value": "+1234567890"
        }
        
        with patch('app.services.auth.AuthService', return_value=mock_auth_service), \
             patch('app.services.sms.SMSService', return_value=mock_sms_service):
            
            response = client.post("/api/v1/chat/auth/request", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data["detail"].lower()


class TestChatAuthVerify:
    """Test cases for POST /api/v1/chat/auth/verify endpoint."""

    def test_auth_verify_success(
        self, 
        client: TestClient, 
        mock_auth_service: Mock
    ):
        """Test successful token verification."""
        request_data = {
            "session_id": "test_session_123",
            "token": "123456"
        }
        
        with patch('app.services.auth.AuthService', return_value=mock_auth_service):
            response = client.post("/api/v1/chat/auth/verify", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "authenticated" in data
            assert data["session_id"] == "test_session_123"

    def test_auth_verify_invalid_token(
        self, 
        client: TestClient, 
        mock_auth_service: Mock
    ):
        """Test token verification with invalid token."""
        mock_auth_service.verify_token.return_value = False
        
        request_data = {
            "session_id": "test_session_123",
            "token": "invalid_token"
        }
        
        with patch('app.services.auth.AuthService', return_value=mock_auth_service):
            response = client.post("/api/v1/chat/auth/verify", json=request_data)
            
            assert response.status_code == 401
            data = response.json()
            assert "invalid" in data["detail"].lower()

    def test_auth_verify_expired_token(
        self, 
        client: TestClient, 
        mock_auth_service: Mock
    ):
        """Test token verification with expired token."""
        mock_auth_service.verify_token.side_effect = Exception("Token expired")
        
        request_data = {
            "session_id": "test_session_123",
            "token": "123456"
        }
        
        with patch('app.services.auth.AuthService', return_value=mock_auth_service):
            response = client.post("/api/v1/chat/auth/verify", json=request_data)
            
            assert response.status_code == 401
            data = response.json()
            assert "expired" in data["detail"].lower()

    def test_auth_verify_missing_session(self, client: TestClient):
        """Test token verification without session_id."""
        request_data = {
            "token": "123456"
        }
        
        response = client.post("/api/v1/chat/auth/verify", json=request_data)
        
        assert response.status_code == 422

    def test_auth_verify_missing_token(self, client: TestClient):
        """Test token verification without token."""
        request_data = {
            "session_id": "test_session_123"
        }
        
        response = client.post("/api/v1/chat/auth/verify", json=request_data)
        
        assert response.status_code == 422


class TestChatHealthCheck:
    """Test cases for health check endpoint."""

    def test_health_check_success(self, client: TestClient):
        """Test successful health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_check_with_dependencies(
        self, 
        client: TestClient,
        mock_redis: Mock,
        mock_vector_db_service: Mock
    ):
        """Test health check with dependency checks."""
        mock_vector_db_service.health_check.return_value = True
        
        with patch('app.core.cache.redis_client', mock_redis), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service):
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "dependencies" in data


class TestChatRateLimiting:
    """Test cases for rate limiting functionality."""

    def test_rate_limit_per_user(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_redis: Mock
    ):
        """Test per-user rate limiting."""
        # Mock rate limit counter
        mock_redis.incr.return_value = 61  # Over limit
        
        with patch('app.core.cache.redis_client', mock_redis):
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 429

    def test_rate_limit_global(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_redis: Mock
    ):
        """Test global rate limiting."""
        # Mock global rate limit counter
        mock_redis.incr.side_effect = [1, 1001]  # User OK, global over limit
        
        with patch('app.core.cache.redis_client', mock_redis):
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 429

    def test_rate_limit_within_bounds(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_redis: Mock,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock
    ):
        """Test request within rate limits."""
        # Mock rate limit counters within bounds
        mock_redis.incr.side_effect = [1, 1]  # Both under limits
        
        with patch('app.core.cache.redis_client', mock_redis), \
             patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200


class TestChatErrorHandling:
    """Test cases for error handling in chat endpoints."""

    def test_database_error_handling(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test handling of database errors."""
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis), \
             patch('app.models.conversation.Conversation') as mock_conversation:
            
            mock_conversation.side_effect = Exception("Database error")
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            # Should handle gracefully and return fallback response
            assert response.status_code == 200
            data = response.json()
            assert "response" in data

    def test_vector_db_error_handling(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test handling of vector database errors."""
        mock_vector_db_service.search_similar.side_effect = Exception("Vector DB error")
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data

    def test_redis_error_handling(
        self, 
        client: TestClient, 
        chat_request_data: dict,
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test handling of Redis errors."""
        mock_redis.get.side_effect = Exception("Redis error")
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=chat_request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data


class TestChatClarificationLoop:
    """Test cases for clarification loop functionality."""

    def test_clarification_loop_success(
        self, 
        client: TestClient, 
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test successful clarification loop."""
        # Mock unclear query requiring clarification
        mock_llm_service.check_relevance.side_effect = [
            {"is_relevant": False, "confidence": 0.3},  # First attempt
            {"is_relevant": True, "confidence": 0.9}    # After clarification
        ]
        
        request_data = {
            "message": "I need help",
            "session_id": "test_session"
        }
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data

    def test_clarification_loop_max_attempts(
        self, 
        client: TestClient, 
        mock_llm_service: Mock,
        mock_vector_db_service: Mock,
        mock_redis: Mock
    ):
        """Test clarification loop reaching max attempts."""
        # Mock consistently unclear query
        mock_llm_service.check_relevance.return_value = {
            "is_relevant": False, 
            "confidence": 0.3
        }
        
        request_data = {
            "message": "unclear query",
            "session_id": "test_session"
        }
        
        with patch('app.services.llm.LLMService', return_value=mock_llm_service), \
             patch('app.services.vector_db.VectorDBService', return_value=mock_vector_db_service), \
             patch('app.core.cache.redis_client', mock_redis):
            
            response = client.post("/api/v1/chat/message", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            # Should provide fallback response after max attempts
