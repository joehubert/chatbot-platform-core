"""
Pipeline Nodes Implementation

This module contains the individual processing nodes used in the LangGraph pipeline.
Each node represents a specific step in the chatbot request processing flow.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import uuid4

from app.core.pipeline import PipelineState
from app.services.rate_limiting import RateLimitingService
from app.services.relevance_checker import RelevanceChecker
from app.services.cache import SemanticCacheService
from app.services.model_router import ModelRouter
from app.services.auth_service import AuthService
from app.services.knowledge_base import KnowledgeBaseService
from app.services.mcp_registry import MCPRegistry
from app.services.response_validator import ResponseValidator
from app.core.database import get_db
from app.models.conversation import Conversation
from app.models.message import Message

logger = logging.getLogger(__name__)


class RateLimitingNode:
    """Node for handling rate limiting checks."""
    
    def __init__(self, rate_limiting_service: RateLimitingService):
        self.rate_limiting_service = rate_limiting_service
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process rate limiting for the request."""
        try:
            # Check rate limits
            is_allowed = await self.rate_limiting_service.is_allowed(
                user_id=state.user_id,
                session_id=state.session_id,
                context=state.context
            )
            
            if not is_allowed:
                state.error_message = "Rate limit exceeded. Please try again later."
                logger.warning(f"Rate limit exceeded for session {state.session_id}")
                return state
            
            # Record the request
            await self.rate_limiting_service.record_request(
                user_id=state.user_id,
                session_id=state.session_id,
                context=state.context
            )
            
            logger.debug(f"Rate limiting passed for session {state.session_id}")
            return state
            
        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            state.error_message = "Rate limiting service unavailable"
            return state


class RelevanceCheckNode:
    """Node for checking query relevance and handling clarification."""
    
    def __init__(self, relevance_checker: RelevanceChecker):
        self.relevance_checker = relevance_checker
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process relevance checking for the query."""
        try:
            # Check relevance
            result = await self.relevance_checker.check_relevance(
                query=state.message,
                context=state.context,
                conversation_history=await self._get_conversation_history(state.session_id)
            )
            
            state.is_relevant = result["is_relevant"]
            state.relevance_confidence = result["confidence"]
            
            if not state.is_relevant:
                # Check if we need clarification
                if state.clarification_attempts < state.max_clarification_attempts:
                    state.clarification_attempts += 1
                    
                    # Generate clarification request
                    clarification_response = await self.relevance_checker.generate_clarification(
                        query=state.message,
                        context=state.context,
                        attempt_number=state.clarification_attempts
                    )
                    
                    state.response = clarification_response
                    logger.info(f"Clarification request #{state.clarification_attempts} for session {state.session_id}")
                    
                else:
                    # Max attempts reached, provide out-of-scope response
                    state.response = await self.relevance_checker.get_out_of_scope_response(
                        query=state.message,
                        context=state.context
                    )
                    logger.info(f"Out of scope response for session {state.session_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Relevance check error: {str(e)}")
            state.error_message = "Relevance checking service unavailable"
            return state
    
    async def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recent conversation history for context."""
        try:
            db = next(get_db())
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if not conversation:
                return []
            
            messages = db.query(Message).filter(
                Message.conversation_id == conversation.id
            ).order_by(Message.timestamp.desc()).limit(10).all()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in reversed(messages)
            ]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []


class SemanticCacheNode:
    """Node for checking semantic cache for similar queries."""
    
    def __init__(self, semantic_cache: SemanticCacheService):
        self.semantic_cache = semantic_cache
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process semantic cache check."""
        try:
            # Check cache for similar queries
            cache_result = await self.semantic_cache.get_similar_response(
                query=state.message,
                context=state.context,
                session_id=state.session_id
            )
            
            if cache_result:
                state.cache_hit = True
                state.cached_response = cache_result["response"]
                state.cache_similarity = cache_result["similarity"]
                state.response = cache_result["response"]
                
                logger.info(f"Cache hit for session {state.session_id}, similarity: {state.cache_similarity}")
            else:
                state.cache_hit = False
                logger.debug(f"Cache miss for session {state.session_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Semantic cache error: {str(e)}")
            state.cache_hit = False
            return state


class ModelRoutingNode:
    """Node for routing requests to appropriate models."""
    
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process model routing decision."""
        try:
            # Analyze query complexity and determine model
            routing_result = await self.model_router.route_query(
                query=state.message,
                context=state.context,
                conversation_history=await self._get_conversation_history(state.session_id)
            )
            
            state.selected_model = routing_result["model"]
            state.query_complexity = routing_result["complexity"]
            state.routing_reasoning = routing_result["reasoning"]
            
            logger.info(f"Model routing for session {state.session_id}: {state.selected_model} (complexity: {state.query_complexity})")
            
            return state
            
        except Exception as e:
            logger.error(f"Model routing error: {str(e)}")
            state.error_message = "Model routing service unavailable"
            return state
    
    async def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recent conversation history for context."""
        try:
            db = next(get_db())
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if not conversation:
                return []
            
            messages = db.query(Message).filter(
                Message.conversation_id == conversation.id
            ).order_by(Message.timestamp.desc()).limit(5).all()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in reversed(messages)
            ]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []


class AuthenticationNode:
    """Node for handling authentication requirements."""
    
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process authentication requirements."""
        try:
            # Check if authentication is required for this query
            auth_requirement = await self.auth_service.check_auth_requirement(
                query=state.message,
                context=state.context,
                selected_model=state.selected_model
            )
            
            state.requires_auth = auth_requirement["required"]
            state.auth_methods = auth_requirement["methods"]
            
            if state.requires_auth:
                # Check if already authenticated
                auth_status = await self.auth_service.check_auth_status(
                    session_id=state.session_id
                )
                
                state.is_authenticated = auth_status["authenticated"]
                state.auth_session_id = auth_status.get("auth_session_id")
                
                if not state.is_authenticated:
                    state.response = "This request requires authentication. Please verify your identity."
                    logger.info(f"Authentication required for session {state.session_id}")
                else:
                    logger.info(f"Authentication verified for session {state.session_id}")
            else:
                state.is_authenticated = True
                logger.debug(f"No authentication required for session {state.session_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            state.error_message = "Authentication service unavailable"
            return state


class QuestionProcessingNode:
    """Node for processing the actual question using RAG and MCP tools."""
    
    def __init__(self, knowledge_base: KnowledgeBaseService, mcp_registry: MCPRegistry):
        self.knowledge_base = knowledge_base
        self.mcp_registry = mcp_registry
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process the question using available resources."""
        try:
            state.resolution_attempts += 1
            
            # Get relevant context from knowledge base
            rag_context = await self.knowledge_base.get_relevant_context(
                query=state.message,
                context=state.context,
                max_results=5
            )
            
            state.rag_sources = rag_context["sources"]
            
            # Check if MCP tools are needed
            mcp_tools = await self.mcp_registry.get_relevant_tools(
                query=state.message,
                context=state.context
            )
            
            # Execute MCP tools if available
            tool_results = {}
            if mcp_tools:
                for tool in mcp_tools:
                    try:
                        result = await self.mcp_registry.execute_tool(
                            tool_name=tool["name"],
                            parameters=tool["parameters"],
                            session_id=state.session_id
                        )
                        tool_results[tool["name"]] = result
                        state.mcp_tools_used.append(tool["name"])
                    except Exception as e:
                        logger.error(f"MCP tool execution error: {str(e)}")
            
            # Generate response using selected model
            from app.services.llm_factory import LLMFactory
            llm_factory = LLMFactory()
            llm_client = llm_factory.get_client(state.selected_model)
            
            response_result = await llm_client.generate_response(
                query=state.message,
                context=state.context,
                rag_context=rag_context["context"],
                tool_results=tool_results,
                conversation_history=await self._get_conversation_history(state.session_id)
            )
            
            state.response = response_result["response"]
            state.model_used = response_result["model_used"]
            state.tokens_used = response_result["tokens_used"]
            state.cost_estimate = response_result["cost_estimate"]
            
            logger.info(f"Question processed for session {state.session_id} using model {state.model_used}")
            
            return state
            
        except Exception as e:
            logger.error(f"Question processing error: {str(e)}")
            state.error_message = "Question processing service unavailable"
            return state
    
    async def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recent conversation history for context."""
        try:
            db = next(get_db())
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if not conversation:
                return []
            
            messages = db.query(Message).filter(
                Message.conversation_id == conversation.id
            ).order_by(Message.timestamp.desc()).limit(10).all()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in reversed(messages)
            ]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []


class ResponseValidationNode:
    """Node for validating the generated response."""
    
    def __init__(self, response_validator: ResponseValidator):
        self.response_validator = response_validator
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process response validation."""
        try:
            # Validate the response
            validation_result = await self.response_validator.validate_response(
                response=state.response,
                original_query=state.message,
                context=state.context,
                rag_sources=state.rag_sources
            )
            
            state.is_valid_response = validation_result["is_valid"]
            state.validation_issues = validation_result["issues"]
            
            if not state.is_valid_response:
                logger.warning(f"Response validation failed for session {state.session_id}: {state.validation_issues}")
                
                # If validation fails, we might retry or provide a fallback
                if state.retry_count < state.max_retries:
                    state.should_retry = True
                else:
                    state.response = "I apologize, but I'm having difficulty providing a satisfactory response. Please try rephrasing your question or contact support."
                    state.is_valid_response = True  # Accept fallback response
            else:
                logger.debug(f"Response validation passed for session {state.session_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Response validation error: {str(e)}")
            state.is_valid_response = True  # Default to valid if validation fails
            return state


class ConversationRecordingNode:
    """Node for recording the conversation and tracking metrics."""
    
    def __init__(self):
        pass
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process conversation recording and metrics."""
        try:
            # Calculate processing time
            processing_time = (datetime.now() - state.processing_start_time).total_seconds() * 1000
            state.processing_time_ms = int(processing_time)
            
            # Get or create conversation
            db = next(get_db())
            conversation = db.query(Conversation).filter(
                Conversation.session_id == state.session_id
            ).first()
            
            if not conversation:
                conversation = Conversation(
                    id=uuid4(),
                    session_id=state.session_id,
                    user_identifier=state.user_id,
                    started_at=datetime.now(),
                    resolved=False,
                    resolution_attempts=0,
                    authenticated=state.is_authenticated
                )
                db.add(conversation)
                db.commit()
                state.conversation_id = str(conversation.id)
            
            # Update conversation metrics
            conversation.resolution_attempts = state.resolution_attempts
            conversation.resolved = state.is_resolved
            conversation.authenticated = state.is_authenticated
            
            # Record user message
            user_message = Message(
                id=uuid4(),
                conversation_id=conversation.id,
                content=state.message,
                role="user",
                timestamp=datetime.now(),
                model_used="",
                cached=False,
                processing_time_ms=0
            )
            db.add(user_message)
            
            # Record assistant response
            assistant_message = Message(
                id=uuid4(),
                conversation_id=conversation.id,
                content=state.response,
                role="assistant",
                timestamp=datetime.now(),
                model_used=state.model_used,
                cached=state.cache_hit,
                processing_time_ms=state.processing_time_ms
            )
            db.add(assistant_message)
            
            db.commit()
            
            logger.info(f"Conversation recorded for session {state.session_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Conversation recording error: {str(e)}")
            return state


class CacheUpdateNode:
    """Node for updating the semantic cache with successful responses."""
    
    def __init__(self, semantic_cache: SemanticCacheService):
        self.semantic_cache = semantic_cache
    
    async def process(self, state: PipelineState) -> PipelineState:
        """Process cache update for successful responses."""
        try:
            # Update cache with successful response
            await self.semantic_cache.store_response(
                query=state.message,
                response=state.response,
                context=state.context,
                session_id=state.session_id,
                metadata={
                    "model_used": state.model_used,
                    "tokens_used": state.tokens_used,
                    "cost_estimate": state.cost_estimate,
                    "rag_sources": state.rag_sources,
                    "mcp_tools_used": state.mcp_tools_used,
                    "resolution_attempts": state.resolution_attempts,
                    "processing_time_ms": state.processing_time_ms
                }
            )
            
            logger.debug(f"Cache updated for session {state.session_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Cache update error: {str(e)}")
            return state
