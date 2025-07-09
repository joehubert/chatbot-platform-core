"""
LangGraph Pipeline Implementation

This module implements the core request processing pipeline using LangGraph.
The pipeline handles the complete flow from rate limiting through response validation.
"""

import logging
from typing import Dict, Any, List, Optional, Annotated
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from app.core.pipeline_nodes import (
    RateLimitingNode,
    RelevanceCheckNode,
    SemanticCacheNode,
    ModelRoutingNode,
    AuthenticationNode,
    QuestionProcessingNode,
    ResponseValidationNode,
    ConversationRecordingNode,
    CacheUpdateNode
)
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

@dataclass
class PipelineState:
    """State maintained throughout the pipeline execution."""
    # Request data
    message: str = ""
    session_id: str = ""
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Processing state
    conversation_id: str = ""
    is_relevant: bool = False
    relevance_confidence: float = 0.0
    clarification_attempts: int = 0
    max_clarification_attempts: int = 3
    
    # Cache state
    cache_hit: bool = False
    cached_response: str = ""
    cache_similarity: float = 0.0
    
    # Model routing
    selected_model: str = ""
    query_complexity: str = ""
    routing_reasoning: str = ""
    
    # Authentication
    requires_auth: bool = False
    auth_methods: List[str] = field(default_factory=list)
    is_authenticated: bool = False
    auth_session_id: Optional[str] = None
    
    # Processing results
    response: str = ""
    rag_sources: List[Dict[str, Any]] = field(default_factory=list)
    mcp_tools_used: List[str] = field(default_factory=list)
    
    # Validation
    is_valid_response: bool = False
    validation_issues: List[str] = field(default_factory=list)
    
    # Resolution tracking
    resolution_attempts: int = 0
    is_resolved: bool = False
    user_feedback: Optional[str] = None
    
    # Metadata
    processing_start_time: datetime = field(default_factory=datetime.now)
    processing_time_ms: int = 0
    model_used: str = ""
    tokens_used: int = 0
    cost_estimate: float = 0.0
    
    # Error handling
    error_message: str = ""
    should_retry: bool = False
    retry_count: int = 0
    max_retries: int = 2


class ChatbotPipeline:
    """Main pipeline orchestrator using LangGraph."""
    
    def __init__(
        self,
        rate_limiting_service: RateLimitingService,
        relevance_checker: RelevanceChecker,
        semantic_cache: SemanticCacheService,
        model_router: ModelRouter,
        auth_service: AuthService,
        knowledge_base: KnowledgeBaseService,
        mcp_registry: MCPRegistry,
        response_validator: ResponseValidator
    ):
        self.rate_limiting_service = rate_limiting_service
        self.relevance_checker = relevance_checker
        self.semantic_cache = semantic_cache
        self.model_router = model_router
        self.auth_service = auth_service
        self.knowledge_base = knowledge_base
        self.mcp_registry = mcp_registry
        self.response_validator = response_validator
        
        # Initialize nodes
        self.rate_limiting_node = RateLimitingNode(rate_limiting_service)
        self.relevance_check_node = RelevanceCheckNode(relevance_checker)
        self.semantic_cache_node = SemanticCacheNode(semantic_cache)
        self.model_routing_node = ModelRoutingNode(model_router)
        self.auth_node = AuthenticationNode(auth_service)
        self.question_processing_node = QuestionProcessingNode(
            knowledge_base, mcp_registry
        )
        self.response_validation_node = ResponseValidationNode(response_validator)
        self.conversation_recording_node = ConversationRecordingNode()
        self.cache_update_node = CacheUpdateNode(semantic_cache)
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph processing pipeline."""
        
        # Create the graph
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("rate_limiting", self.rate_limiting_node.process)
        workflow.add_node("relevance_check", self.relevance_check_node.process)
        workflow.add_node("semantic_cache", self.semantic_cache_node.process)
        workflow.add_node("model_routing", self.model_routing_node.process)
        workflow.add_node("authentication", self.auth_node.process)
        workflow.add_node("question_processing", self.question_processing_node.process)
        workflow.add_node("response_validation", self.response_validation_node.process)
        workflow.add_node("conversation_recording", self.conversation_recording_node.process)
        workflow.add_node("cache_update", self.cache_update_node.process)
        
        # Define the flow
        workflow.set_entry_point("rate_limiting")
        
        # Rate limiting -> relevance check
        workflow.add_edge("rate_limiting", "relevance_check")
        
        # Relevance check -> conditional routing
        workflow.add_conditional_edges(
            "relevance_check",
            self._should_continue_after_relevance,
            {
                "continue": "semantic_cache",
                "clarify": "relevance_check",
                "irrelevant": "conversation_recording"
            }
        )
        
        # Semantic cache -> conditional routing
        workflow.add_conditional_edges(
            "semantic_cache",
            self._should_continue_after_cache,
            {
                "cache_hit": "response_validation",
                "cache_miss": "model_routing"
            }
        )
        
        # Model routing -> authentication check
        workflow.add_edge("model_routing", "authentication")
        
        # Authentication -> conditional routing
        workflow.add_conditional_edges(
            "authentication",
            self._should_continue_after_auth,
            {
                "authenticated": "question_processing",
                "needs_auth": "conversation_recording"
            }
        )
        
        # Question processing -> response validation
        workflow.add_edge("question_processing", "response_validation")
        
        # Response validation -> conditional routing
        workflow.add_conditional_edges(
            "response_validation",
            self._should_continue_after_validation,
            {
                "valid": "conversation_recording",
                "retry": "question_processing",
                "failed": "conversation_recording"
            }
        )
        
        # Conversation recording -> cache update
        workflow.add_conditional_edges(
            "conversation_recording",
            self._should_update_cache,
            {
                "update_cache": "cache_update",
                "end": END
            }
        )
        
        # Cache update -> end
        workflow.add_edge("cache_update", END)
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _should_continue_after_relevance(self, state: PipelineState) -> str:
        """Determine the next step after relevance checking."""
        if state.error_message:
            return "irrelevant"
        
        if not state.is_relevant:
            if state.clarification_attempts < state.max_clarification_attempts:
                return "clarify"
            else:
                return "irrelevant"
        
        return "continue"
    
    def _should_continue_after_cache(self, state: PipelineState) -> str:
        """Determine the next step after cache check."""
        return "cache_hit" if state.cache_hit else "cache_miss"
    
    def _should_continue_after_auth(self, state: PipelineState) -> str:
        """Determine the next step after authentication."""
        if state.requires_auth and not state.is_authenticated:
            return "needs_auth"
        return "authenticated"
    
    def _should_continue_after_validation(self, state: PipelineState) -> str:
        """Determine the next step after response validation."""
        if not state.is_valid_response:
            if state.retry_count < state.max_retries:
                state.retry_count += 1
                return "retry"
            else:
                return "failed"
        return "valid"
    
    def _should_update_cache(self, state: PipelineState) -> str:
        """Determine if cache should be updated."""
        if (state.is_resolved and 
            state.is_valid_response and 
            not state.cache_hit and 
            not state.error_message):
            return "update_cache"
        return "end"
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a chat message through the complete pipeline."""
        
        # Initialize state
        state = PipelineState(
            message=message,
            session_id=session_id,
            user_id=user_id,
            context=context or {},
            conversation_id=str(uuid4()),
            processing_start_time=datetime.now()
        )
        
        try:
            # Run the pipeline
            config = {"configurable": {"thread_id": session_id}}
            result = await self.graph.ainvoke(state, config=config)
            
            # Calculate processing time
            processing_time = (datetime.now() - state.processing_start_time).total_seconds() * 1000
            result.processing_time_ms = int(processing_time)
            
            # Return the response
            return {
                "response": result.response,
                "session_id": session_id,
                "conversation_id": result.conversation_id,
                "requires_auth": result.requires_auth,
                "auth_methods": result.auth_methods,
                "cached": result.cache_hit,
                "model_used": result.model_used,
                "processing_time_ms": result.processing_time_ms,
                "is_resolved": result.is_resolved,
                "rag_sources": result.rag_sources,
                "mcp_tools_used": result.mcp_tools_used,
                "error_message": result.error_message
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {str(e)}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "session_id": session_id,
                "conversation_id": state.conversation_id,
                "requires_auth": False,
                "auth_methods": [],
                "cached": False,
                "model_used": "",
                "processing_time_ms": 0,
                "is_resolved": False,
                "rag_sources": [],
                "mcp_tools_used": [],
                "error_message": str(e)
            }
    
    async def handle_auth_request(
        self,
        session_id: str,
        contact_method: str,
        contact_value: str
    ) -> Dict[str, Any]:
        """Handle authentication token request."""
        try:
            return await self.auth_service.request_token(
                session_id, contact_method, contact_value
            )
        except Exception as e:
            logger.error(f"Auth request error: {str(e)}")
            return {
                "success": False,
                "error": "Failed to send authentication token"
            }
    
    async def handle_auth_verification(
        self,
        session_id: str,
        token: str
    ) -> Dict[str, Any]:
        """Handle authentication token verification."""
        try:
            return await self.auth_service.verify_token(session_id, token)
        except Exception as e:
            logger.error(f"Auth verification error: {str(e)}")
            return {
                "success": False,
                "error": "Failed to verify authentication token"
            }
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            # Get conversation from database
            db = next(get_db())
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if not conversation:
                return []
            
            # Get messages
            messages = db.query(Message).filter(
                Message.conversation_id == conversation.id
            ).order_by(Message.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    "id": str(msg.id),
                    "content": msg.content,
                    "role": msg.role,
                    "timestamp": msg.timestamp.isoformat(),
                    "model_used": msg.model_used,
                    "cached": msg.cached,
                    "processing_time_ms": msg.processing_time_ms
                }
                for msg in reversed(messages)
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    async def get_pipeline_health(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        try:
            health_status = {
                "pipeline": "healthy",
                "components": {
                    "rate_limiting": await self.rate_limiting_service.health_check(),
                    "relevance_checker": await self.relevance_checker.health_check(),
                    "semantic_cache": await self.semantic_cache.health_check(),
                    "model_router": await self.model_router.health_check(),
                    "auth_service": await self.auth_service.health_check(),
                    "knowledge_base": await self.knowledge_base.health_check(),
                    "mcp_registry": await self.mcp_registry.health_check(),
                    "response_validator": await self.response_validator.health_check()
                }
            }
            
            # Check if any component is unhealthy
            unhealthy_components = [
                name for name, status in health_status["components"].items()
                if status != "healthy"
            ]
            
            if unhealthy_components:
                health_status["pipeline"] = "degraded"
                health_status["issues"] = unhealthy_components
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                "pipeline": "unhealthy",
                "error": str(e)
            }
