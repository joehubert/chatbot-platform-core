from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

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
    llm_model_used: str = ""
    tokens_used: int = 0
    cost_estimate: float = 0.0
    
    # Error handling
    error_message: str = ""
    should_retry: bool = False
    retry_count: int = 0
    max_retries: int = 2
