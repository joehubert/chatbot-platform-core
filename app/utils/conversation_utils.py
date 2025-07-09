"""
Conversation Utilities

Utility classes and functions for conversation management, session handling,
and metrics collection. Contains data structures and helper functions used
by the conversation and session managers.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """
    Summary of a conversation for caching and analytics purposes.
    
    Used by the semantic cache to store conversation summaries
    and by analytics to track conversation effectiveness.
    """
    conversation_id: UUID
    session_id: str
    user_messages: List[str]
    assistant_messages: List[str]
    resolved: bool
    resolution_attempts: int
    authenticated: bool
    started_at: datetime
    ended_at: Optional[datetime]
    total_messages: int
    avg_processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": str(self.conversation_id),
            "session_id": self.session_id,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "resolved": self.resolved,
            "resolution_attempts": self.resolution_attempts,
            "authenticated": self.authenticated,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_messages": self.total_messages,
            "avg_processing_time_ms": self.avg_processing_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary."""
        return cls(
            conversation_id=UUID(data["conversation_id"]),
            session_id=data["session_id"],
            user_messages=data["user_messages"],
            assistant_messages=data["assistant_messages"],
            resolved=data["resolved"],
            resolution_attempts=data["resolution_attempts"],
            authenticated=data["authenticated"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data["ended_at"] else None,
            total_messages=data["total_messages"],
            avg_processing_time_ms=data["avg_processing_time_ms"]
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConversationSummary":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_cache_key(self) -> str:
        """Generate cache key for semantic cache."""
        # Combine user messages to create a unique key
        user_query = " ".join(self.user_messages)
        return f"conv_summary:{hash(user_query)}"
    
    def get_embedding_text(self) -> str:
        """Get text for embedding generation."""
        return " ".join(self.user_messages + self.assistant_messages)


@dataclass
class ConversationMetrics:
    """
    Metrics for conversation analytics and monitoring.
    
    Used by the analytics service to track system performance
    and conversation effectiveness.
    """
    total_conversations: int
    resolved_conversations: int
    resolution_rate: float
    avg_resolution_attempts: float
    avg_duration_seconds: float
    total_messages: int
    avg_processing_time_ms: float
    cache_hit_rate: float
    start_date: datetime
    end_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_conversations": self.total_conversations,
            "resolved_conversations": self.resolved_conversations,
            "resolution_rate": self.resolution_rate,
            "avg_resolution_attempts": self.avg_resolution_attempts,
            "avg_duration_seconds": self.avg_duration_seconds,
            "total_messages": self.total_messages,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMetrics":
        """Create from dictionary."""
        return cls(
            total_conversations=data["total_conversations"],
            resolved_conversations=data["resolved_conversations"],
            resolution_rate=data["resolution_rate"],
            avg_resolution_attempts=data["avg_resolution_attempts"],
            avg_duration_seconds=data["avg_duration_seconds"],
            total_messages=data["total_messages"],
            avg_processing_time_ms=data["avg_processing_time_ms"],
            cache_hit_rate=data["cache_hit_rate"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"])
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConversationMetrics":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ConversationContext:
    """
    Context for maintaining conversation state across requests.
    
    Used by the session manager to preserve conversation context
    and manage conversation flow.
    """
    session_id: str
    messages: List[Dict[str, Any]]
    entities: Dict[str, Any]
    summary: str
    turn_count: int
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "entities": self.entities,
            "summary": self.summary,
            "turn_count": self.turn_count,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            messages=data["messages"],
            entities=data["entities"],
            summary=data["summary"],
            turn_count=data["turn_count"],
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConversationContext":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from context."""
        return self.messages[-count:] if len(self.messages) > count else self.messages
    
    def get_user_messages(self) -> List[str]:
        """Get all user messages from context."""
        return [msg["content"] for msg in self.messages if msg["role"] == "user"]
    
    def get_assistant_messages(self) -> List[str]:
        """Get all assistant messages from context."""
        return [msg["content"] for msg in self.messages if msg["role"] == "assistant"]
    
    def extract_entities(self) -> Dict[str, Any]:
        """Extract entities from conversation context."""
        # Simple entity extraction - in production, use NER
        entities = {}
        
        for message in self.messages:
            if message["role"] == "user":
                content = message["content"].lower()
                
                # Extract email addresses
                import re
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                if emails:
                    entities["emails"] = emails
                
                # Extract phone numbers (simple pattern)
                phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b', content)
                if phones:
                    entities["phone_numbers"] = phones
                
                # Extract common business terms
                business_terms = ["order", "product", "service", "support", "help", "issue", "problem"]
                mentioned_terms = [term for term in business_terms if term in content]
                if mentioned_terms:
                    entities["business_topics"] = mentioned_terms
        
        return entities


@dataclass
class SessionData:
    """
    Session data for user session management.
    
    Used by the session manager to track user sessions
    and maintain authentication state.
    """
    session_id: str
    user_identifier: Optional[str]
    created_at: datetime
    last_activity: datetime
    authenticated: bool
    context: Dict[str, Any]
    conversation_count: int
    total_messages: int
    authenticated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "user_identifier": self.user_identifier,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "authenticated": self.authenticated,
            "context": self.context,
            "conversation_count": self.conversation_count,
            "total_messages": self.total_messages,
            "authenticated_at": self.authenticated_at.isoformat() if self.authenticated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_identifier=data.get("user_identifier"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            authenticated=data["authenticated"],
            context=data["context"],
            conversation_count=data["conversation_count"],
            total_messages=data["total_messages"],
            authenticated_at=datetime.fromisoformat(data["authenticated_at"]) if data.get("authenticated_at") else None
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "SessionData":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session is expired."""
        from datetime import timedelta
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.utcnow() - self.last_activity > timeout
    
    def get_duration(self) -> float:
        """Get session duration in seconds."""
        return (self.last_activity - self.created_at).total_seconds()


# Utility functions for conversation management

def format_conversation_for_llm(messages: List[Dict[str, Any]]) -> str:
    """
    Format conversation messages for LLM input.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Formatted conversation string
    """
    formatted_messages = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")
        
        if role == "user":
            formatted_messages.append(f"User: {content}")
        elif role == "assistant":
            formatted_messages.append(f"Assistant: {content}")
        elif role == "system":
            formatted_messages.append(f"System: {content}")
    
    return "\n".join(formatted_messages)


def extract_conversation_topics(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Extract main topics from conversation messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of extracted topics
    """
    topics = []
    
    # Simple keyword-based topic extraction
    topic_keywords = {
        "support": ["help", "issue", "problem", "error", "bug", "trouble"],
        "product": ["product", "feature", "service", "offering", "solution"],
        "billing": ["payment", "bill", "charge", "cost", "price", "invoice"],
        "account": ["account", "profile", "settings", "login", "password"],
        "technical": ["api", "code", "integration", "technical", "development"],
        "sales": ["buy", "purchase", "order", "quote", "pricing", "sales"]
    }
    
    user_messages = [msg["content"].lower() for msg in messages if msg["role"] == "user"]
    all_user_text = " ".join(user_messages)
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in all_user_text for keyword in keywords):
            topics.append(topic)
    
    return topics


def calculate_conversation_sentiment(messages: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate basic sentiment analysis for conversation.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary with sentiment scores
    """
    # Simple sentiment analysis using keyword matching
    positive_words = ["good", "great", "excellent", "awesome", "helpful", "thanks", "thank you", "perfect"]
    negative_words = ["bad", "terrible", "awful", "hate", "horrible", "disappointed", "frustrated", "angry"]
    
    user_messages = [msg["content"].lower() for msg in messages if msg["role"] == "user"]
    all_user_text = " ".join(user_messages)
    
    positive_count = sum(1 for word in positive_words if word in all_user_text)
    negative_count = sum(1 for word in negative_words if word in all_user_text)
    
    total_words = len(all_user_text.split())
    
    sentiment_score = (positive_count - negative_count) / max(total_words, 1)
    
    return {
        "sentiment_score": sentiment_score,
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "total_words": total_words
    }


def validate_conversation_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize conversation data.
    
    Args:
        data: Raw conversation data
        
    Returns:
        Validated and sanitized data
    """
    validated = {}
    
    # Validate session_id
    if "session_id" in data and isinstance(data["session_id"], str):
        validated["session_id"] = data["session_id"]
    
    # Validate messages
    if "messages" in data and isinstance(data["messages"], list):
        validated_messages = []
        for message in data["messages"]:
            if isinstance(message, dict):
                validated_message = {}
                
                # Validate role
                if "role" in message and message["role"] in ["user", "assistant", "system"]:
                    validated_message["role"] = message["role"]
                
                # Validate content
                if "content" in message and isinstance(message["content"], str):
                    # Sanitize content (remove potential harmful content)
                    content = message["content"].strip()
                    if len(content) <= 10000:  # Limit message length
                        validated_message["content"] = content
                
                # Validate timestamp
                if "timestamp" in message:
                    try:
                        if isinstance(message["timestamp"], str):
                            datetime.fromisoformat(message["timestamp"])
                            validated_message["timestamp"] = message["timestamp"]
                        elif isinstance(message["timestamp"], datetime):
                            validated_message["timestamp"] = message["timestamp"].isoformat()
                    except ValueError:
                        validated_message["timestamp"] = datetime.utcnow().isoformat()
                
                if "role" in validated_message and "content" in validated_message:
                    validated_messages.append(validated_message)
        
        validated["messages"] = validated_messages
    
    # Validate other fields
    for field in ["entities", "context", "summary"]:
        if field in data:
            validated[field] = data[field]
    
    return validated


def merge_conversation_contexts(
    primary_context: ConversationContext,
    secondary_context: ConversationContext
) -> ConversationContext:
    """
    Merge two conversation contexts.
    
    Args:
        primary_context: Primary context (takes precedence)
        secondary_context: Secondary context to merge
        
    Returns:
        Merged conversation context
    """
    merged_messages = []
    
    # Merge messages by timestamp
    all_messages = primary_context.messages + secondary_context.messages
    
    # Sort by timestamp if available
    try:
        all_messages.sort(key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01")))
    except (ValueError, TypeError):
        # If timestamp parsing fails, keep original order
        pass
    
    # Remove duplicates based on content and timestamp
    seen_messages = set()
    for message in all_messages:
        key = (message.get("content", ""), message.get("timestamp", ""))
        if key not in seen_messages:
            merged_messages.append(message)
            seen_messages.add(key)
    
    # Merge entities
    merged_entities = {}
    merged_entities.update(secondary_context.entities)
    merged_entities.update(primary_context.entities)  # Primary takes precedence
    
    # Use primary context's other fields
    return ConversationContext(
        session_id=primary_context.session_id,
        messages=merged_messages,
        entities=merged_entities,
        summary=primary_context.summary or secondary_context.summary,
        turn_count=len(merged_messages),
        last_updated=max(primary_context.last_updated, secondary_context.last_updated)
    )


def create_conversation_embedding_text(
    conversation_summary: ConversationSummary,
    include_metadata: bool = True
) -> str:
    """
    Create text for embedding generation from conversation summary.
    
    Args:
        conversation_summary: Conversation summary object
        include_metadata: Whether to include metadata in embedding text
        
    Returns:
        Text suitable for embedding generation
    """
    text_parts = []
    
    # Add user messages
    if conversation_summary.user_messages:
        text_parts.append("User queries: " + " ".join(conversation_summary.user_messages))
    
    # Add assistant messages
    if conversation_summary.assistant_messages:
        text_parts.append("Assistant responses: " + " ".join(conversation_summary.assistant_messages))
    
    # Add metadata if requested
    if include_metadata:
        metadata_parts = []
        
        if conversation_summary.resolved:
            metadata_parts.append("resolved")
        
        if conversation_summary.authenticated:
            metadata_parts.append("authenticated")
        
        if conversation_summary.resolution_attempts > 1:
            metadata_parts.append(f"attempts:{conversation_summary.resolution_attempts}")
        
        if metadata_parts:
            text_parts.append("Metadata: " + " ".join(metadata_parts))
    
    return " | ".join(text_parts)


def generate_conversation_hash(messages: List[Dict[str, Any]]) -> str:
    """
    Generate a hash for conversation messages for deduplication.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Hash string for the conversation
    """
    import hashlib
    
    # Create a string representation of the conversation
    conversation_text = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        conversation_text += f"{role}:{content}|"
    
    # Generate hash
    return hashlib.md5(conversation_text.encode()).hexdigest()


def estimate_conversation_complexity(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Estimate conversation complexity for routing decisions.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary with complexity metrics
    """
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    
    if not user_messages:
        return {"complexity_score": 0, "reasoning": "No user messages"}
    
    complexity_indicators = {
        "technical_terms": 0,
        "question_count": 0,
        "total_words": 0,
        "unique_words": 0,
        "avg_sentence_length": 0
    }
    
    all_text = " ".join(user_messages).lower()
    
    # Technical terms
    technical_terms = ["api", "code", "integration", "database", "server", "error", "bug", "technical"]
    complexity_indicators["technical_terms"] = sum(1 for term in technical_terms if term in all_text)
    
    # Question count
    complexity_indicators["question_count"] = all_text.count("?")
    
    # Word statistics
    words = all_text.split()
    complexity_indicators["total_words"] = len(words)
    complexity_indicators["unique_words"] = len(set(words))
    
    # Average sentence length
    sentences = all_text.split(".")
    if sentences:
        complexity_indicators["avg_sentence_length"] = sum(len(s.split()) for s in sentences) / len(sentences)
    
    # Calculate complexity score (0-1)
    complexity_score = min(1.0, (
        complexity_indicators["technical_terms"] * 0.3 +
        complexity_indicators["question_count"] * 0.2 +
        min(complexity_indicators["total_words"] / 100, 1.0) * 0.3 +
        min(complexity_indicators["unique_words"] / 50, 1.0) * 0.2
    ))
    
    return {
        "complexity_score": complexity_score,
        "indicators": complexity_indicators,
        "reasoning": f"Score based on {complexity_indicators['technical_terms']} technical terms, "
                    f"{complexity_indicators['question_count']} questions, "
                    f"{complexity_indicators['total_words']} words"
    }
