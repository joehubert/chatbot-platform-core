"""
Analytics schemas for API request/response validation.

This module contains Pydantic schemas for analytics endpoints including
conversation metrics, performance monitoring, user engagement, and
system health indicators.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class DateRangeParams(BaseModel):
    """Schema for date range query parameters."""
    
    start_date: datetime = Field(..., description="Start date for analytics period")
    end_date: datetime = Field(..., description="End date for analytics period")
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v

    class Config:
        schema_extra = {
            "example": {
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z"
            }
        }


class TimeSeriesData(BaseModel):
    """Schema for time series data points."""
    
    timestamp: datetime = Field(..., description="Data point timestamp")
    value: Union[int, float] = Field(..., description="Metric value")
    label: Optional[str] = Field(None, description="Optional data point label")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-01T00:00:00Z",
                "value": 45,
                "label": "daily_conversations"
            }
        }


class ConversationMetrics(BaseModel):
    """Schema for conversation-specific metrics."""
    
    total_conversations: int = Field(..., ge=0, description="Total number of conversations")
    total_messages: int = Field(..., ge=0, description="Total number of messages")
    average_messages_per_conversation: float = Field(
        ..., ge=0, description="Average messages per conversation"
    )
    completion_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Conversation completion rate"
    )
    resolution_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Issue resolution rate"
    )
    escalation_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Rate of escalations to human agents"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_conversations": 1250,
                "total_messages": 8420,
                "average_messages_per_conversation": 6.7,
                "completion_rate": 0.92,
                "resolution_rate": 0.78,
                "escalation_rate": 0.15
            }
        }


class UserEngagementMetrics(BaseModel):
    """Schema for user engagement analytics."""
    
    unique_users: int = Field(..., ge=0, description="Number of unique users")
    returning_users: int = Field(..., ge=0, description="Number of returning users")
    average_session_duration_minutes: float = Field(
        ..., ge=0, description="Average session duration in minutes"
    )
    bounce_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Percentage of single-message sessions"
    )
    user_satisfaction_score: Optional[float] = Field(
        None, ge=1.0, le=5.0, description="Average user satisfaction rating"
    )
    feedback_response_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Rate of users providing feedback"
    )

    class Config:
        schema_extra = {
            "example": {
                "unique_users": 892,
                "returning_users": 234,
                "average_session_duration_minutes": 12.3,
                "bounce_rate": 0.18,
                "user_satisfaction_score": 4.2,
                "feedback_response_rate": 0.34
            }
        }


class PerformanceMetrics(BaseModel):
    """Schema for system performance metrics."""
    
    average_response_time_ms: float = Field(
        ..., ge=0, description="Average response time in milliseconds"
    )
    p95_response_time_ms: float = Field(
        ..., ge=0, description="95th percentile response time"
    )
    p99_response_time_ms: float = Field(
        ..., ge=0, description="99th percentile response time"
    )
    throughput_requests_per_second: float = Field(
        ..., ge=0, description="Average requests per second"
    )
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Overall error rate"
    )
    uptime_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="System uptime percentage"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "average_response_time_ms": 1250.5,
                "p95_response_time_ms": 2800.0,
                "p99_response_time_ms": 4200.0,
                "throughput_requests_per_second": 45.2,
                "error_rate": 0.023,
                "uptime_percentage": 99.8
            }
        }


class CacheMetrics(BaseModel):
    """Schema for cache performance metrics."""
    
    hit_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Cache hit rate"
    )
    miss_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Cache miss rate"
    )
    total_requests: int = Field(
        ..., ge=0, description="Total cache requests"
    )
    cache_size_mb: float = Field(
        ..., ge=0, description="Cache size in megabytes"
    )
    eviction_count: int = Field(
        ..., ge=0, description="Number of cache evictions"
    )
    average_retrieval_time_ms: float = Field(
        ..., ge=0, description="Average cache retrieval time"
    )

    class Config:
        schema_extra = {
            "example": {
                "hit_rate": 0.68,
                "miss_rate": 0.32,
                "total_requests": 15420,
                "cache_size_mb": 256.8,
                "eviction_count": 45,
                "average_retrieval_time_ms": 12.5
            }
        }


class ModelUsageStats(BaseModel):
    """Schema for LLM model usage statistics."""

    llm_model_name: str = Field(..., description="Name of the LLM model")
    total_requests: int = Field(..., ge=0, description="Total requests to this model")
    total_tokens_input: int = Field(..., ge=0, description="Total input tokens")
    total_tokens_output: int = Field(..., ge=0, description="Total output tokens")
    average_response_time_ms: float = Field(
        ..., ge=0, description="Average response time for this model"
    )
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Error rate for this model"
    )
    cost_usd: Optional[float] = Field(
        None, ge=0, description="Total cost in USD (if available)"
    )

    class Config:
        schema_extra = {
            "example": {
                "llm_model_name": "gpt-3.5-turbo",
                "total_requests": 1250,
                "total_tokens_input": 425000,
                "total_tokens_output": 185000,
                "average_response_time_ms": 1180.5,
                "error_rate": 0.015,
                "cost_usd": 15.75
            }
        }


class ErrorAnalytics(BaseModel):
    """Schema for error analysis metrics."""
    
    total_errors: int = Field(..., ge=0, description="Total number of errors")
    error_types: Dict[str, int] = Field(
        ..., description="Breakdown of errors by type"
    )
    critical_errors: int = Field(..., ge=0, description="Number of critical errors")
    recoverable_errors: int = Field(..., ge=0, description="Number of recoverable errors")
    most_common_error: Optional[str] = Field(
        None, description="Most frequently occurring error type"
    )
    error_trends: List[TimeSeriesData] = Field(
        ..., description="Error trends over time"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_errors": 45,
                "error_types": {
                    "timeout": 15,
                    "rate_limit": 12,
                    "authentication": 8,
                    "internal_server_error": 10
                },
                "critical_errors": 3,
                "recoverable_errors": 42,
                "most_common_error": "timeout",
                "error_trends": []
            }
        }


class CostAnalytics(BaseModel):
    """Schema for cost analysis metrics."""
    
    total_cost_usd: float = Field(..., ge=0, description="Total cost in USD")
    cost_by_model: Dict[str, float] = Field(
        ..., description="Cost breakdown by model"
    )
    cost_per_conversation: float = Field(
        ..., ge=0, description="Average cost per conversation"
    )
    cost_trends: List[TimeSeriesData] = Field(
        ..., description="Cost trends over time"
    )
    projected_monthly_cost: Optional[float] = Field(
        None, ge=0, description="Projected monthly cost based on current usage"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_cost_usd": 127.45,
                "cost_by_model": {
                    "gpt-3.5-turbo": 45.20,
                    "gpt-4": 82.25
                },
                "cost_per_conversation": 0.102,
                "cost_trends": [],
                "projected_monthly_cost": 425.80
            }
        }


class KnowledgeBaseAnalytics(BaseModel):
    """Schema for knowledge base usage analytics."""
    
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    total_document_size_mb: float = Field(
        ..., ge=0, description="Total size of documents in MB"
    )
    search_queries: int = Field(..., ge=0, description="Total search queries")
    average_relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average search result relevance"
    )
    most_accessed_documents: List[Dict[str, Any]] = Field(
        ..., description="Most frequently accessed documents"
    )
    document_usage_patterns: List[TimeSeriesData] = Field(
        ..., description="Document usage over time"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_documents": 156,
                "total_document_size_mb": 45.8,
                "search_queries": 2890,
                "average_relevance_score": 0.78,
                "most_accessed_documents": [
                    {"document_name": "FAQ.pdf", "access_count": 145},
                    {"document_name": "Product Guide.docx", "access_count": 98}
                ],
                "document_usage_patterns": []
            }
        }


class ConversationAnalytics(BaseModel):
    """Comprehensive conversation analytics response schema."""
    
    date_range: DateRangeParams = Field(..., description="Analytics date range")
    conversation_metrics: ConversationMetrics = Field(
        ..., description="Conversation-specific metrics"
    )
    user_engagement: UserEngagementMetrics = Field(
        ..., description="User engagement metrics"
    )
    performance: PerformanceMetrics = Field(
        ..., description="System performance metrics"
    )
    time_series_data: List[TimeSeriesData] = Field(
        ..., description="Time-based conversation data"
    )
    cache_metrics: Optional[CacheMetrics] = Field(
        None, description="Cache performance metrics"
    )

    class Config:
        schema_extra = {
            "example": {
                "date_range": {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-31T23:59:59Z"
                },
                "conversation_metrics": {
                    "total_conversations": 1250,
                    "total_messages": 8420,
                    "average_messages_per_conversation": 6.7,
                    "completion_rate": 0.92,
                    "resolution_rate": 0.78,
                    "escalation_rate": 0.15
                },
                "user_engagement": {
                    "unique_users": 892,
                    "returning_users": 234,
                    "average_session_duration_minutes": 12.3,
                    "bounce_rate": 0.18,
                    "user_satisfaction_score": 4.2,
                    "feedback_response_rate": 0.34
                },
                "performance": {
                    "average_response_time_ms": 1250.5,
                    "p95_response_time_ms": 2800.0,
                    "p99_response_time_ms": 4200.0,
                    "throughput_requests_per_second": 45.2,
                    "error_rate": 0.023,
                    "uptime_percentage": 99.8
                },
                "time_series_data": [],
                "cache_metrics": {
                    "hit_rate": 0.68,
                    "miss_rate": 0.32,
                    "total_requests": 15420,
                    "cache_size_mb": 256.8,
                    "eviction_count": 45,
                    "average_retrieval_time_ms": 12.5
                }
            }
        }


class SystemAnalytics(BaseModel):
    """Comprehensive system analytics response schema."""
    
    date_range: DateRangeParams = Field(..., description="Analytics date range")
    performance: PerformanceMetrics = Field(..., description="System performance")
    llm_model_usage: List[ModelUsageStats] = Field(..., description="Model usage statistics")
    cache_metrics: CacheMetrics = Field(..., description="Cache performance")
    error_analytics: ErrorAnalytics = Field(..., description="Error analysis")
    cost_analytics: Optional[CostAnalytics] = Field(None, description="Cost analysis")
    knowledge_base: Optional[KnowledgeBaseAnalytics] = Field(
        None, description="Knowledge base analytics"
    )

    class Config:
        schema_extra = {
            "example": {
                "date_range": {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-31T23:59:59Z"
                },
                "performance": {},
                "llm_model_usage": [],
                "cache_metrics": {},
                "error_analytics": {},
                "cost_analytics": {},
                "knowledge_base": {}
            }
        }


# Export schemas for use in API endpoints
__all__ = [
    "DateRangeParams",
    "TimeSeriesData",
    "ConversationMetrics",
    "UserEngagementMetrics",
    "PerformanceMetrics",
    "CacheMetrics",
    "ModelUsageStats",
    "ErrorAnalytics",
    "CostAnalytics",
    "KnowledgeBaseAnalytics",
    "ConversationAnalytics",
    "SystemAnalytics",
]