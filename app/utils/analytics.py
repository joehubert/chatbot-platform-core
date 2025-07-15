"""
Analytics utility functions for data processing and calculations.

This module provides helper functions for computing analytics metrics,
aggregating data, and formatting responses for the analytics API endpoints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.analytics import TimeSeriesData


def generate_time_buckets(
    start_date: datetime,
    end_date: datetime,
    granularity: str
) -> List[Tuple[datetime, datetime]]:
    """
    Generate time buckets for analytics aggregation.
    
    Args:
        start_date: Start of the analytics period
        end_date: End of the analytics period
        granularity: Time bucket size ('hour', 'day', 'week', 'month')
    
    Returns:
        List of (bucket_start, bucket_end) tuples
    """
    buckets = []
    current = start_date
    
    while current < end_date:
        if granularity == "hour":
            next_bucket = current + timedelta(hours=1)
        elif granularity == "day":
            next_bucket = current + timedelta(days=1)
        elif granularity == "week":
            next_bucket = current + timedelta(weeks=1)
        elif granularity == "month":
            # Handle month boundaries more accurately
            if current.month == 12:
                next_bucket = current.replace(year=current.year + 1, month=1)
            else:
                next_bucket = current.replace(month=current.month + 1)
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
        
        # Don't exceed end_date
        bucket_end = min(next_bucket, end_date)
        buckets.append((current, bucket_end))
        current = next_bucket
        
    return buckets


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate the specified percentile from a list of values.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
    
    Returns:
        The percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    
    if index.is_integer():
        return sorted_values[int(index)]
    else:
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))


def calculate_conversation_metrics(
    db: Session,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Calculate conversation-specific metrics for the given period.
    
    Args:
        db: Database session
        start_date: Start of analytics period
        end_date: End of analytics period
    
    Returns:
        Dictionary containing conversation metrics
    """
    # Base query for conversations in date range
    conversations_query = db.query(Conversation).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date
        )
    )
    
    # Total conversations
    total_conversations = conversations_query.count()
    
    if total_conversations == 0:
        return {
            "total_conversations": 0,
            "total_messages": 0,
            "average_messages_per_conversation": 0.0,
            "completion_rate": 0.0,
            "resolution_rate": 0.0,
            "escalation_rate": 0.0
        }
    
    # Total messages
    total_messages = db.query(func.count(Message.id)).filter(
        Message.conversation_id.in_(
            conversations_query.with_entities(Conversation.id)
        )
    ).scalar()
    
    # Average messages per conversation
    avg_messages = total_messages / total_conversations if total_conversations > 0 else 0.0
    
    # Completion rate (conversations with end_time set)
    completed_conversations = conversations_query.filter(
        Conversation.ended_at.isnot(None)
    ).count()
    completion_rate = completed_conversations / total_conversations
    
    # Resolution rate (placeholder - would depend on conversation status/feedback)
    # This would typically check for positive feedback or resolution status
    resolution_rate = 0.78  # Placeholder value
    
    # Escalation rate (placeholder - would check for human handoff)
    escalation_rate = 0.15  # Placeholder value
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "average_messages_per_conversation": round(avg_messages, 2),
        "completion_rate": round(completion_rate, 3),
        "resolution_rate": round(resolution_rate, 3),
        "escalation_rate": round(escalation_rate, 3)
    }


def calculate_user_engagement_metrics(
    db: Session,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Calculate user engagement metrics for the given period.
    
    Args:
        db: Database session
        start_date: Start of analytics period
        end_date: End of analytics period
    
    Returns:
        Dictionary containing user engagement metrics
    """
    conversations_query = db.query(Conversation).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date
        )
    )
    
    # Unique users (non-null user_identifier)
    unique_users = db.query(func.count(func.distinct(Conversation.user_identifier))).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date,
            Conversation.user_identifier.isnot(None)
        )
    ).scalar() or 0
    
    # Returning users (users with multiple conversations)
    returning_users_subquery = db.query(
        Conversation.user_identifier,
        func.count(Conversation.id).label('conversation_count')
    ).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date,
            Conversation.user_identifier.isnot(None)
        )
    ).group_by(Conversation.user_identifier).subquery()
    
    returning_users = db.query(func.count()).filter(
        returning_users_subquery.c.conversation_count > 1
    ).scalar() or 0
    
    # Average session duration (placeholder calculation)
    # Would typically calculate from ended_at - started_at
    avg_session_duration = 12.3  # minutes, placeholder
    
    # Bounce rate (single message conversations)
    single_message_conversations = db.query(func.count(Conversation.id)).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date,
            db.query(func.count(Message.id)).filter(
                Message.conversation_id == Conversation.id
            ).correlate(Conversation).as_scalar() == 1
        )
    ).scalar() or 0
    
    total_conversations = conversations_query.count()
    bounce_rate = single_message_conversations / total_conversations if total_conversations > 0 else 0.0
    
    # User satisfaction (placeholder - would come from feedback)
    user_satisfaction_score = 4.2  # Placeholder value
    
    # Feedback response rate (placeholder)
    feedback_response_rate = 0.34  # Placeholder value
    
    return {
        "unique_users": unique_users,
        "returning_users": returning_users,
        "average_session_duration_minutes": avg_session_duration,
        "bounce_rate": round(bounce_rate, 3),
        "user_satisfaction_score": user_satisfaction_score,
        "feedback_response_rate": feedback_response_rate
    }


def calculate_performance_metrics(
    db: Session,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Calculate system performance metrics for the given period.
    
    Args:
        db: Database session
        start_date: Start of analytics period
        end_date: End of analytics period
    
    Returns:
        Dictionary containing performance metrics
    """
    # This would typically pull from a metrics/monitoring table
    # For now, returning placeholder values that would be calculated
    # from actual response time logs
    
    response_times = []  # Would be populated from actual data
    
    # Placeholder calculations - in real implementation, these would
    # come from stored metrics or calculated from message timestamps
    average_response_time = 1250.5
    p95_response_time = 2800.0
    p99_response_time = 4200.0
    
    # Calculate from actual data if available
    if response_times:
        average_response_time = sum(response_times) / len(response_times)
        p95_response_time = calculate_percentile(response_times, 95)
        p99_response_time = calculate_percentile(response_times, 99)
    
    return {
        "average_response_time_ms": round(average_response_time, 1),
        "p95_response_time_ms": round(p95_response_time, 1),
        "p99_response_time_ms": round(p99_response_time, 1),
        "throughput_requests_per_second": 45.2,  # Placeholder
        "error_rate": 0.023,  # Placeholder
        "uptime_percentage": 99.8  # Placeholder
    }


def generate_time_series_data(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    granularity: str,
    metric_name: str = "conversations"
) -> List[TimeSeriesData]:
    """
    Generate time series data for conversation analytics.
    
    Args:
        db: Database session
        start_date: Start of analytics period
        end_date: End of analytics period
        granularity: Time bucket granularity
        metric_name: Name of the metric to track
    
    Returns:
        List of TimeSeriesData objects
    """
    buckets = generate_time_buckets(start_date, end_date, granularity)
    time_series = []
    
    for bucket_start, bucket_end in buckets:
        if metric_name == "conversations":
            # Count conversations in this time bucket
            value = db.query(func.count(Conversation.id)).filter(
                and_(
                    Conversation.started_at >= bucket_start,
                    Conversation.started_at < bucket_end
                )
            ).scalar() or 0
        elif metric_name == "messages":
            # Count messages in this time bucket
            conversation_ids = db.query(Conversation.id).filter(
                and_(
                    Conversation.started_at >= bucket_start,
                    Conversation.started_at < bucket_end
                )
            ).subquery()
            
            value = db.query(func.count(Message.id)).filter(
                Message.conversation_id.in_(conversation_ids)
            ).scalar() or 0
        else:
            value = 0  # Default for unknown metrics
        
        time_series.append(TimeSeriesData(
            timestamp=bucket_start,
            value=value,
            label=f"{granularity}ly_{metric_name}"
        ))
    
    return time_series


def calculate_cache_metrics() -> Dict[str, Any]:
    """
    Calculate cache performance metrics.
    
    Returns:
        Dictionary containing cache metrics
    """
    # This would typically pull from Redis INFO or similar cache statistics
    # Returning placeholder values for now
    return {
        "hit_rate": 0.68,
        "miss_rate": 0.32,
        "total_requests": 15420,
        "cache_size_mb": 256.8,
        "eviction_count": 45,
        "average_retrieval_time_ms": 12.5
    }


def anonymize_user_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anonymize user data in analytics results.
    
    Args:
        data: Analytics data that may contain user information
    
    Returns:
        Anonymized analytics data
    """
    # Remove or hash any personally identifiable information
    # This is a placeholder implementation
    anonymized = data.copy()
    
    # Remove specific user identifiers if present
    if 'user_identifiers' in anonymized:
        del anonymized['user_identifiers']
    
    if 'user_details' in anonymized:
        del anonymized['user_details']
    
    return anonymized


# Export utility functions
__all__ = [
    "generate_time_buckets",
    "calculate_percentile",
    "calculate_conversation_metrics",
    "calculate_user_engagement_metrics",
    "calculate_performance_metrics",
    "generate_time_series_data",
    "calculate_cache_metrics",
    "anonymize_user_data",
]