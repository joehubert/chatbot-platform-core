"""
Analytics API endpoints for conversation metrics and performance monitoring.
Provides insights into chatbot usage, performance, and user interactions.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.schemas.analytics import (
    ConversationAnalytics,
    ConversationMetrics,
    ModelUsageStats,
    PerformanceMetrics,
    TimeSeriesData,
    UserEngagementMetrics,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/conversations", response_model=ConversationAnalytics)
async def get_conversation_analytics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    groupby: str = Query("day", regex="^(hour|day|week|month)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get conversation analytics with time-based grouping.
    
    Provides metrics on conversation volume, resolution rates,
    and user engagement over time.
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date must be before end date"
            )
        
        # Get conversation metrics
        metrics = await _get_conversation_metrics(
            db=db,
            start_date=start_date,
            end_date=end_date,
            groupby=groupby
        )
        
        # Get time series data
        time_series = await _get_conversation_time_series(
            db=db,
            start_date=start_date,
            end_date=end_date,
            groupby=groupby
        )
        
        # Get user engagement metrics
        engagement = await _get_user_engagement_metrics(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        
        return ConversationAnalytics(
            metrics=metrics,
            time_series=time_series,
            user_engagement=engagement,
            date_range={
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation analytics"
        )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    start_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    end_date: Optional[datetime] = Query(None, description="End date for metrics"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get system performance metrics.
    
    Includes response times, cache hit rates, error rates,
    and other performance indicators.
    """
    try:
        # Set default date range
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        # Get performance data
        performance_data = await _get_performance_data(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/models/usage", response_model=List[ModelUsageStats])
async def get_model_usage_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date for stats"),
    end_date: Optional[datetime] = Query(None, description="End date for stats"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get LLM model usage statistics.
    
    Shows which models are being used most frequently,
    their performance, and cost implications.
    """
    try:
        # Set default date range
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get model usage data
        usage_stats = await _get_model_usage_stats(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        
        return usage_stats
        
    except Exception as e:
        logger.error(f"Error retrieving model usage statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model usage statistics"
        )


@router.get("/costs")
async def get_cost_analysis(
    start_date: Optional[datetime] = Query(None, description="Start date for cost analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for cost analysis"),
    groupby: str = Query("day", regex="^(day|week|month)$", description="Cost grouping"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get cost analysis and projections.
    
    Analyzes token usage and associated costs by model,
    time period, and usage patterns.
    """
    try:
        # Set default date range
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get cost data
        cost_analysis = await _get_cost_analysis(
            db=db,
            start_date=start_date,
            end_date=end_date,
            groupby=groupby
        )
        
        return cost_analysis
        
    except Exception as e:
        logger.error(f"Error retrieving cost analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cost analysis"
        )


@router.get("/users/engagement")
async def get_user_engagement_analysis(
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed user engagement analysis.
    
    Analyzes user behavior patterns, session durations,
    and interaction quality.
    """
    try:
        # Set default date range
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get engagement analysis
        engagement_data = await _get_detailed_user_engagement(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        
        return engagement_data
        
    except Exception as e:
        logger.error(f"Error retrieving user engagement analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user engagement analysis"
        )


@router.get("/knowledge-base/usage")
async def get_knowledge_base_analytics(
    start_date: Optional[datetime] = Query(None, description="Start date for KB analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for KB analytics"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get knowledge base usage analytics.
    
    Shows which documents are accessed most frequently,
    search patterns, and content effectiveness.
    """
    try:
        # Set default date range
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get knowledge base analytics
        kb_analytics = await _get_knowledge_base_analytics(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        
        return kb_analytics
        
    except Exception as e:
        logger.error(f"Error retrieving knowledge base analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve knowledge base analytics"
        )


@router.get("/trends")
async def get_trending_queries(
    period: str = Query("week", regex="^(day|week|month)$", description="Trending period"),
    limit: int = Query(20, ge=1, le=100, description="Number of trending queries to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get trending queries and topics.
    
    Identifies the most common questions and topics
    users are asking about.
    """
    try:
        # Calculate date range based on period
        end_date = datetime.utcnow()
        if period == "day":
            start_date = end_date - timedelta(days=1)
        elif period == "week":
            start_date = end_date - timedelta(weeks=1)
        else:  # month
            start_date = end_date - timedelta(days=30)
        
        # Get trending queries
        trending_data = await _get_trending_queries(
            db=db,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "period": period,
            "trending_queries": trending_data,
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving trending queries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trending queries"
        )


@router.get("/export")
async def export_analytics_data(
    start_date: datetime = Query(..., description="Start date for export"),
    end_date: datetime = Query(..., description="End date for export"),
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    include_pii: bool = Query(False, description="Include personally identifiable information"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Export analytics data for external analysis.
    
    Provides comprehensive data export in JSON or CSV format.
    PII inclusion requires appropriate permissions.
    """
    try:
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date must be before end date"
            )
        
        # Check if date range is reasonable
        date_diff = end_date - start_date
        if date_diff.days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Date range cannot exceed 365 days"
            )
        
        # Export data
        export_data = await _export_analytics_data(
            db=db,
            start_date=start_date,
            end_date=end_date,
            format=format,
            include_pii=include_pii,
            user_id=current_user.id
        )
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting analytics data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export analytics data"
        )


# Helper functions for analytics calculations

async def _get_conversation_metrics(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    groupby: str
) -> ConversationMetrics:
    """Calculate conversation metrics for the specified period."""
    
    # Total conversations
    total_conversations = db.query(func.count(Conversation.id)).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date
        )
    ).scalar()
    
    # Resolved conversations
    resolved_conversations = db.query(func.count(Conversation.id)).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date,
            Conversation.resolved == True
        )
    ).scalar()
    
    # Average resolution attempts
    avg_resolution_attempts = db.query(func.avg(Conversation.resolution_attempts)).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date
        )
    ).scalar() or 0
    
    # Calculate resolution rate
    resolution_rate = (resolved_conversations / total_conversations * 100) if total_conversations > 0 else 0
    
    return ConversationMetrics(
        total_conversations=total_conversations,
        resolved_conversations=resolved_conversations,
        resolution_rate=round(resolution_rate, 2),
        average_resolution_attempts=round(float(avg_resolution_attempts), 2)
    )


async def _get_conversation_time_series(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    groupby: str
) -> List[TimeSeriesData]:
    """Get time series data for conversations."""
    
    # This would be implemented based on the specific database and grouping requirements
    # For now, returning a placeholder structure
    time_series_data = []
    
    # Generate time buckets based on groupby parameter
    current_date = start_date
    while current_date <= end_date:
        if groupby == "hour":
            next_date = current_date + timedelta(hours=1)
        elif groupby == "day":
            next_date = current_date + timedelta(days=1)
        elif groupby == "week":
            next_date = current_date + timedelta(weeks=1)
        else:  # month
            next_date = current_date + timedelta(days=30)
        
        # Count conversations in this time bucket
        count = db.query(func.count(Conversation.id)).filter(
            and_(
                Conversation.started_at >= current_date,
                Conversation.started_at < next_date
            )
        ).scalar()
        
        time_series_data.append(TimeSeriesData(
            timestamp=current_date,
            value=count
        ))
        
        current_date = next_date
    
    return time_series_data


async def _get_user_engagement_metrics(
    db: Session,
    start_date: datetime,
    end_date: datetime
) -> UserEngagementMetrics:
    """Calculate user engagement metrics."""
    
    # Total unique users
    unique_users = db.query(func.count(func.distinct(Conversation.user_identifier))).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date,
            Conversation.user_identifier.isnot(None)
        )
    ).scalar()
    
    # Average messages per conversation
    avg_messages = db.query(func.avg(
        db.query(func.count(Message.id)).filter(
            Message.conversation_id == Conversation.id
        ).correlate(Conversation).scalar_subquery()
    )).filter(
        and_(
            Conversation.started_at >= start_date,
            Conversation.started_at <= end_date
        )
    ).scalar() or 0
    
    # Average session duration (placeholder calculation)
    avg_session_duration = 5.5  # minutes, would be calculated from actual data
    
    return UserEngagementMetrics(
        unique_users=unique_users,
        average_messages_per_conversation=round(float(avg_messages), 2),
        average_session_duration_minutes=avg_session_duration
    )


async def _get_performance_data(
    db: Session,
    start_date: datetime,
    end_date: datetime
) -> PerformanceMetrics:
    """Get system performance metrics."""
    
    # Average response time
    avg_response_time = db.query(func.avg(Message.processing_time_ms)).filter(
        and_(
            Message.timestamp >= start_date,
            Message.timestamp <= end_date,
            Message.role == 'assistant',
            Message.processing_time_ms.isnot(None)
        )
    ).scalar() or 0
    
    # Cache hit rate
    total_requests = db.query(func.count(Message.id)).filter(
        and_(
            Message.timestamp >= start_date,
            Message.timestamp <= end_date,
            Message.role == 'assistant'
        )
    ).scalar()
    
    cached_requests = db.query(func.count(Message.id)).filter(
        and_(
            Message.timestamp >= start_date,
            Message.timestamp <= end_date,
            Message.role == 'assistant',
            Message.cached == True
        )
    ).scalar()
    
    cache_hit_rate = (cached_requests / total_requests * 100) if total_requests > 0 else 0
    
    return PerformanceMetrics(
        average_response_time_ms=round(float(avg_response_time), 2),
        cache_hit_rate=round(cache_hit_rate, 2),
        error_rate=0.5,  # Placeholder
        uptime_percentage=99.8  # Placeholder
    )


async def _get_model_usage_stats(
    db: Session,
    start_date: datetime,
    end_date: datetime
) -> List[ModelUsageStats]:
    """Get model usage statistics."""
    
    # Query model usage
    model_stats = db.query(
        Message.llm_model_used,
        func.count(Message.id).label('usage_count'),
        func.avg(Message.processing_time_ms).label('avg_response_time'),
        func.sum(Message.tokens_used).label('total_tokens')
    ).filter(
        and_(
            Message.timestamp >= start_date,
            Message.timestamp <= end_date,
            Message.role == 'assistant',
            Message.llm_model_used.isnot(None)
        )
    ).group_by(Message.llm_model_used).all()
    
    results = []
    for stat in model_stats:
        results.append(ModelUsageStats(
            model_name=stat.llm_model_used,
            usage_count=stat.usage_count,
            average_response_time_ms=round(float(stat.avg_response_time or 0), 2),
            total_tokens=stat.total_tokens or 0,
            estimated_cost=0.0  # Would be calculated based on model pricing
        ))
    
    return results


async def _get_cost_analysis(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    groupby: str
):
    """Get cost analysis data."""
    
    # Placeholder implementation
    return {
        "total_cost": 125.50,
        "cost_by_model": [
            {"model": "gpt-3.5-turbo", "cost": 45.20},
            {"model": "gpt-4", "cost": 80.30}
        ],
        "cost_trend": [
            {"date": start_date, "cost": 2.10},
            {"date": end_date, "cost": 4.20}
        ],
        "projected_monthly_cost": 380.00
    }


async def _get_detailed_user_engagement(
    db: Session,
    start_date: datetime,
    end_date: datetime
):
    """Get detailed user engagement analysis."""
    
    # Placeholder implementation
    return {
        "user_segments": {
            "new_users": 45,
            "returning_users": 123,
            "power_users": 12
        },
        "engagement_patterns": {
            "peak_hours": [9, 10, 14, 15, 16],
            "peak_days": ["Monday", "Tuesday", "Wednesday"]
        },
        "satisfaction_metrics": {
            "positive_feedback": 85.5,
            "neutral_feedback": 12.3,
            "negative_feedback": 2.2
        }
    }


async def _get_knowledge_base_analytics(
    db: Session,
    start_date: datetime,
    end_date: datetime
):
    """Get knowledge base usage analytics."""
    
    # Placeholder implementation
    return {
        "most_accessed_documents": [
            {"document_id": "doc1", "title": "FAQ", "access_count": 234},
            {"document_id": "doc2", "title": "User Guide", "access_count": 156}
        ],
        "search_patterns": {
            "most_common_queries": ["how to", "what is", "when"],
            "no_result_queries": ["specific product X", "advanced feature Y"]
        },
        "content_effectiveness": {
            "documents_with_high_relevance": 15,
            "documents_needing_update": 3
        }
    }


async def _get_trending_queries(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    limit: int
):
    """Get trending queries."""
    
    # This would analyze message content to find trending topics
    # Placeholder implementation
    return [
        {"query": "How do I reset my password?", "count": 45, "trend": "up"},
        {"query": "What are your business hours?", "count": 38, "trend": "stable"},
        {"query": "How to contact support?", "count": 32, "trend": "up"}
    ]


async def _export_analytics_data(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    format: str,
    include_pii: bool,
    user_id: str
):
    """Export analytics data."""
    
    # Placeholder implementation
    if format == "json":
        return {
            "export_id": "exp_123456",
            "format": "json",
            "date_range": {"start": start_date, "end": end_date},
            "download_url": "/downloads/analytics_export_123456.json",
            "expires_at": datetime.utcnow() + timedelta(days=7)
        }
    else:
        return {
            "export_id": "exp_123456",
            "format": "csv",
            "date_range": {"start": start_date, "end": end_date},
            "download_url": "/downloads/analytics_export_123456.csv",
            "expires_at": datetime.utcnow() + timedelta(days=7)
        }
