"""
Chat API endpoints.

This module provides the chat functionality endpoints including message processing,
conversation management, and RAG-based responses.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user, check_rate_limit
from app.core.database import get_database_session
from app.core.chat_pipeline import ChatPipeline
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ConversationResponse,
    ConversationListResponse
)
from app.schemas.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.auth_service import AuthService
from app.services.conversation_service import ConversationService
from app.services.cache_service import CacheService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: bool = Depends(check_rate_limit)
):
    """
    Process a chat message through the complete pipeline.
    
    This endpoint handles the full message processing pipeline including:
    - Rate limiting
    - Relevance checking
    - Semantic cache lookup
    - Model routing
    - Authentication (if required)
    - RAG processing
    - Response generation
    - Cache updates
    """
    try:
        # Initialize chat pipeline
        chat_pipeline = ChatPipeline(db=db)
        
        # Process the message through the pipeline
        result = await chat_pipeline.process_message(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            context=request.context
        )
        
        # Schedule background tasks for analytics and cache updates
        background_tasks.add_task(
            _update_conversation_metrics,
            conversation_id=result.conversation_id,
            processing_time=result.processing_time_ms
        )
        
        if result.cached:
            background_tasks.add_task(
                _update_cache_hit_metrics,
                session_id=result.session_id
            )
        
        return ChatMessageResponse(
            response=result.response,
            session_id=result.session_id,
            requires_auth=result.requires_auth,
            auth_methods=result.auth_methods,
            conversation_id=result.conversation_id,
            cached=result.cached,
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing message"
        )


@router.get("/conversations", response_model=ConversationListResponse)
async def get_conversations(
    skip: int = 0,
    limit: int = 20,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve conversation history.
    
    Returns a paginated list of conversations, optionally filtered by user_id.
    """
    try:
        conversation_service = ConversationService(db)
        
        conversations = conversation_service.get_conversations(
            skip=skip,
            limit=limit,
            user_id=user_id,
            requester_id=current_user.id if current_user else None
        )
        
        return ConversationListResponse(
            conversations=conversations,
            total=len(conversations),
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error retrieving conversations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving conversation history"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve a specific conversation with all messages.
    """
    try:
        conversation_service = ConversationService(db)
        
        conversation = conversation_service.get_conversation_by_id(
            conversation_id=conversation_id,
            requester_id=current_user.id if current_user else None
        )
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        return ConversationResponse.from_orm(conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving conversation"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a conversation and all associated messages.
    """
    try:
        conversation_service = ConversationService(db)
        
        success = conversation_service.delete_conversation(
            conversation_id=conversation_id,
            requester_id=current_user.id if current_user else None
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or access denied"
            )
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error deleting conversation"
        )


@router.post("/conversations/{conversation_id}/resolve")
async def resolve_conversation(
    conversation_id: UUID,
    resolved: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Mark a conversation as resolved or unresolved.
    """
    try:
        conversation_service = ConversationService(db)
        
        conversation = conversation_service.update_conversation_status(
            conversation_id=conversation_id,
            resolved=resolved,
            requester_id=current_user.id if current_user else None
        )
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or access denied"
            )
        
        return {
            "message": f"Conversation marked as {'resolved' if resolved else 'unresolved'}",
            "conversation_id": conversation_id,
            "resolved": resolved
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error updating conversation status"
        )


async def _update_conversation_metrics(conversation_id: UUID, processing_time: int):
    """Background task to update conversation metrics."""
    try:
        # This would update analytics/metrics in the background
        logger.info(f"Updated metrics for conversation {conversation_id}: {processing_time}ms")
    except Exception as e:
        logger.error(f"Error updating conversation metrics: {str(e)}")


async def _update_cache_hit_metrics(session_id: str):
    """Background task to update cache hit metrics."""
    try:
        # This would update cache hit analytics in the background
        logger.info(f"Cache hit recorded for session {session_id}")
    except Exception as e:
        logger.error(f"Error updating cache metrics: {str(e)}")
