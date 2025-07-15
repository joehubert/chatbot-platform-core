"""
Knowledge Base API endpoints for document management and retrieval.
Handles document upload, processing, and management operations.
"""

import json
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db
from app.core.config import get_settings
from app.models.document import Document
from app.models.user import User
from app.schemas.knowledge import (
    DocumentUpload,
    DocumentResponse,
    DocumentList,
    DocumentStatus,
    ProcessingStatus,
    DocumentUpdate,
)
from app.services.document_processor import DocumentProcessor
from app.services.knowledge_base import KnowledgeBaseService
from app.services.vector_db import VectorDBService
from app.services.vector_db import create_vector_db_service


logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Initialize services
# TODO: see related todo in depencencies.py for vector DB service
vector_service = create_vector_db_service(
    db_type="chroma",  # or "pinecone", "weaviate", "qdrant"
    connection_params={
        "host": "localhost",
        "port": 8000,
        # other connection parameters
    },
    index_name="your_index_name",
    dimension=1536,
    similarity_metric="cosine"
)
document_processor = DocumentProcessor()
knowledge_service = KnowledgeBaseService(
    vector_db_service=vector_service,
    document_processor=document_processor
)



@router.post("/documents", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form("general"),
    expires_at: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON string of tags
    metadata: Optional[str] = Form(None),  # JSON string
    # ... rest of implementation
):
    # Add file_size calculation
    content = await file.read()
    file_size = len(content)

    # Parse tags and metadata from JSON strings
    parsed_tags = json.loads(tags) if tags else []
    parsed_metadata = json.loads(metadata) if metadata else {}

    # Create proper DocumentUpload object
    document_data = DocumentUpload(
        filename=file.filename,
        content_type=file.content_type,
        file_size=file_size,
        title=title or file.filename,
        category=category,
        tags=parsed_tags,
        expires_at=expires_at,
        metadata=parsed_metadata,
    )


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    category: Optional[str] = None,
    status_filter: Optional[List[ProcessingStatus]] = None,
    # ... existing parameters
):
    # Return DocumentList with pagination
    total_count = await knowledge_service.count_documents(...)
    documents = await knowledge_service.list_documents(...)

    return DocumentList(
        documents=documents,
        total_count=total_count,
        page=(skip // limit) + 1,
        page_size=limit,
        total_pages=(total_count + limit - 1) // limit,
        filters_applied={
            "category": category,
            "status_filter": status_filter,
        },
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get details of a specific document.
    """
    try:
        document = await knowledge_service.get_document(document_id=document_id, db=db)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document",
        )


@router.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    document_update: DocumentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update document metadata.

    Allows updating title, category, and expiration date.
    """
    try:
        document = await knowledge_service.update_document(
            document_id=document_id, document_update=document_update, db=db
        )

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        logger.info(f"Document updated successfully: {document_id}")
        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document",
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a document and remove from vector database.

    This operation cannot be undone.
    """
    try:
        # Get document first to check if it exists
        document = await knowledge_service.get_document(document_id=document_id, db=db)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        # Delete document and associated vectors
        await knowledge_service.delete_document(document_id=document_id, db=db)

        logger.info(f"Document deleted successfully: {document_id}")
        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )


@router.post("/documents/{document_id}/reprocess")
async def reprocess_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Reprocess a document to update its embeddings.

    Useful when changing embedding models or processing settings.
    """
    try:
        document = await knowledge_service.get_document(document_id=document_id, db=db)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        # Trigger reprocessing
        result = await knowledge_service.reprocess_document(
            document_id=document_id, db=db
        )

        logger.info(f"Document reprocessing initiated: {document_id}")
        return {"message": "Document reprocessing initiated", "status": result.status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reprocess document",
        )


@router.get("/search")
async def search_knowledge_base(
    query: str,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Search the knowledge base using semantic similarity.

    Returns relevant document chunks with similarity scores.
    """
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty"
            )

        results = await knowledge_service.search_knowledge_base(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            category=category,
            db=db,
        )

        return {"query": query, "results": results, "total_results": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search knowledge base",
        )


@router.get("/stats")
async def get_knowledge_base_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get statistics about the knowledge base.

    Returns document counts, processing status, and storage information.
    """
    try:
        stats = await knowledge_service.get_knowledge_base_stats(db=db)

        return stats

    except Exception as e:
        logger.error(f"Error retrieving knowledge base stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics",
        )


@router.post("/rebuild-index")
async def rebuild_vector_index(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Rebuild the entire vector database index.

    This is a maintenance operation that reprocesses all documents.
    Use with caution as it may take considerable time.
    """
    try:
        # This is a potentially long-running operation
        # In production, this should be handled as a background task
        result = await knowledge_service.rebuild_vector_index(db=db)

        logger.info("Vector index rebuild initiated")
        return {
            "message": "Vector index rebuild initiated",
            "job_id": result.get("job_id"),
            "estimated_time": result.get("estimated_time"),
        }

    except Exception as e:
        logger.error(f"Error rebuilding vector index: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rebuild vector index",
        )


@router.delete("/expired")
async def cleanup_expired_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Remove expired documents from the knowledge base.

    This is a maintenance operation that cleans up documents
    that have passed their expiration date.
    """
    try:
        result = await knowledge_service.cleanup_expired_documents(db=db)

        logger.info(f"Cleanup completed: {result['deleted_count']} documents removed")
        return {
            "message": "Expired documents cleanup completed",
            "deleted_count": result["deleted_count"],
            "freed_space": result.get("freed_space", "Unknown"),
        }

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup expired documents",
        )


@router.get("/documents/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the processing status of a specific document.
    """
    try:
        status = await knowledge_service.get_document_status(
            document_id=document_id, db=db
        )
        return status

    except Exception as e:
        logger.error(f"Error retrieving document status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document status",
        )
