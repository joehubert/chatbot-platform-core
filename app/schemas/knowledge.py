"""
Knowledge Base API Schemas

Pydantic schemas for knowledge base management including document upload,
processing status, and content management operations.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""

    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MARKDOWN = "md"
    HTML = "html"


class ProcessingStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class DocumentCategory(str, Enum):
    """Document categories for organization"""

    GENERAL = "general"
    PRODUCT = "product"
    POLICY = "policy"
    FAQ = "faq"
    TECHNICAL = "technical"
    LEGAL = "legal"
    SUPPORT = "support"


class DocumentUpload(BaseModel):
    """Schema for document upload request"""

    filename: str = Field(
        ..., min_length=1, max_length=255, description="Original filename"
    )
    content_type: str = Field(..., description="MIME type of the document")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    category: DocumentCategory = Field(
        default=DocumentCategory.GENERAL, description="Document category"
    )
    title: Optional[str] = Field(
        None, max_length=500, description="Document title override"
    )
    description: Optional[str] = Field(
        None, max_length=2000, description="Document description"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for document organization"
    )
    expires_at: Optional[datetime] = Field(None, description="Document expiration date")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        # Check for valid filename
        if not v or v.isspace():
            raise ValueError("Filename cannot be empty")

        # Extract extension and validate
        if "." not in v:
            raise ValueError("Filename must have an extension")

        extension = v.split(".")[-1].lower()
        allowed_extensions = {e.value for e in DocumentType}
        if extension not in allowed_extensions:
            raise ValueError(
                f"File type .{extension} not supported. Allowed: {allowed_extensions}"
            )

        return v

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v):
        allowed_types = {
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/markdown",
            "text/html",
        }
        if v not in allowed_types:
            raise ValueError(f"Content type {v} not supported")
        return v

    @field_validator("file_size")
    @classmethod
    def validate_file_size(cls, v):
        max_size = 50 * 1024 * 1024  # 50MB default limit
        if v > max_size:
            raise ValueError(f"File size exceeds maximum limit of {max_size} bytes")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")

        for tag in v:
            if not tag or len(tag) > 50:
                raise ValueError("Tags must be non-empty and max 50 characters")

        return list(set(v))  # Remove duplicates

    class Config:
        schema_extra = {
            "example": {
                "filename": "user_manual.pdf",
                "content_type": "application/pdf",
                "file_size": 1048576,
                "category": "product",
                "title": "Product User Manual v2.1",
                "description": "Complete user manual for product features and troubleshooting",
                "tags": ["manual", "troubleshooting", "features"],
                "expires_at": "2025-12-31T23:59:59Z",
                "metadata": {"version": "2.1", "author": "Technical Writing Team"},
            }
        }


class DocumentUpdate(BaseModel):
    """Schema for document update operations"""

    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    category: Optional[DocumentCategory] = None
    tags: Optional[List[str]] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Schema for document upload response"""

    document_id: UUID = Field(..., description="Unique document identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    chunks_created: Optional[int] = Field(
        None, description="Number of text chunks created"
    )
    processing_time_ms: Optional[int] = Field(
        None, description="Processing time in milliseconds"
    )
    expiration_date: Optional[datetime] = Field(
        None, description="Document expiration date"
    )
    vector_count: Optional[int] = Field(None, description="Number of vectors generated")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "processing",
                "message": "Document uploaded successfully and processing started",
                "chunks_created": None,
                "processing_time_ms": None,
                "expiration_date": "2025-12-31T23:59:59Z",
                "vector_count": None,
            }
        }


class DocumentInfo(BaseModel):
    """Schema for document information"""

    id: UUID = Field(..., description="Document unique identifier")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")
    category: DocumentCategory = Field(..., description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    status: ProcessingStatus = Field(..., description="Current status")
    chunk_count: int = Field(default=0, description="Number of text chunks")
    vector_count: int = Field(default=0, description="Number of vectors in database")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    access_count: int = Field(default=0, description="Number of times accessed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "user_manual.pdf",
                "title": "Product User Manual v2.1",
                "description": "Complete user manual for product features",
                "content_type": "application/pdf",
                "file_size": 1048576,
                "category": "product",
                "tags": ["manual", "troubleshooting"],
                "uploaded_at": "2024-01-15T10:00:00Z",
                "expires_at": "2025-12-31T23:59:59Z",
                "status": "completed",
                "chunk_count": 45,
                "vector_count": 45,
                "last_accessed": "2024-01-15T15:30:00Z",
                "access_count": 12,
                "metadata": {"version": "2.1"},
            }
        }


class DocumentList(BaseModel):
    """Schema for document list response"""

    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total_count: int = Field(..., ge=0, description="Total number of documents")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Applied filters"
    )

    class Config:
        schema_extra = {
            "example": {
                "documents": [],
                "total_count": 25,
                "page": 1,
                "page_size": 20,
                "total_pages": 2,
                "filters_applied": {"category": "product", "status": "completed"},
            }
        }


class DocumentStatus(BaseModel):
    """Schema for document processing status"""

    document_id: UUID = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress_percentage: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Processing progress"
    )
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(
        None, description="Processing completion time"
    )
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )

    class Config:
        schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "processing",
                "progress_percentage": 65.5,
                "current_step": "Generating embeddings",
                "error_message": None,
                "started_at": "2024-01-15T10:00:00Z",
                "completed_at": None,
                "estimated_completion": "2024-01-15T10:05:00Z",
            }
        }


class DocumentSearch(BaseModel):
    """Schema for document search request"""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    categories: Optional[List[DocumentCategory]] = Field(
        None, description="Filter by categories"
    )
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    status_filter: Optional[List[ProcessingStatus]] = None
    date_from: Optional[datetime] = Field(
        None, description="Filter by upload date from"
    )
    date_to: Optional[datetime] = Field(None, description="Filter by upload date to")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold for search"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "troubleshooting network issues",
                "categories": ["technical", "support"],
                "tags": ["network", "troubleshooting"],
                "status_filter": ["completed"],
                "date_from": "2024-01-01T00:00:00Z",
                "date_to": "2024-12-31T23:59:59Z",
                "limit": 10,
                "similarity_threshold": 0.8,
            }
        }


class DocumentSearchResult(BaseModel):
    """Schema for document search result"""

    document_id: UUID = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    title: Optional[str] = Field(None, description="Document title")
    category: DocumentCategory = Field(..., description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score for query"
    )
    matched_chunks: List[str] = Field(
        default_factory=list, description="Relevant text chunks"
    )
    upload_date: datetime = Field(..., description="Document upload date")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "network_troubleshooting.pdf",
                "title": "Network Troubleshooting Guide",
                "category": "technical",
                "tags": ["network", "troubleshooting", "guide"],
                "relevance_score": 0.85,
                "matched_chunks": [
                    "To troubleshoot network connectivity issues, first check...",
                    "Common network problems include DNS resolution failures...",
                ],
                "upload_date": "2024-01-10T14:30:00Z",
            }
        }


class DocumentSearchResponse(BaseModel):
    """Schema for document search response"""

    results: List[DocumentSearchResult] = Field(..., description="Search results")
    total_results: int = Field(
        ..., ge=0, description="Total number of matching documents"
    )
    query: str = Field(..., description="Original search query")
    search_time_ms: int = Field(
        ..., ge=0, description="Search execution time in milliseconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "results": [],
                "total_results": 5,
                "query": "troubleshooting network issues",
                "search_time_ms": 125,
            }
        }


class BulkDocumentOperation(BaseModel):
    """Schema for bulk document operations"""

    document_ids: List[UUID] = Field(
        ..., min_items=1, max_items=100, description="List of document IDs"
    )
    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Operation parameters"
    )

    @field_validator("operation")
    def validate_operation(cls, v):
        allowed_operations = {
            "delete",
            "reprocess",
            "update_category",
            "update_tags",
            "extend_expiration",
        }
        if v not in allowed_operations:
            raise ValueError(f"Operation must be one of: {allowed_operations}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "document_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "987fcdeb-51a2-43d1-b789-012345678900",
                ],
                "operation": "update_category",
                "parameters": {"category": "technical"},
            }
        }


class BulkOperationResponse(BaseModel):
    """Schema for bulk operation response"""

    operation_id: UUID = Field(..., description="Unique operation identifier")
    operation: str = Field(..., description="Operation performed")
    total_documents: int = Field(..., ge=0, description="Total documents in operation")
    successful: int = Field(
        default=0, ge=0, description="Successfully processed documents"
    )
    failed: int = Field(default=0, ge=0, description="Failed documents")
    status: str = Field(..., description="Operation status")
    started_at: datetime = Field(..., description="Operation start time")
    completed_at: Optional[datetime] = Field(
        None, description="Operation completion time"
    )
    errors: List[str] = Field(
        default_factory=list, description="Error messages for failed operations"
    )

    class Config:
        schema_extra = {
            "example": {
                "operation_id": "456e7890-e12b-34c5-d678-901234567890",
                "operation": "update_category",
                "total_documents": 2,
                "successful": 2,
                "failed": 0,
                "status": "completed",
                "started_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:01:30Z",
                "errors": [],
            }
        }
