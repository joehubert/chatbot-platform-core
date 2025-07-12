"""
Document model for the Turnkey AI Chatbot platform.

This module defines the Document model that represents uploaded documents
in the knowledge base, including metadata, processing status, and vector
database integration.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Integer,
    Text,
    Index,
    BigInteger,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSON
from sqlalchemy.orm import Mapped
from .base import Base


class DocumentStatus(str, Enum):
    """
    Enumeration of document processing statuses.

    Values:
        UPLOADED: Document has been uploaded but not processed
        PROCESSING: Document is currently being processed
        PROCESSED: Document has been successfully processed and indexed
        FAILED: Document processing failed
        EXPIRED: Document has passed its expiration date
    """

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


class Document(Base):
    """
    Document model representing uploaded knowledge base documents.

    This model tracks documents uploaded for the chatbot's knowledge base,
    including their processing status, vector database integration,
    and expiration management.

    Attributes:
        id: Unique identifier for the document
        filename: Original filename of the uploaded document
        content_type: MIME type of the document
        file_size: Size of the file in bytes
        uploaded_at: Timestamp when document was uploaded
        expires_at: Timestamp when document expires (set by admin)
        processed_at: Timestamp when processing completed
        status: Current processing status
        chunk_count: Number of chunks created from this document
        vector_ids: List of vector IDs in the vector database
        content_hash: Hash of the document content for deduplication
        metadata: Additional metadata stored as JSON
        error_message: Error message if processing failed
        processing_duration_ms: Time taken to process the document
        admin_notes: Administrative notes about the document
    """

    __tablename__ = "documents"

    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the document",
    )

    # File information
    filename: Mapped[str] = Column(
        String(255),
        nullable=False,
        comment="Original filename of the uploaded document",
    )

    content_type: Mapped[str] = Column(
        String(100), nullable=False, comment="MIME type of the document"
    )

    file_size: Mapped[int] = Column(
        BigInteger, nullable=False, comment="Size of the file in bytes"
    )

    # Timestamps
    uploaded_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
        comment="Timestamp when document was uploaded",
    )

    expires_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Timestamp when document expires (set by admin)",
    )

    processed_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when processing completed",
    )

    # Processing status
    status: Mapped[DocumentStatus] = Column(
        SQLEnum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.UPLOADED,
        index=True,
        comment="Current processing status",
    )

    # Processing results
    chunk_count: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of chunks created from this document",
    )

    vector_ids: Mapped[Optional[List[str]]] = Column(
        ARRAY(String),
        nullable=True,
        comment="List of vector IDs in the vector database",
    )

    # Content validation
    content_hash: Mapped[Optional[str]] = Column(
        String(64),  # SHA-256 hash
        nullable=True,
        unique=True,
        comment="Hash of the document content for deduplication",
    )

    # Additional data
    data: Mapped[Optional[dict]] = Column(
        JSON, nullable=True, comment="Additional data stored as JSON"
    )

    error_message: Mapped[Optional[str]] = Column(
        Text, nullable=True, comment="Error message if processing failed"
    )

    processing_duration_ms: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True,
        comment="Time taken to process the document in milliseconds",
    )

    # Administrative fields
    admin_notes: Mapped[Optional[str]] = Column(
        Text, nullable=True, comment="Administrative notes about the document"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_document_filename", "filename"),
        Index("idx_document_content_type", "content_type"),
        Index("idx_document_uploaded_at", "uploaded_at"),
        Index("idx_document_expires_at", "expires_at"),
        Index("idx_document_status", "status"),
        Index("idx_document_content_hash", "content_hash"),
        Index("idx_document_status_uploaded", "status", "uploaded_at"),
        Index("idx_document_expires_status", "expires_at", "status"),
    )

    def __repr__(self) -> str:
        """String representation of the document."""
        return (
            f"<Document(id={self.id}, filename='{self.filename}', "
            f"status={self.status.value}, uploaded_at={self.uploaded_at})>"
        )

    @property
    def is_processed(self) -> bool:
        """Check if document has been successfully processed."""
        return self.status == DocumentStatus.PROCESSED

    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        return self.status == DocumentStatus.PROCESSING

    @property
    def has_failed(self) -> bool:
        """Check if document processing failed."""
        return self.status == DocumentStatus.FAILED

    @property
    def is_expired(self) -> bool:
        """
        Check if document has expired based on expiration date.

        Returns:
            True if document has expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)

    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """
        Get processing duration in seconds.

        Returns:
            Processing duration in seconds, or None if not recorded
        """
        if self.processing_duration_ms is not None:
            return self.processing_duration_ms / 1000.0
        return None

    @property
    def days_until_expiration(self) -> Optional[int]:
        """
        Calculate days until expiration.

        Returns:
            Number of days until expiration, None if no expiration set,
            negative number if already expired
        """
        if self.expires_at is None:
            return None

        delta = self.expires_at - datetime.utcnow()
        return delta.days

    @property
    def file_extension(self) -> str:
        """Extract file extension from filename."""
        if "." not in self.filename:
            return ""
        return self.filename.rsplit(".", 1)[1].lower()

    def mark_processing(self) -> None:
        """Mark document as currently being processed."""
        self.status = DocumentStatus.PROCESSING

    def mark_processed(
        self,
        chunk_count: int,
        vector_ids: List[str],
        processing_duration_ms: Optional[int] = None,
    ) -> None:
        """
        Mark document as successfully processed.

        Args:
            chunk_count: Number of chunks created
            vector_ids: List of vector database IDs
            processing_duration_ms: Processing time in milliseconds
        """
        self.status = DocumentStatus.PROCESSED
        self.processed_at = datetime.utcnow()
        self.chunk_count = chunk_count
        self.vector_ids = vector_ids
        if processing_duration_ms is not None:
            self.processing_duration_ms = processing_duration_ms
        self.error_message = None  # Clear any previous errors

    def mark_failed(self, error_message: str) -> None:
        """
        Mark document processing as failed.

        Args:
            error_message: Description of the error that occurred
        """
        self.status = DocumentStatus.FAILED
        self.error_message = error_message
        self.processed_at = datetime.utcnow()

    def mark_expired(self) -> None:
        """Mark document as expired."""
        self.status = DocumentStatus.EXPIRED

    def set_expiration(self, expires_at: datetime) -> None:
        """
        Set the expiration date for the document.

        Args:
            expires_at: When the document should expire
        """
        self.expires_at = expires_at

    def extend_expiration(self, days: int) -> None:
        """
        Extend the expiration date by a number of days.

        Args:
            days: Number of days to extend expiration
        """
        if self.expires_at is None:
            # If no expiration set, set it from now
            from datetime import timedelta

            self.expires_at = datetime.utcnow() + timedelta(days=days)
        else:
            from datetime import timedelta

            self.expires_at += timedelta(days=days)

    def can_be_deleted(self) -> bool:
        """
        Check if document can be safely deleted.

        Returns:
            True if document is expired or failed and can be removed
        """
        return (
            self.status in [DocumentStatus.EXPIRED, DocumentStatus.FAILED]
            or self.is_expired
        )

    def get_vector_count(self) -> int:
        """
        Get the number of vectors stored for this document.

        Returns:
            Number of vector IDs, or 0 if none
        """
        return len(self.vector_ids) if self.vector_ids else 0

    @classmethod
    def get_supported_content_types(cls) -> List[str]:
        """
        Get list of supported document content types.

        Returns:
            List of supported MIME types
        """
        return [
            "application/pdf",
            "text/plain",
            "text/markdown",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]

    @classmethod
    def is_supported_content_type(cls, content_type: str) -> bool:
        """
        Check if a content type is supported.

        Args:
            content_type: MIME type to check

        Returns:
            True if content type is supported
        """
        return content_type in cls.get_supported_content_types()
