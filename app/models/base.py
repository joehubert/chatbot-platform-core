"""
Database base configuration for the Turnkey AI Chatbot platform.

This module provides the shared SQLAlchemy base class and common
database configuration used across all models.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped

# Create the shared declarative base
Base = declarative_base()


class TimestampMixin:
    """
    Mixin class that provides created_at and updated_at timestamps.

    This can be used by models that need automatic timestamp tracking.
    """

    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Timestamp when record was created",
    )

    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="Timestamp when record was last updated",
    )


class UUIDMixin:
    """
    Mixin class that provides a UUID primary key.

    This can be used by models that need UUID primary keys.
    """

    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the record",
    )


# Common database metadata
DATABASE_SCHEMA_VERSION = "1.0.0"
