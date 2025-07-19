"""
Initial database schema

Revision ID: 001
Revises:
Create Date: 2024-01-15 10:00:00.000000

This migration creates the initial database schema for the Turnkey AI Chatbot platform,
including all core tables: users, conversations, messages, documents, and auth_tokens.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create initial database schema.

    This function creates all the core tables needed for the chatbot platform:
    - users: User information and contact details
    - conversations: Chat sessions and metadata
    - messages: Individual messages within conversations
    - documents: Knowledge base documents and processing status
    - auth_tokens: One-time authentication tokens
    """

    # Create users table
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="Unique identifier for the user",
        ),
        sa.Column(
            "mobile_number",
            sa.String(length=20),
            nullable=True,
            comment="User's mobile phone number for SMS authentication",
        ),
        sa.Column(
            "email",
            sa.String(length=255),
            nullable=True,
            comment="User's email address for email authentication",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Timestamp when user was first created",
        ),
        sa.Column(
            "last_authenticated",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of most recent successful authentication",
        ),
        sa.Column(
            "last_seen",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when user was last active",
        ),
        sa.Column(
            "authentication_count",
            sa.Integer(),
            nullable=False,
            comment="Total number of successful authentications",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            comment="Whether the user account is active",
        ),
        sa.Column(
            "preferred_contact_method",
            sa.String(length=10),
            nullable=True,
            comment="Preferred method for authentication (sms/email)",
        ),
        sa.Column(
            "timezone",
            sa.String(length=50),
            nullable=True,
            comment="User's timezone for scheduling and timestamps",
        ),
        sa.Column(
            "language_preference",
            sa.String(length=10),
            nullable=True,
            comment="Preferred language code (e.g., 'en', 'es')",
        ),
        sa.Column(
            "metadata",
            sa.Text(),
            nullable=True,
            comment="Additional user metadata stored as JSON",
        ),
        sa.Column(
            "notes",
            sa.Text(),
            nullable=True,
            comment="Administrative notes about the user",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("mobile_number"),
    )

    # Create indexes for users table
    op.create_index("idx_user_mobile_number", "users", ["mobile_number"], unique=False)
    op.create_index("idx_user_email", "users", ["email"], unique=False)
    op.create_index("idx_user_created_at", "users", ["created_at"], unique=False)
    op.create_index(
        "idx_user_last_authenticated", "users", ["last_authenticated"], unique=False
    )
    op.create_index("idx_user_is_active", "users", ["is_active"], unique=False)
    op.create_index(
        "idx_user_preferred_contact",
        "users",
        ["preferred_contact_method"],
        unique=False,
    )

    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="Unique identifier for the conversation",
        ),
        sa.Column(
            "session_id",
            sa.String(length=255),
            nullable=False,
            comment="Session identifier for tracking user sessions",
        ),
        sa.Column(
            "user_identifier",
            sa.String(length=255),
            nullable=True,
            comment="Optional user identifier (email or phone number)",
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Timestamp when conversation began",
        ),
        sa.Column(
            "ended_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when conversation ended",
        ),
        sa.Column(
            "resolved",
            sa.Boolean(),
            nullable=False,
            comment="Whether the conversation was successfully resolved",
        ),
        sa.Column(
            "resolution_attempts",
            sa.Integer(),
            nullable=False,
            comment="Number of attempts made to resolve the query",
        ),
        sa.Column(
            "authenticated",
            sa.Boolean(),
            nullable=False,
            comment="Whether user was authenticated during conversation",
        ),
        sa.Column(
            "context_data",
            sa.Text(),
            nullable=True,
            comment="Additional context information stored as JSON",
        ),
        sa.Column(
            "user_agent",
            sa.String(length=512),
            nullable=True,
            comment="User agent string from the client",
        ),
        sa.Column(
            "ip_address",
            sa.String(length=45),
            nullable=True,
            comment="Client IP address for analytics and security",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for conversations table
    op.create_index(
        "idx_conversation_session_id", "conversations", ["session_id"], unique=False
    )
    op.create_index(
        "idx_conversation_user_identifier",
        "conversations",
        ["user_identifier"],
        unique=False,
    )
    op.create_index(
        "idx_conversation_started_at", "conversations", ["started_at"], unique=False
    )
    op.create_index(
        "idx_conversation_resolved", "conversations", ["resolved"], unique=False
    )
    op.create_index(
        "idx_conversation_authenticated",
        "conversations",
        ["authenticated"],
        unique=False,
    )
    op.create_index(
        "idx_conversation_session_started",
        "conversations",
        ["session_id", "started_at"],
        unique=False,
    )

    # Create messages table
    op.create_table(
        "messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="Unique identifier for the message",
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="Foreign key to the parent conversation",
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
            comment="The actual message content/text",
        ),
        sa.Column(
            "role",
            sa.Enum("user", "assistant", "system", name="messagerole"),
            nullable=False,
            comment="Role of the message sender (user/assistant/system)",
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="When the message was created",
        ),
        sa.Column(
            "llm_model_used",
            sa.String(length=100),
            nullable=True,
            comment="LLM model used to generate response (for assistant messages)",
        ),
        sa.Column(
            "cached",
            sa.Boolean(),
            nullable=False,
            comment="Whether the response was served from cache",
        ),
        sa.Column(
            "processing_time_ms",
            sa.Integer(),
            nullable=True,
            comment="Time taken to process/generate the message in milliseconds",
        ),
        sa.Column(
            "tokens_used",
            sa.Integer(),
            nullable=True,
            comment="Number of tokens consumed for LLM calls",
        ),
        sa.Column(
            "confidence_score",
            sa.Float(),
            nullable=True,
            comment="Confidence score for the response (0.0-1.0)",
        ),
        sa.Column(
            "requires_clarification",
            sa.Boolean(),
            nullable=False,
            comment="Whether the message needs clarification",
        ),
        sa.Column(
            "metadata",
            sa.Text(),
            nullable=True,
            comment="Additional metadata stored as JSON",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for messages table
    op.create_index(
        "idx_message_conversation_id", "messages", ["conversation_id"], unique=False
    )
    op.create_index("idx_message_role", "messages", ["role"], unique=False)
    op.create_index("idx_message_timestamp", "messages", ["timestamp"], unique=False)
    op.create_index("idx_message_cached", "messages", ["cached"], unique=False)
    op.create_index("idx_message_llm_model_used", "messages", ["llm_model_used"], unique=False)
    op.create_index(
        "idx_message_conversation_timestamp",
        "messages",
        ["conversation_id", "timestamp"],
        unique=False,
    )
    op.create_index(
        "idx_message_role_timestamp", "messages", ["role", "timestamp"], unique=False
    )

    # Create documents table
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="Unique identifier for the document",
        ),
        sa.Column(
            "filename",
            sa.String(length=255),
            nullable=False,
            comment="Original filename of the uploaded document",
        ),
        sa.Column(
            "content_type",
            sa.String(length=100),
            nullable=False,
            comment="MIME type of the document",
        ),
        sa.Column(
            "file_size",
            sa.BigInteger(),
            nullable=False,
            comment="Size of the file in bytes",
        ),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Timestamp when document was uploaded",
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when document expires (set by admin)",
        ),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when processing completed",
        ),
        sa.Column(
            "status",
            sa.Enum(
                "uploaded",
                "processing",
                "processed",
                "failed",
                "expired",
                name="documentstatus",
            ),
            nullable=False,
            comment="Current processing status",
        ),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=False,
            comment="Number of chunks created from this document",
        ),
        sa.Column(
            "vector_ids",
            postgresql.ARRAY(sa.String()),
            nullable=True,
            comment="List of vector IDs in the vector database",
        ),
        sa.Column(
            "content_hash",
            sa.String(length=64),
            nullable=True,
            comment="Hash of the document content for deduplication",
        ),
        sa.Column(
            "metadata",
            sa.Text(),
            nullable=True,
            comment="Additional metadata stored as JSON",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Error message if processing failed",
        ),
        sa.Column(
            "processing_duration_ms",
            sa.Integer(),
            nullable=True,
            comment="Time taken to process the document in milliseconds",
        ),
        sa.Column(
            "admin_notes",
            sa.Text(),
            nullable=True,
            comment="Administrative notes about the document",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash"),
    )

    # Create indexes for documents table
    op.create_index("idx_document_filename", "documents", ["filename"], unique=False)
    op.create_index(
        "idx_document_content_type", "documents", ["content_type"], unique=False
    )
    op.create_index(
        "idx_document_uploaded_at", "documents", ["uploaded_at"], unique=False
    )
    op.create_index(
        "idx_document_expires_at", "documents", ["expires_at"], unique=False
    )
    op.create_index("idx_document_status", "documents", ["status"], unique=False)
    op.create_index(
        "idx_document_content_hash", "documents", ["content_hash"], unique=False
    )
    op.create_index(
        "idx_document_status_uploaded",
        "documents",
        ["status", "uploaded_at"],
        unique=False,
    )
    op.create_index(
        "idx_document_expires_status",
        "documents",
        ["expires_at", "status"],
        unique=False,
    )

    # Create auth_tokens table
    op.create_table(
        "auth_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="Unique identifier for the token",
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="Foreign key to the user this token belongs to",
        ),
        sa.Column(
            "token",
            sa.String(length=64),
            nullable=False,
            comment="The hashed token string for security",
        ),
        sa.Column(
            "session_id",
            sa.String(length=255),
            nullable=False,
            comment="Session ID this token is associated with",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Timestamp when token was created",
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Timestamp when the token expires",
        ),
        sa.Column(
            "used_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when token was used (if applicable)",
        ),
        sa.Column(
            "used",
            sa.Boolean(),
            nullable=False,
            comment="Whether the token has been used",
        ),
        sa.Column(
            "delivery_method",
            sa.String(length=10),
            nullable=False,
            comment="How the token was delivered (sms/email)",
        ),
        sa.Column(
            "delivery_address",
            sa.String(length=255),
            nullable=False,
            comment="Where the token was sent (phone number or email)",
        ),
        sa.Column(
            "attempts",
            sa.Integer(),
            nullable=False,
            comment="Number of verification attempts made",
        ),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            nullable=False,
            comment="Maximum allowed verification attempts",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token"),
    )

    # Create indexes for auth_tokens table
    op.create_index("idx_auth_token_user_id", "auth_tokens", ["user_id"], unique=False)
    op.create_index("idx_auth_token_token", "auth_tokens", ["token"], unique=False)
    op.create_index(
        "idx_auth_token_session_id", "auth_tokens", ["session_id"], unique=False
    )
    op.create_index(
        "idx_auth_token_expires_at", "auth_tokens", ["expires_at"], unique=False
    )
    op.create_index("idx_auth_token_used", "auth_tokens", ["used"], unique=False)
    op.create_index(
        "idx_auth_token_delivery_method",
        "auth_tokens",
        ["delivery_method"],
        unique=False,
    )
    op.create_index(
        "idx_auth_token_user_session",
        "auth_tokens",
        ["user_id", "session_id"],
        unique=False,
    )
    op.create_index(
        "idx_auth_token_expires_used",
        "auth_tokens",
        ["expires_at", "used"],
        unique=False,
    )


def downgrade() -> None:
    """
    Drop all tables created in the upgrade.

    This function removes all tables and indexes created during the upgrade,
    effectively reverting the database to its pre-migration state.
    """

    # Drop auth_tokens table and its indexes
    op.drop_index("idx_auth_token_expires_used", table_name="auth_tokens")
    op.drop_index("idx_auth_token_user_session", table_name="auth_tokens")
    op.drop_index("idx_auth_token_delivery_method", table_name="auth_tokens")
    op.drop_index("idx_auth_token_used", table_name="auth_tokens")
    op.drop_index("idx_auth_token_expires_at", table_name="auth_tokens")
    op.drop_index("idx_auth_token_session_id", table_name="auth_tokens")
    op.drop_index("idx_auth_token_token", table_name="auth_tokens")
    op.drop_index("idx_auth_token_user_id", table_name="auth_tokens")
    op.drop_table("auth_tokens")

    # Drop documents table and its indexes
    op.drop_index("idx_document_expires_status", table_name="documents")
    op.drop_index("idx_document_status_uploaded", table_name="documents")
    op.drop_index("idx_document_content_hash", table_name="documents")
    op.drop_index("idx_document_status", table_name="documents")
    op.drop_index("idx_document_expires_at", table_name="documents")
    op.drop_index("idx_document_uploaded_at", table_name="documents")
    op.drop_index("idx_document_content_type", table_name="documents")
    op.drop_index("idx_document_filename", table_name="documents")
    op.drop_table("documents")

    # Drop messages table and its indexes
    op.drop_index("idx_message_role_timestamp", table_name="messages")
    op.drop_index("idx_message_conversation_timestamp", table_name="messages")
    op.drop_index("idx_message_llm_model_used", table_name="messages")
    op.drop_index("idx_message_cached", table_name="messages")
    op.drop_index("idx_message_timestamp", table_name="messages")
    op.drop_index("idx_message_role", table_name="messages")
    op.drop_index("idx_message_conversation_id", table_name="messages")
    op.drop_table("messages")

    # Drop conversations table and its indexes
    op.drop_index("idx_conversation_session_started", table_name="conversations")
    op.drop_index("idx_conversation_authenticated", table_name="conversations")
    op.drop_index("idx_conversation_resolved", table_name="conversations")
    op.drop_index("idx_conversation_started_at", table_name="conversations")
    op.drop_index("idx_conversation_user_identifier", table_name="conversations")
    op.drop_index("idx_conversation_session_id", table_name="conversations")
    op.drop_table("conversations")

    # Drop users table and its indexes
    op.drop_index("idx_user_preferred_contact", table_name="users")
    op.drop_index("idx_user_is_active", table_name="users")
    op.drop_index("idx_user_last_authenticated", table_name="users")
    op.drop_index("idx_user_created_at", table_name="users")
    op.drop_index("idx_user_email", table_name="users")
    op.drop_index("idx_user_mobile_number", table_name="users")
    op.drop_table("users")

    # Drop custom enum types
    op.execute("DROP TYPE IF EXISTS messagerole")
    op.execute("DROP TYPE IF EXISTS documentstatus")
