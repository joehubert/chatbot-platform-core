"""
Alembic environment configuration for the Turnkey AI Chatbot platform.

This module configures Alembic for database migrations, including
model imports, connection settings, and migration context setup.
"""

import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import all models to ensure they are registered with SQLAlchemy
from app.models import Conversation, Message, Document, User, AuthToken

# Import the base class to get metadata
from app.models.base import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Logger for this module
logger = logging.getLogger(__name__)

# Set the target metadata for 'autogenerate' support
target_metadata = Base.metadata

# Other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url() -> str:
    """
    Get database URL from environment variable or config.

    Returns:
        Database URL string
    """
    # First try to get from environment variable
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        logger.info("Using DATABASE_URL from environment variable")
        return database_url

    # Fall back to config file
    database_url = config.get_main_option("sqlalchemy.url")
    if database_url:
        logger.info("Using DATABASE_URL from alembic.ini config file")
        return database_url
    
    # If neither source provides a URL, raise an error
    raise ValueError("DATABASE_URL must be set either as environment variable or in alembic.ini")


def include_name(name, type_, parent_names):
    """
    Filter function to include/exclude tables from migrations.

    Args:
        name: Name of the object
        type_: Type of object (table, column, etc.)
        parent_names: Parent object names

    Returns:
        True if object should be included in migrations
    """
    # Include all tables by default
    if type_ == "table":
        # You can add logic here to exclude certain tables
        # For example, to exclude tables starting with 'temp_':
        # return not name.startswith('temp_')
        return True

    # Include all other objects (columns, indexes, etc.)
    return True


def compare_type(
    context, inspected_column, metadata_column, inspected_type, metadata_type
):
    """
    Custom type comparison function for Alembic.

    This function helps Alembic detect type changes more accurately,
    especially for PostgreSQL-specific types.

    Args:
        context: Migration context
        inspected_column: Column from database inspection
        metadata_column: Column from model metadata
        inspected_type: Type from database
        metadata_type: Type from model

    Returns:
        True if types are different and migration is needed
    """
    # Handle UUID type comparisons
    if hasattr(metadata_type, "python_type"):
        # Check for UUID types
        if str(metadata_type).startswith("UUID"):
            return not str(inspected_type).startswith("UUID")

    # Default comparison
    return context.impl.compare_type(inspected_column, metadata_column)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=include_name,
        compare_type=compare_type,
        render_as_batch=True,  # Enable batch mode for SQLite compatibility
        process_revision_directives=process_revision_directives,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Get database URL
    database_url = get_database_url()

    # Override the sqlalchemy.url in config
    config.set_main_option("sqlalchemy.url", database_url)

    # Create engine configuration
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = database_url

    # Create engine with connection pooling
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Use NullPool for better connection management
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=include_name,
            compare_type=compare_type,
            render_as_batch=True,  # Enable batch mode for SQLite compatibility
            # Enable constraint naming for better constraint management
            render_item=render_item,
            # Transaction per migration for better error handling
            transaction_per_migration=True,
            process_revision_directives=process_revision_directives,
        )

        with context.begin_transaction():
            context.run_migrations()


def render_item(type_, obj, autogen_context):
    """
    Custom rendering function for Alembic objects.

    This function customizes how certain database objects are rendered
    in migration scripts, ensuring consistent naming and formatting.

    Args:
        type_: Type of object being rendered
        obj: The object being rendered
        autogen_context: Autogeneration context

    Returns:
        Custom rendering or False to use default
    """
    # Custom constraint naming
    if type_ == "foreign_key":
        # Ensure foreign key constraints have consistent names
        if hasattr(obj, "name") and obj.name:
            return f"sa.ForeignKeyConstraint({obj.column_keys!r}, {[str(c) for c in obj.elements]!r}, name={obj.name!r})"

    # Custom index naming
    if type_ == "index":
        # Ensure indexes have consistent names
        if hasattr(obj, "name") and obj.name:
            return f"sa.Index({obj.name!r}, {[str(c) for c in obj.columns]!r})"

    # Use default rendering
    return False


def process_revision_directives(context, revision, directives):
    """
    Process revision directives to customize migration generation.

    This function can be used to modify the generated migration scripts,
    add custom headers, or implement advanced migration patterns.

    Args:
        context: Migration context
        revision: Revision being generated
        directives: List of migration directives
    """
    # Add custom header to migration files
    if hasattr(revision, "message") and revision.message:
        script = directives[0]
        if hasattr(script, "upgrade_ops") and script.upgrade_ops:
            # Add custom docstring to upgrade function
            script.upgrade_ops.ops.insert(
                0,
                f'    """\n    Migration: {revision.message}\n    Generated: {revision.revision_id}\n    """',
            )


# Main execution logic
if context.is_offline_mode():
    logger.info("Running migrations in offline mode")
    run_migrations_offline()
else:
    logger.info("Running migrations in online mode")
    run_migrations_online()