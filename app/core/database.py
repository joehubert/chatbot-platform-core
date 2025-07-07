"""
Database configuration and session management for the Turnkey AI Chatbot platform.
Handles SQLAlchemy setup, connection pooling, and database sessions.
"""

import logging
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
import asyncpg

from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

# Database engines and session makers
engine: Optional[Engine] = None
async_engine = None
SessionLocal: Optional[sessionmaker] = None
AsyncSessionLocal: Optional[async_sessionmaker] = None


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        """Initialize database manager."""
        self._engine: Optional[Engine] = None
        self._async_engine = None
        self._session_local: Optional[sessionmaker] = None
        self._async_session_local: Optional[async_sessionmaker] = None
        self._is_initialized = False
    
    def initialize(self) -> None:
        """Initialize database connections."""
        if self._is_initialized:
            logger.warning("Database manager already initialized")
            return
        
        try:
            self._setup_sync_engine()
            self._setup_async_engine()
            self._setup_session_makers()
            self._setup_event_listeners()
            self._is_initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    def _setup_sync_engine(self) -> None:
        """Setup synchronous database engine."""
        database_url = settings.DATABASE_URL
        
        # Convert async URL to sync if needed
        if database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        
        engine_kwargs = {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            "pool_pre_ping": True,
            "echo": settings.DEBUG,
        }
        
        # Use NullPool for testing environments
        if settings.is_testing():
            engine_kwargs["poolclass"] = NullPool
        else:
            engine_kwargs["poolclass"] = QueuePool
        
        self._engine = create_engine(database_url, **engine_kwargs)
        
        logger.info(f"Synchronous database engine created with pool_size={settings.DATABASE_POOL_SIZE}")
    
    def _setup_async_engine(self) -> None:
        """Setup asynchronous database engine."""
        database_url = settings.DATABASE_URL
        
        # Ensure async URL format
        if not database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        engine_kwargs = {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            "pool_pre_ping": True,
            "echo": settings.DEBUG,
        }
        
        # Use NullPool for testing environments
        if settings.is_testing():
            engine_kwargs["poolclass"] = NullPool
        
        self._async_engine = create_async_engine(database_url, **engine_kwargs)
        
        logger.info(f"Asynchronous database engine created with pool_size={settings.DATABASE_POOL_SIZE}")
    
    def _setup_session_makers(self) -> None:
        """Setup session makers for sync and async operations."""
        if not self._engine or not self._async_engine:
            raise RuntimeError("Engines must be initialized before session makers")
        
        # Synchronous session maker
        self._session_local = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        
        # Asynchronous session maker
        self._async_session_local = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        
        logger.info("Database session makers created")
    
    def _setup_event_listeners(self) -> None:
        """Setup database event listeners for monitoring and debugging."""
        if not self._engine:
            return
        
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set database-specific pragmas on connection."""
            if "sqlite" in str(dbapi_connection):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        
        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log database connection checkout."""
            if settings.DEBUG:
                logger.debug("Database connection checked out")
        
        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log database connection checkin."""
            if settings.DEBUG:
                logger.debug("Database connection checked in")
        
        @event.listens_for(self._engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Log database connection invalidation."""
            logger.warning(f"Database connection invalidated: {exception}")
    
    def get_session(self) -> Session:
        """Get synchronous database session."""
        if not self._session_local:
            raise RuntimeError("Database manager not initialized")
        return self._session_local()
    
    def get_async_session(self) -> AsyncSession:
        """Get asynchronous database session."""
        if not self._async_session_local:
            raise RuntimeError("Database manager not initialized")
        return self._async_session_local()
    
    @asynccontextmanager
    async def get_async_session_context(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with context manager."""
        if not self._async_session_local:
            raise RuntimeError("Database manager not initialized")
        
        async with self._async_session_local() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def test_connection(self) -> bool:
        """Test database connection."""
        if not self._async_engine:
            return False
        
        try:
            async with self._async_engine.begin() as conn:
                await conn.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Async database engine disposed")
        
        if self._engine:
            self._engine.dispose()
            logger.info("Sync database engine disposed")
        
        self._is_initialized = False


# Global database manager instance
db_manager = DatabaseManager()


def init_db() -> None:
    """Initialize database connections."""
    global engine, async_engine, SessionLocal, AsyncSessionLocal
    
    db_manager.initialize()
    
    # Set global variables for backward compatibility
    engine = db_manager._engine
    async_engine = db_manager._async_engine
    SessionLocal = db_manager._session_local
    AsyncSessionLocal = db_manager._async_session_local


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get database session.
    
    Yields:
        AsyncSession: Database session for the request
    """
    async with db_manager.get_async_session_context() as session:
        yield session


def get_sync_db() -> Session:
    """
    Get synchronous database session.
    
    Returns:
        Session: Synchronous database session
    """
    session = db_manager.get_session()
    try:
        return session
    except Exception as e:
        session.close()
        raise


class DatabaseHealthCheck:
    """Database health check utilities."""
    
    @staticmethod
    async def check_async_connection() -> bool:
        """Check async database connection health."""
        return await db_manager.test_connection()
    
    @staticmethod
    def check_sync_connection() -> bool:
        """Check sync database connection health."""
        if not db_manager._engine:
            return False
        
        try:
            with db_manager._engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Sync database health check failed: {e}")
            return False
    
    @staticmethod
    async def get_connection_info() -> dict:
        """Get database connection information."""
        info = {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            "is_initialized": db_manager._is_initialized,
        }
        
        if db_manager._engine:
            pool = db_manager._engine.pool
            info.update({
                "pool_checked_in": pool.checkedin(),
                "pool_checked_out": pool.checkedout(),
                "pool_overflow": pool.overflow(),
            })
        
        return info


class DatabaseError(Exception):
    """Custom database error."""
    pass


def handle_database_errors(func):
    """Decorator to handle database errors."""
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except DisconnectionError as e:
            logger.error(f"Database disconnection error in {func.__name__}: {e}")
            raise DatabaseError("Database connection lost. Please try again.")
        except SQLAlchemyError as e:
            logger.error(f"Database error in {func.__name__}: {e}")
            raise DatabaseError("Database operation failed. Please try again.")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DisconnectionError as e:
            logger.error(f"Database disconnection error in {func.__name__}: {e}")
            raise DatabaseError("Database connection lost. Please try again.")
        except SQLAlchemyError as e:
            logger.error(f"Database error in {func.__name__}: {e}")
            raise DatabaseError("Database operation failed. Please try again.")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
