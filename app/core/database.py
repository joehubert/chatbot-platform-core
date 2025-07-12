# app/core/database.py - Fix for the database connection test

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Any
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import StaticPool
import redis.asyncio as redis
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Create declarative base for models
Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self):
        self.async_engine = None
        self.sync_engine = None
        self.async_session_maker = None
        self.sync_session_maker = None
        self._initialized = False

    def initialize(self):
        """Initialize database engines and session makers"""
        if self._initialized:
            return

        # Create sync engine
        self.sync_engine = create_engine(
            settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"),
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            echo=settings.DATABASE_ECHO,
        )
        logger.info(
            f"Synchronous database engine created with pool_size={settings.DATABASE_POOL_SIZE}"
        )

        # Create async engine
        self.async_engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            echo=settings.DATABASE_ECHO,
        )
        logger.info(
            f"Asynchronous database engine created with pool_size={settings.DATABASE_POOL_SIZE}"
        )

        # Create session makers
        self.sync_session_maker = sessionmaker(
            bind=self.sync_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

        self.async_session_maker = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

        logger.info("Database session makers created")
        self._initialized = True
        logger.info("Database manager initialized successfully")

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            # Test sync connection
            with self.sync_engine.connect() as conn:
                # Use text() to create executable SQL
                result = conn.execute(text("SELECT 1"))
                if result.fetchone() is None:
                    return False

            # Test async connection
            async with self.async_engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                if result.fetchone() is None:
                    return False

            logger.info("Database connection test successful")
            return True

        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")

        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_sync_session(self) -> Session:
        """Get sync database session"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")

        return self.sync_session_maker()

    async def create_tables(self):
        """Create database tables"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    async def drop_tables(self):
        """Drop database tables"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")

    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("Async database engine disposed")

        if self.sync_engine:
            self.sync_engine.dispose()
            logger.info("Sync database engine disposed")

        self._initialized = False


class RedisManager:
    """Manages Redis connections"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False

    async def initialize(self):
        """Initialize Redis connection"""
        if self._initialized:
            return

        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                max_connections=20,
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established successfully")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            raise

    async def test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            if not self.redis_client:
                return False
            await self.redis_client.ping()
            logger.info("Redis connection test successful")
            return True
        except Exception as e:
            logger.error(f"Redis connection test failed: {str(e)}")
            return False

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("Redis connection closed")
        self._initialized = False


# Global instances
db_manager = DatabaseManager()
redis_manager = RedisManager()


# Dependency functions for FastAPI
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get async database session"""
    async with db_manager.get_async_session() as session:
        yield session


def get_sync_db() -> Session:
    """Dependency to get sync database session"""
    session = db_manager.get_sync_session()
    try:
        yield session
    finally:
        session.close()


async def get_redis() -> redis.Redis:
    """Dependency to get Redis client"""
    if not redis_manager.redis_client:
        raise RuntimeError("Redis not initialized")
    return redis_manager.redis_client


# Utility functions
async def init_db():
    """Initialize database and create tables"""
    db_manager.initialize()

    # Test connection
    if not await db_manager.test_connection():
        raise RuntimeError("Database connection failed")

    # Initialize Redis
    await redis_manager.initialize()

    # Test Redis connection
    if not await redis_manager.test_connection():
        logger.warning("Redis connection failed - some features may not work")

    # Create tables
    await db_manager.create_tables()


async def close_db():
    """Close database connections"""
    await db_manager.close()
    await redis_manager.close()
