"""
Redis Utilities
Centralized Redis connection management, health checks, and utility functions.
Provides connection pooling, retry logic, and monitoring capabilities.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlparse

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import (
    RedisError,
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
    ResponseError,
)

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RedisConfig:
    """Redis configuration parameters"""

    host: str
    port: int
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_keepalive_options: Optional[Dict[str, int]] = None
    health_check_interval: int = 30

    @classmethod
    def from_url(cls, redis_url: str) -> "RedisConfig":
        """Create config from Redis URL"""
        parsed = urlparse(redis_url)

        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            db=int(parsed.path.lstrip("/")) if parsed.path else 0,
            ssl=parsed.scheme == "rediss",
        )


@dataclass
class RedisHealthStatus:
    """Redis health check result"""

    is_healthy: bool
    latency_ms: Optional[float]
    error_message: Optional[str]
    memory_usage: Optional[Dict[str, Any]]
    connection_count: Optional[int]
    last_check: float


class RedisConnectionManager:
    """
    Redis connection manager with pooling, health checks, and retry logic.
    """

    def __init__(self, config: RedisConfig):
        """
        Initialize Redis connection manager.

        Args:
            config: Redis configuration
        """
        self.config = config
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self._health_status = RedisHealthStatus(
            is_healthy=False,
            latency_ms=None,
            error_message=None,
            memory_usage=None,
            connection_count=None,
            last_check=0,
        )
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize Redis connection pool"""
        try:
            # Create connection pool
            socket_keepalive_options = self.config.socket_keepalive_options or {}

            self.pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                ssl=self.config.ssl,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=socket_keepalive_options,
                decode_responses=True,
            )

            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)

            # Test connection
            await self.client.ping()

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(
                f"Redis connection initialized: {self.config.host}:{self.config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connections"""
        try:
            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Close client and pool
            if self.client:
                await self.client.close()

            if self.pool:
                await self.pool.disconnect()

            logger.info("Redis connections closed")

        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")

    async def get_client(self) -> redis.Redis:
        """
        Get Redis client with automatic reconnection.

        Returns:
            Redis client instance

        Raises:
            RedisConnectionError: If connection cannot be established
        """
        if not self.client:
            raise RedisConnectionError("Redis client not initialized")

        # Check if connection is healthy
        if not self._health_status.is_healthy:
            await self._attempt_reconnection()

        return self.client

    async def execute_with_retry(
        self,
        operation: str,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """
        Execute Redis operation with retry logic.

        Args:
            operation: Redis operation name (e.g., 'get', 'set', 'hget')
            *args: Operation arguments
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Operation keyword arguments

        Returns:
            Operation result

        Raises:
            RedisError: If operation fails after all retries
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                client = await self.get_client()
                operation_func = getattr(client, operation)

                if asyncio.iscoroutinefunction(operation_func):
                    return await operation_func(*args, **kwargs)
                else:
                    return operation_func(*args, **kwargs)

            except (RedisConnectionError, RedisTimeoutError) as e:
                last_error = e

                if attempt < max_retries:
                    logger.warning(
                        f"Redis operation '{operation}' failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff
                    await self._attempt_reconnection()
                else:
                    logger.error(
                        f"Redis operation '{operation}' failed after {max_retries + 1} attempts"
                    )

            except Exception as e:
                # Non-connection errors shouldn't be retried
                logger.error(
                    f"Redis operation '{operation}' failed with non-recoverable error: {e}"
                )
                raise

        raise last_error or RedisError(f"Operation '{operation}' failed")

    async def health_check(self) -> RedisHealthStatus:
        """
        Perform Redis health check.

        Returns:
            Health status result
        """
        start_time = time.time()

        try:
            if not self.client:
                return RedisHealthStatus(
                    is_healthy=False,
                    latency_ms=None,
                    error_message="Redis client not initialized",
                    memory_usage=None,
                    connection_count=None,
                    last_check=time.time(),
                )

            # Test basic connectivity
            await self.client.ping()
            latency_ms = (time.time() - start_time) * 1000

            # Get additional info
            try:
                info = await self.client.info()
                memory_usage = {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "Unknown"),
                    "used_memory_peak": info.get("used_memory_peak", 0),
                    "used_memory_peak_human": info.get(
                        "used_memory_peak_human", "Unknown"
                    ),
                }
                connection_count = info.get("connected_clients", 0)
            except:
                memory_usage = None
                connection_count = None

            return RedisHealthStatus(
                is_healthy=True,
                latency_ms=latency_ms,
                error_message=None,
                memory_usage=memory_usage,
                connection_count=connection_count,
                last_check=time.time(),
            )

        except Exception as e:
            return RedisHealthStatus(
                is_healthy=False,
                latency_ms=None,
                error_message=str(e),
                memory_usage=None,
                connection_count=None,
                last_check=time.time(),
            )

    def get_health_status(self) -> RedisHealthStatus:
        """Get current health status"""
        return self._health_status

    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while True:
            try:
                self._health_status = await self.health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to Redis"""
        async with self._lock:
            if self._health_status.is_healthy:
                return  # Another task already reconnected

            try:
                logger.info("Attempting Redis reconnection...")

                # Close existing connections
                if self.client:
                    await self.client.close()

                if self.pool:
                    await self.pool.disconnect()

                # Reinitialize
                await self.initialize()

                logger.info("Redis reconnection successful")

            except Exception as e:
                logger.error(f"Redis reconnection failed: {e}")
                raise


class RedisManager:
    """
    Global Redis manager singleton.
    Provides centralized Redis access across the application.
    """

    _instance: Optional["RedisManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "RedisManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._connection_manager: Optional[RedisConnectionManager] = None
        self._config: Optional[RedisConfig] = None
        self._initialized = False

    async def initialize(self, redis_url: Optional[str] = None) -> None:
        """
        Initialize Redis manager.

        Args:
            redis_url: Redis connection URL. If None, uses settings.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                # Create configuration
                if redis_url:
                    self._config = RedisConfig.from_url(redis_url)
                else:
                    self._config = RedisConfig.from_url(settings.REDIS_URL)

                # Set additional config from settings
                self._config.max_connections = settings.REDIS_MAX_CONNECTIONS
                self._config.health_check_interval = (
                    settings.REDIS_HEALTH_CHECK_INTERVAL
                )

                # Create connection manager
                self._connection_manager = RedisConnectionManager(self._config)
                await self._connection_manager.initialize()

                self._initialized = True
                logger.info("Redis manager initialized")

            except Exception as e:
                logger.error(f"Failed to initialize Redis manager: {e}")
                raise

    async def close(self) -> None:
        """Close Redis manager"""
        if self._connection_manager:
            await self._connection_manager.close()

        self._initialized = False
        logger.info("Redis manager closed")

    async def get_connection(self) -> redis.Redis:
        """
        Get Redis connection.

        Returns:
            Redis client instance
        """
        if not self._initialized or not self._connection_manager:
            await self.initialize()

        return await self._connection_manager.get_client()

    async def execute_with_retry(self, operation: str, *args, **kwargs) -> Any:
        """Execute Redis operation with retry logic"""
        if not self._initialized or not self._connection_manager:
            await self.initialize()

        return await self._connection_manager.execute_with_retry(
            operation, *args, **kwargs
        )

    async def health_check(self) -> RedisHealthStatus:
        """Perform health check"""
        if not self._initialized or not self._connection_manager:
            return RedisHealthStatus(
                is_healthy=False,
                latency_ms=None,
                error_message="Redis manager not initialized",
                memory_usage=None,
                connection_count=None,
                last_check=time.time(),
            )

        return await self._connection_manager.health_check()

    def get_health_status(self) -> RedisHealthStatus:
        """Get current health status"""
        if not self._initialized or not self._connection_manager:
            return RedisHealthStatus(
                is_healthy=False,
                latency_ms=None,
                error_message="Redis manager not initialized",
                memory_usage=None,
                connection_count=None,
                last_check=time.time(),
            )

        return self._connection_manager.get_health_status()


class RedisLock:
    """
    Distributed lock implementation using Redis.
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        key: str,
        timeout: float = 30.0,
        sleep: float = 0.1,
    ):
        """
        Initialize Redis lock.

        Args:
            redis_manager: Redis manager instance
            key: Lock key
            timeout: Lock timeout in seconds
            sleep: Sleep interval between attempts
        """
        self.redis_manager = redis_manager
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.sleep = sleep
        self.identifier = f"{time.time()}:{id(self)}"

    async def acquire(self) -> bool:
        """
        Acquire lock.

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            redis_client = await self.redis_manager.get_connection()

            # Try to acquire lock
            result = await redis_client.set(
                self.key,
                self.identifier,
                nx=True,  # Only set if key doesn't exist
                ex=int(self.timeout),  # Set expiration
            )

            return result is not None

        except Exception as e:
            logger.error(f"Error acquiring lock {self.key}: {e}")
            return False

    async def release(self) -> bool:
        """
        Release lock.

        Returns:
            True if lock released, False otherwise
        """
        try:
            redis_client = await self.redis_manager.get_connection()

            # Lua script to safely release lock
            release_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """

            result = await redis_client.eval(
                release_script, 1, self.key, self.identifier
            )
            return bool(result)

        except Exception as e:
            logger.error(f"Error releasing lock {self.key}: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry"""
        while True:
            if await self.acquire():
                return self
            await asyncio.sleep(self.sleep)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.release()


# Utility functions


async def redis_pipeline(
    redis_manager: RedisManager, commands: List[Dict[str, Any]]
) -> List[Any]:
    """
    Execute multiple Redis commands in a pipeline.

    Args:
        redis_manager: Redis manager instance
        commands: List of command dictionaries with 'operation', 'args', 'kwargs'

    Returns:
        List of command results
    """
    try:
        redis_client = await redis_manager.get_connection()

        pipe = redis_client.pipeline()

        for cmd in commands:
            operation = cmd["operation"]
            args = cmd.get("args", [])
            kwargs = cmd.get("kwargs", {})

            getattr(pipe, operation)(*args, **kwargs)

        return await pipe.execute()

    except Exception as e:
        logger.error(f"Error executing Redis pipeline: {e}")
        raise


async def redis_scan_keys(
    redis_manager: RedisManager, pattern: str, count: int = 100
) -> List[str]:
    """
    Scan Redis keys matching pattern.

    Args:
        redis_manager: Redis manager instance
        pattern: Key pattern to match
        count: Number of keys per scan iteration

    Returns:
        List of matching keys
    """
    try:
        redis_client = await redis_manager.get_connection()
        keys = []

        async for key in redis_client.scan_iter(match=pattern, count=count):
            keys.append(key)

        return keys

    except Exception as e:
        logger.error(f"Error scanning Redis keys: {e}")
        return []


async def redis_memory_usage(redis_manager: RedisManager, key: str) -> Optional[int]:
    """
    Get memory usage for a Redis key.

    Args:
        redis_manager: Redis manager instance
        key: Redis key

    Returns:
        Memory usage in bytes, or None if error
    """
    try:
        redis_client = await redis_manager.get_connection()
        return await redis_client.memory_usage(key)

    except Exception as e:
        logger.error(f"Error getting memory usage for key {key}: {e}")
        return None


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None


async def get_redis_manager() -> RedisManager:
    """Get or create global Redis manager instance"""
    global _redis_manager

    if _redis_manager is None:
        _redis_manager = RedisManager()
        await _redis_manager.initialize()

    return _redis_manager


# FastAPI dependency
async def redis_dependency() -> RedisManager:
    """FastAPI dependency for Redis manager"""
    return await get_redis_manager()


# Context manager for Redis operations
@asynccontextmanager
async def redis_transaction(redis_manager: RedisManager):
    """
    Context manager for Redis transactions.

    Args:
        redis_manager: Redis manager instance

    Yields:
        Redis pipeline for transaction
    """
    redis_client = await redis_manager.get_connection()
    pipe = redis_client.pipeline(transaction=True)

    try:
        yield pipe
        await pipe.execute()
    except Exception as e:
        logger.error(f"Redis transaction failed: {e}")
        await pipe.reset()
        raise
    finally:
        await pipe.reset()


# Utility decorators


def redis_retry(max_retries: int = 3, retry_delay: float = 1.0):
    """
    Decorator for Redis operations with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (RedisConnectionError, RedisTimeoutError) as e:
                    last_error = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Redis operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        await asyncio.sleep(retry_delay * (2**attempt))
                    else:
                        logger.error(
                            f"Redis operation failed after {max_retries + 1} attempts"
                        )
                except Exception as e:
                    logger.error(
                        f"Redis operation failed with non-recoverable error: {e}"
                    )
                    raise

            raise last_error or RedisError("Operation failed")

        return wrapper

    return decorator


class RedisKeyBuilder:
    """
    Utility class for building consistent Redis keys.
    """

    @staticmethod
    def cache_key(cache_type: str, identifier: str, suffix: str = "") -> str:
        """Build cache key"""
        key = f"cache:{cache_type}:{identifier}"
        return f"{key}:{suffix}" if suffix else key

    @staticmethod
    def rate_limit_key(limit_type: str, identifier: str) -> str:
        """Build rate limit key"""
        return f"rate_limit:{limit_type}:{identifier}"

    @staticmethod
    def session_key(session_id: str) -> str:
        """Build session key"""
        return f"session:{session_id}"

    @staticmethod
    def lock_key(resource: str) -> str:
        """Build lock key"""
        return f"lock:{resource}"

    @staticmethod
    def user_key(user_id: str, data_type: str = "") -> str:
        """Build user key"""
        key = f"user:{user_id}"
        return f"{key}:{data_type}" if data_type else key

    @staticmethod
    def conversation_key(conversation_id: str, data_type: str = "") -> str:
        """Build conversation key"""
        key = f"conversation:{conversation_id}"
        return f"{key}:{data_type}" if data_type else key

    @staticmethod
    def document_key(document_id: str, data_type: str = "") -> str:
        """Build document key"""
        key = f"document:{document_id}"
        return f"{key}:{data_type}" if data_type else key


class RedisMetrics:
    """
    Redis metrics collection utility.
    """

    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get Redis connection statistics"""
        try:
            redis_client = await self.redis_manager.get_connection()
            info = await redis_client.info()

            return {
                "connected_clients": info.get("connected_clients", 0),
                "client_recent_max_input_buffer": info.get(
                    "client_recent_max_input_buffer", 0
                ),
                "client_recent_max_output_buffer": info.get(
                    "client_recent_max_output_buffer", 0
                ),
                "blocked_clients": info.get("blocked_clients", 0),
                "total_connections_received": info.get("total_connections_received", 0),
                "rejected_connections": info.get("rejected_connections", 0),
            }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get Redis memory statistics"""
        try:
            redis_client = await self.redis_manager.get_connection()
            info = await redis_client.info("memory")

            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_rss": info.get("used_memory_rss", 0),
                "used_memory_rss_human": info.get("used_memory_rss_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "used_memory_peak_perc": info.get("used_memory_peak_perc", "0.00%"),
                "used_memory_overhead": info.get("used_memory_overhead", 0),
                "used_memory_startup": info.get("used_memory_startup", 0),
                "used_memory_dataset": info.get("used_memory_dataset", 0),
                "used_memory_dataset_perc": info.get(
                    "used_memory_dataset_perc", "0.00%"
                ),
                "total_system_memory": info.get("total_system_memory", 0),
                "total_system_memory_human": info.get(
                    "total_system_memory_human", "0B"
                ),
                "used_memory_lua": info.get("used_memory_lua", 0),
                "used_memory_lua_human": info.get("used_memory_lua_human", "0B"),
                "maxmemory": info.get("maxmemory", 0),
                "maxmemory_human": info.get("maxmemory_human", "0B"),
                "maxmemory_policy": info.get("maxmemory_policy", "noeviction"),
                "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0.0),
                "mem_fragmentation_bytes": info.get("mem_fragmentation_bytes", 0),
                "mem_allocator": info.get("mem_allocator", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    async def get_keyspace_stats(self) -> Dict[str, Any]:
        """Get Redis keyspace statistics"""
        try:
            redis_client = await self.redis_manager.get_connection()
            info = await redis_client.info("keyspace")

            keyspace_stats = {}
            for key, value in info.items():
                if key.startswith("db"):
                    # Parse db stats: keys=X,expires=Y,avg_ttl=Z
                    stats = {}
                    for stat in value.split(","):
                        stat_key, stat_value = stat.split("=")
                        stats[stat_key] = (
                            int(stat_value) if stat_value.isdigit() else stat_value
                        )
                    keyspace_stats[key] = stats

            return keyspace_stats
        except Exception as e:
            logger.error(f"Error getting keyspace stats: {e}")
            return {}

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get Redis performance statistics"""
        try:
            redis_client = await self.redis_manager.get_connection()
            info = await redis_client.info("stats")

            return {
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "total_net_input_bytes": info.get("total_net_input_bytes", 0),
                "total_net_output_bytes": info.get("total_net_output_bytes", 0),
                "instantaneous_input_kbps": info.get("instantaneous_input_kbps", 0.0),
                "instantaneous_output_kbps": info.get("instantaneous_output_kbps", 0.0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "pubsub_channels": info.get("pubsub_channels", 0),
                "pubsub_patterns": info.get("pubsub_patterns", 0),
                "latest_fork_usec": info.get("latest_fork_usec", 0),
                "migrate_cached_sockets": info.get("migrate_cached_sockets", 0),
                "slave_expires_tracked_keys": info.get("slave_expires_tracked_keys", 0),
                "active_defrag_hits": info.get("active_defrag_hits", 0),
                "active_defrag_misses": info.get("active_defrag_misses", 0),
                "active_defrag_key_hits": info.get("active_defrag_key_hits", 0),
                "active_defrag_key_misses": info.get("active_defrag_key_misses", 0),
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}

    async def get_full_stats(self) -> Dict[str, Any]:
        """Get comprehensive Redis statistics"""
        return {
            "connection": await self.get_connection_stats(),
            "memory": await self.get_memory_stats(),
            "keyspace": await self.get_keyspace_stats(),
            "performance": await self.get_performance_stats(),
            "health": self.redis_manager.get_health_status().__dict__,
        }


# Background tasks for Redis maintenance


async def redis_cleanup_task(redis_manager: RedisManager, cleanup_interval: int = 3600):
    """
    Background task for Redis cleanup operations.

    Args:
        redis_manager: Redis manager instance
        cleanup_interval: Cleanup interval in seconds
    """
    while True:
        try:
            logger.info("Starting Redis cleanup task")

            # Clean up expired keys (this is mostly handled by Redis TTL,
            # but we can do additional cleanup for specific patterns)

            # Example: Clean up old session data
            session_pattern = "session:*"
            old_sessions = await redis_scan_keys(redis_manager, session_pattern)

            current_time = time.time()
            cleaned_sessions = 0

            redis_client = await redis_manager.get_connection()

            for session_key in old_sessions:
                try:
                    session_data = await redis_client.hgetall(session_key)
                    if session_data.get("last_activity"):
                        last_activity = float(session_data["last_activity"])
                        # Clean sessions older than 24 hours
                        if current_time - last_activity > 86400:
                            await redis_client.delete(session_key)
                            cleaned_sessions += 1
                except Exception as e:
                    logger.warning(f"Error cleaning session {session_key}: {e}")

            logger.info(f"Redis cleanup completed: {cleaned_sessions} sessions cleaned")

        except Exception as e:
            logger.error(f"Error in Redis cleanup task: {e}")

        await asyncio.sleep(cleanup_interval)


# Health check endpoint helper


async def redis_health_check() -> Dict[str, Any]:
    """
    Comprehensive Redis health check for monitoring endpoints.

    Returns:
        Dictionary with health check results
    """
    try:
        redis_manager = await get_redis_manager()
        health_status = await redis_manager.health_check()

        return {
            "status": "healthy" if health_status.is_healthy else "unhealthy",
            "latency_ms": health_status.latency_ms,
            "error": health_status.error_message,
            "last_check": health_status.last_check,
            "details": {
                "memory_usage": health_status.memory_usage,
                "connection_count": health_status.connection_count,
            },
        }

    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "last_check": time.time()}
