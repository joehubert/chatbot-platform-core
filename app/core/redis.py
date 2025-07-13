"""
Redis configuration and connection management for the Turnkey AI Chatbot platform.
Handles Redis connections, connection pooling, and health checks.
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError
from urllib.parse import urlparse

from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis connections and operations."""

    def __init__(self):
        """Initialize Redis manager."""
        self._redis: Optional[Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._is_initialized = False
        self._connection_params = {}

    def initialize(self) -> None:
        """Initialize Redis connection pool."""
        if self._is_initialized:
            logger.warning("Redis manager already initialized")
            return

        try:
            self._setup_connection_params()
            self._create_connection_pool()
            self._create_redis_client()
            self._is_initialized = True
            logger.info("Redis manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Redis manager: {e}")
            raise

    def _setup_connection_params(self) -> None:
        """Setup Redis connection parameters."""
        redis_url = settings.REDIS_URL
        parsed_url = urlparse(redis_url)

        self._connection_params = {
            "host": parsed_url.hostname or "localhost",
            "port": parsed_url.port or 6379,
            "db": settings.REDIS_DB,
            "password": settings.REDIS_PASSWORD or parsed_url.password,
            "socket_timeout": settings.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT,
            "retry_on_timeout": True,
            "retry_on_error": [ConnectionError, TimeoutError],
            "health_check_interval": 30,
            "decode_responses": True,
        }

        # Add SSL configuration if using Redis SSL
        if parsed_url.scheme == "rediss":
            self._connection_params["ssl"] = True
            self._connection_params["ssl_check_hostname"] = False

        logger.info(
            f"Redis connection parameters configured for {self._connection_params['host']}:{self._connection_params['port']}"
        )

    def _create_connection_pool(self) -> None:
        """Create Redis connection pool."""
        pool_params = self._connection_params.copy()
        pool_params["max_connections"] = settings.REDIS_POOL_SIZE

        self._pool = ConnectionPool(**pool_params)
        logger.info(
            f"Redis connection pool created with max_connections={settings.REDIS_POOL_SIZE}"
        )

    def _create_redis_client(self) -> None:
        """Create Redis client with connection pool."""
        if not self._pool:
            raise RuntimeError("Connection pool must be created before Redis client")

        self._redis = Redis(connection_pool=self._pool)
        logger.info("Redis client created with connection pool")

    async def get_redis_client(self) -> Redis:
        """Get Redis client instance."""
        if not self._redis:
            raise RuntimeError("Redis manager not initialized")
        return self._redis

    async def test_connection(self) -> bool:
        """Test Redis connection."""
        if not self._redis:
            return False

        try:
            await self._redis.ping()
            logger.info("Redis connection test successful")
            return True
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            return False

    async def get_info(self) -> dict:
        """Get Redis server information."""
        if not self._redis:
            return {}

        try:
            info = await self._redis.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}

    async def close(self) -> None:
        """Close Redis connections."""
        if self._redis:
            await self._redis.close()
            logger.info("Redis client closed")

        if self._pool:
            await self._pool.disconnect()
            logger.info("Redis connection pool disconnected")

        self._is_initialized = False


# Global Redis manager instance
redis_manager = RedisManager()


class RedisCache:
    """Redis-based caching operations."""

    def __init__(self, redis_client: Redis):
        """Initialize Redis cache with client."""
        self.redis = redis_client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            return json.loads(value)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in cache."""
        try:
            serialized_value = json.dumps(value, default=str)
            return await self.redis.set(key, serialized_value, ex=ttl, nx=nx, xx=xx)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get time to live for key."""
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -1

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value in cache."""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for existing key."""
        try:
            return await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False


class RedisRateLimiter:
    """Redis-based rate limiting using token bucket algorithm."""

    def __init__(self, redis_client: Redis):
        """Initialize rate limiter with Redis client."""
        self.redis = redis_client

    async def is_allowed(
        self, key: str, limit: int, window: int, burst: Optional[int] = None
    ) -> tuple[bool, dict]:
        """
        Check if request is allowed based on rate limit.

        Args:
            key: Unique identifier for the rate limit bucket
            limit: Number of requests allowed per window
            window: Time window in seconds
            burst: Burst capacity (defaults to limit)

        Returns:
            Tuple of (is_allowed, info_dict)
        """
        if burst is None:
            burst = limit

        try:
            current_time = int(asyncio.get_event_loop().time())

            # Use Lua script for atomic operations
            lua_script = """
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local burst = tonumber(ARGV[3])
            local current_time = tonumber(ARGV[4])
            
            local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket[1]) or burst
            local last_refill = tonumber(bucket[2]) or current_time
            
            -- Calculate tokens to add based on time elapsed
            local time_elapsed = current_time - last_refill
            local tokens_to_add = math.floor(time_elapsed * limit / window)
            tokens = math.min(burst, tokens + tokens_to_add)
            
            local allowed = false
            if tokens >= 1 then
                tokens = tokens - 1
                allowed = true
            end
            
            -- Update bucket
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, window * 2)
            
            return {allowed and 1 or 0, tokens, limit, window, burst}
            """

            result = await self.redis.eval(
                lua_script, 1, key, window, limit, burst, current_time
            )

            is_allowed = bool(result[0])
            info = {
                "allowed": is_allowed,
                "tokens_remaining": result[1],
                "limit": result[2],
                "window": result[3],
                "burst": result[4],
                "reset_time": current_time + window,
            }

            return is_allowed, info

        except Exception as e:
            logger.error(f"Rate limiter error for key {key}: {e}")
            # Fail open - allow request if Redis is down
            return True, {"allowed": True, "error": str(e)}

    async def reset(self, key: str) -> bool:
        """Reset rate limit for key."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Rate limiter reset error for key {key}: {e}")
            return False


class RedisSemanticCache:
    """Redis-based semantic cache for conversation responses."""

    def __init__(self, redis_client: Redis):
        """Initialize semantic cache with Redis client."""
        self.redis = redis_client
        self.cache_prefix = "semantic_cache:"
        self.index_key = "semantic_cache_index"

    async def store(
        self,
        query_embedding: List[float],
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store response with query embedding."""
        try:
            cache_id = f"cache_{int(asyncio.get_event_loop().time() * 1000000)}"
            cache_key = f"{self.cache_prefix}{cache_id}"

            cache_data = {
                "embedding": query_embedding,
                "response": response,
                "metadata": metadata or {},
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Store cache entry
            await self.redis.hset(
                cache_key, mapping={"data": json.dumps(cache_data, default=str)}
            )

            if ttl:
                await self.redis.expire(cache_key, ttl)

            # Add to index
            await self.redis.sadd(self.index_key, cache_id)

            return cache_id

        except Exception as e:
            logger.error(f"Semantic cache store error: {e}")
            return ""

    async def search(
        self,
        query_embedding: List[float],
        similarity_threshold: float = 0.85,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar cached responses."""
        try:
            # Get all cache IDs
            cache_ids = await self.redis.smembers(self.index_key)
            results = []

            for cache_id in cache_ids:
                cache_key = f"{self.cache_prefix}{cache_id}"
                cache_data_raw = await self.redis.hget(cache_key, "data")

                if not cache_data_raw:
                    continue

                try:
                    cache_data = json.loads(cache_data_raw)
                    cached_embedding = cache_data.get("embedding", [])

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(
                        query_embedding, cached_embedding
                    )

                    if similarity >= similarity_threshold:
                        results.append(
                            {
                                "cache_id": cache_id,
                                "similarity": similarity,
                                "response": cache_data.get("response"),
                                "metadata": cache_data.get("metadata", {}),
                                "timestamp": cache_data.get("timestamp"),
                            }
                        )

                except json.JSONDecodeError:
                    continue

            # Sort by similarity and return top results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Semantic cache search error: {e}")
            return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        try:
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = sum(x * x for x in a) ** 0.5
            magnitude_b = sum(x * x for x in b) ** 0.5

            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0

            return dot_product / (magnitude_a * magnitude_b)

        except Exception:
            return 0.0

    async def invalidate(self, cache_id: str) -> bool:
        """Remove cached entry."""
        try:
            cache_key = f"{self.cache_prefix}{cache_id}"
            await self.redis.delete(cache_key)
            await self.redis.srem(self.index_key, cache_id)
            return True
        except Exception as e:
            logger.error(f"Semantic cache invalidate error for {cache_id}: {e}")
            return False

    async def clear_all(self) -> bool:
        """Clear all cached entries."""
        try:
            cache_ids = await self.redis.smembers(self.index_key)

            if cache_ids:
                cache_keys = [
                    f"{self.cache_prefix}{cache_id}" for cache_id in cache_ids
                ]
                await self.redis.delete(*cache_keys)

            await self.redis.delete(self.index_key)
            return True
        except Exception as e:
            logger.error(f"Semantic cache clear error: {e}")
            return False


async def init_redis() -> None:
    """Initialize Redis connections."""
    redis_manager.initialize()


async def get_redis_client() -> Redis:
    """Get Redis client for dependency injection."""
    return await redis_manager.get_redis_client()


async def get_cache() -> RedisCache:
    """Get Redis cache instance."""
    redis_client = await redis_manager.get_redis_client()
    return RedisCache(redis_client)


async def get_rate_limiter() -> RedisRateLimiter:
    """Get Redis rate limiter instance."""
    redis_client = await redis_manager.get_redis_client()
    return RedisRateLimiter(redis_client)


async def get_semantic_cache() -> RedisSemanticCache:
    """Get Redis semantic cache instance."""
    redis_client = await redis_manager.get_redis_client()
    return RedisSemanticCache(redis_client)


class RedisHealthCheck:
    """Redis health check utilities."""

    @staticmethod
    async def check_connection() -> bool:
        """Check Redis connection health."""
        return await redis_manager.test_connection()

    @staticmethod
    async def get_connection_info() -> dict:
        """Get Redis connection information."""
        info = {
            "pool_size": settings.REDIS_POOL_SIZE,
            "socket_timeout": settings.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT,
            "is_initialized": redis_manager._is_initialized,
        }

        redis_info = await redis_manager.get_info()
        info.update(redis_info)

        return info


class RedisError(Exception):
    """Custom Redis error."""

    pass
