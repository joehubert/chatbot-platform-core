"""
Rate Limiting Service
Implementation of Redis-based token bucket algorithm for rate limiting.
Supports both per-user and global rate limiting with configurable thresholds.
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from ..utils.redis_utils import RedisManager
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimitType(Enum):
    """Rate limit types"""
    USER = "user"
    GLOBAL = "global"
    IP = "ip"
    SESSION = "session"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int
    burst_capacity: int
    window_size_seconds: int = 60
    
    def __post_init__(self):
        """Validate configuration"""
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.burst_capacity <= 0:
            raise ValueError("burst_capacity must be positive")
        if self.window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    limit: Optional[int] = None
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Limit": str(self.limit) if self.limit else "unknown",
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time)),
        }
        
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
            
        return headers


class RateLimitError(Exception):
    """Rate limit exceeded error"""
    
    def __init__(self, message: str, result: RateLimitResult):
        super().__init__(message)
        self.result = result


class RateLimitService:
    """
    Redis-based rate limiting service using token bucket algorithm.
    
    Features:
    - Token bucket algorithm for smooth rate limiting
    - Per-user and global rate limiting
    - Configurable burst capacity
    - Graceful degradation on Redis failures
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, redis_manager: Optional[RedisManager] = None):
        """
        Initialize rate limiting service.
        
        Args:
            redis_manager: Redis connection manager. If None, creates default.
        """
        self.redis_manager = redis_manager or RedisManager()
        
        # Default configurations
        self.configs = {
            RateLimitType.USER: RateLimitConfig(
                requests_per_minute=settings.RATE_LIMIT_PER_USER_PER_MINUTE,
                burst_capacity=settings.RATE_LIMIT_USER_BURST_CAPACITY,
            ),
            RateLimitType.GLOBAL: RateLimitConfig(
                requests_per_minute=settings.RATE_LIMIT_GLOBAL_PER_MINUTE,
                burst_capacity=settings.RATE_LIMIT_GLOBAL_BURST_CAPACITY,
            ),
            RateLimitType.IP: RateLimitConfig(
                requests_per_minute=settings.RATE_LIMIT_PER_IP_PER_MINUTE,
                burst_capacity=settings.RATE_LIMIT_IP_BURST_CAPACITY,
            ),
            RateLimitType.SESSION: RateLimitConfig(
                requests_per_minute=settings.RATE_LIMIT_PER_SESSION_PER_MINUTE,
                burst_capacity=settings.RATE_LIMIT_SESSION_BURST_CAPACITY,
            ),
        }
        
        # Lua script for atomic token bucket operations
        self.token_bucket_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local tokens_requested = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local window_size = tonumber(ARGV[4])
        local current_time = tonumber(ARGV[5])
        
        -- Get current bucket state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local current_tokens = tonumber(bucket[1])
        local last_refill = tonumber(bucket[2])
        
        -- Initialize bucket if it doesn't exist
        if current_tokens == nil then
            current_tokens = capacity
            last_refill = current_time
        end
        
        -- Calculate tokens to add based on time elapsed
        local time_elapsed = current_time - last_refill
        local tokens_to_add = math.floor(time_elapsed * refill_rate / window_size)
        
        -- Add tokens but don't exceed capacity
        current_tokens = math.min(capacity, current_tokens + tokens_to_add)
        
        -- Check if request can be satisfied
        local allowed = current_tokens >= tokens_requested
        local remaining_tokens = current_tokens
        
        if allowed then
            remaining_tokens = current_tokens - tokens_requested
        end
        
        -- Update bucket state
        redis.call('HMSET', key, 
                  'tokens', remaining_tokens,
                  'last_refill', current_time)
        redis.call('EXPIRE', key, window_size * 2)  -- TTL is 2x window size
        
        -- Calculate reset time
        local reset_time = current_time + ((capacity - remaining_tokens) * window_size / refill_rate)
        
        return {
            allowed and 1 or 0,
            remaining_tokens,
            reset_time,
            capacity
        }
        """
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit_type: RateLimitType,
        tokens_requested: int = 1,
        config_override: Optional[RateLimitConfig] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limits using token bucket algorithm.
        
        Args:
            identifier: Unique identifier (user_id, ip_address, session_id, etc.)
            limit_type: Type of rate limit to apply
            tokens_requested: Number of tokens to consume (default: 1)
            config_override: Override default configuration
            
        Returns:
            RateLimitResult with decision and metadata
            
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        try:
            config = config_override or self.configs[limit_type]
            current_time = time.time()
            
            # Generate Redis key
            key = self._generate_key(limit_type, identifier)
            
            # Get Redis connection
            redis_client = await self.redis_manager.get_connection()
            
            # Execute token bucket algorithm
            result = await redis_client.eval(
                self.token_bucket_script,
                1,  # Number of keys
                key,
                config.burst_capacity,
                tokens_requested,
                config.requests_per_minute,
                config.window_size_seconds,
                current_time
            )
            
            allowed, remaining, reset_time, capacity = result
            allowed = bool(allowed)
            
            # Create result object
            rate_limit_result = RateLimitResult(
                allowed=allowed,
                remaining=int(remaining),
                reset_time=float(reset_time),
                limit=capacity,
                retry_after=int(reset_time - current_time) if not allowed else None
            )
            
            # Log rate limit check
            logger.debug(
                f"Rate limit check: type={limit_type.value}, "
                f"identifier={identifier}, allowed={allowed}, "
                f"remaining={remaining}, reset_time={reset_time}"
            )
            
            # Track metrics
            await self._track_metrics(limit_type, allowed, remaining)
            
            if not allowed:
                logger.warning(
                    f"Rate limit exceeded: type={limit_type.value}, "
                    f"identifier={identifier}, retry_after={rate_limit_result.retry_after}"
                )
                raise RateLimitError(
                    f"Rate limit exceeded for {limit_type.value}: {identifier}",
                    rate_limit_result
                )
            
            return rate_limit_result
            
        except RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")
            # Graceful degradation - allow request but log error
            return self._create_fallback_result(config_override or self.configs[limit_type])
        except Exception as e:
            logger.error(f"Unexpected error in rate limiting: {e}")
            # Graceful degradation
            return self._create_fallback_result(config_override or self.configs[limit_type])
    
    async def get_rate_limit_status(
        self,
        identifier: str,
        limit_type: RateLimitType
    ) -> RateLimitResult:
        """
        Get current rate limit status without consuming tokens.
        
        Args:
            identifier: Unique identifier
            limit_type: Type of rate limit
            
        Returns:
            Current rate limit status
        """
        try:
            config = self.configs[limit_type]
            current_time = time.time()
            key = self._generate_key(limit_type, identifier)
            
            redis_client = await self.redis_manager.get_connection()
            
            # Get current bucket state
            bucket_data = await redis_client.hmget(key, 'tokens', 'last_refill')
            
            if not bucket_data[0]:
                # Bucket doesn't exist, return full capacity
                return RateLimitResult(
                    allowed=True,
                    remaining=config.burst_capacity,
                    reset_time=current_time,
                    limit=config.burst_capacity
                )
            
            current_tokens = float(bucket_data[0])
            last_refill = float(bucket_data[1] or current_time)
            
            # Calculate current tokens after refill
            time_elapsed = current_time - last_refill
            tokens_to_add = time_elapsed * config.requests_per_minute / config.window_size_seconds
            current_tokens = min(config.burst_capacity, current_tokens + tokens_to_add)
            
            # Calculate reset time
            reset_time = current_time + ((config.burst_capacity - current_tokens) * 
                                       config.window_size_seconds / config.requests_per_minute)
            
            return RateLimitResult(
                allowed=current_tokens >= 1,
                remaining=int(current_tokens),
                reset_time=reset_time,
                limit=config.burst_capacity
            )
            
        except Exception as e:
            logger.error(f"Error getting rate limit status: {e}")
            config = self.configs[limit_type]
            return RateLimitResult(
                allowed=True,
                remaining=config.burst_capacity,
                reset_time=time.time(),
                limit=config.burst_capacity
            )
    
    async def reset_rate_limit(
        self,
        identifier: str,
        limit_type: RateLimitType
    ) -> bool:
        """
        Reset rate limit for a specific identifier.
        
        Args:
            identifier: Unique identifier to reset
            limit_type: Type of rate limit
            
        Returns:
            True if reset successful, False otherwise
        """
        try:
            key = self._generate_key(limit_type, identifier)
            redis_client = await self.redis_manager.get_connection()
            
            await redis_client.delete(key)
            
            logger.info(f"Reset rate limit: type={limit_type.value}, identifier={identifier}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
            return False
    
    async def configure_rate_limit(
        self,
        limit_type: RateLimitType,
        config: RateLimitConfig
    ) -> None:
        """
        Update rate limit configuration.
        
        Args:
            limit_type: Type of rate limit to configure
            config: New configuration
        """
        self.configs[limit_type] = config
        logger.info(f"Updated rate limit config: type={limit_type.value}, config={config}")
    
    def _generate_key(self, limit_type: RateLimitType, identifier: str) -> str:
        """Generate Redis key for rate limit bucket."""
        return f"rate_limit:{limit_type.value}:{identifier}"
    
    def _create_fallback_result(self, config: RateLimitConfig) -> RateLimitResult:
        """Create fallback result when Redis is unavailable."""
        current_time = time.time()
        return RateLimitResult(
            allowed=True,  # Allow request in fallback mode
            remaining=config.burst_capacity,
            reset_time=current_time + config.window_size_seconds,
            limit=config.burst_capacity
        )
    
    async def _track_metrics(
        self,
        limit_type: RateLimitType,
        allowed: bool,
        remaining: int
    ) -> None:
        """Track rate limiting metrics."""
        try:
            # This would integrate with your metrics system
            # For now, just log key metrics
            logger.debug(
                f"Rate limit metrics: type={limit_type.value}, "
                f"allowed={allowed}, remaining={remaining}"
            )
        except Exception as e:
            logger.error(f"Error tracking rate limit metrics: {e}")
    
    async def get_global_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get global rate limiting statistics.
        
        Returns:
            Dictionary with rate limiting stats
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            
            stats = {
                "active_buckets": {},
                "total_requests": 0,
                "blocked_requests": 0
            }
            
            # Count active buckets for each type
            for limit_type in RateLimitType:
                pattern = f"rate_limit:{limit_type.value}:*"
                keys = await redis_client.keys(pattern)
                stats["active_buckets"][limit_type.value] = len(keys)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting rate limit stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_buckets(self) -> int:
        """
        Clean up expired rate limit buckets.
        
        Returns:
            Number of buckets cleaned up
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            
            # This is handled automatically by Redis TTL, but we can
            # optionally implement explicit cleanup for monitoring
            
            cleaned_count = 0
            current_time = time.time()
            
            for limit_type in RateLimitType:
                pattern = f"rate_limit:{limit_type.value}:*"
                async for key in redis_client.scan_iter(match=pattern):
                    bucket_data = await redis_client.hmget(key, 'last_refill')
                    if bucket_data[0]:
                        last_refill = float(bucket_data[0])
                        config = self.configs[limit_type]
                        
                        # If bucket hasn't been used in 2x window size, consider for cleanup
                        if current_time - last_refill > (config.window_size_seconds * 2):
                            await redis_client.delete(key)
                            cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired rate limit buckets")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up rate limit buckets: {e}")
            return 0


# Global rate limit service instance
_rate_limit_service: Optional[RateLimitService] = None


async def get_rate_limit_service() -> RateLimitService:
    """Get or create global rate limit service instance."""
    global _rate_limit_service
    
    if _rate_limit_service is None:
        _rate_limit_service = RateLimitService()
    
    return _rate_limit_service


# Dependency for FastAPI
async def rate_limit_dependency() -> RateLimitService:
    """FastAPI dependency for rate limiting service."""
    return await get_rate_limit_service()
