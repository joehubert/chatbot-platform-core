"""
Cache Service
Implementation of Redis-based semantic caching using vector similarity.
Supports conversation summaries, query caching, and automatic cache invalidation.
"""

import json
import time
import hashlib
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

import redis.asyncio as redis
import numpy as np
from redis.exceptions import RedisError

from ..utils.redis_utils import RedisManager
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheType(Enum):
    """Cache entry types"""
    CONVERSATION_SUMMARY = "conversation_summary"
    QUERY_RESPONSE = "query_response"
    DOCUMENT_EMBEDDING = "document_embedding"
    SESSION_DATA = "session_data"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass
class CacheEntry:
    """Cache entry data structure"""
    key: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    cache_type: CacheType
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        return {
            'key': self.key,
            'content': self.content,
            'embedding': json.dumps(self.embedding) if self.embedding else None,
            'metadata': json.dumps(self.metadata),
            'cache_type': self.cache_type.value,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from Redis dictionary"""
        return cls(
            key=data['key'],
            content=data['content'],
            embedding=json.loads(data['embedding']) if data.get('embedding') else None,
            metadata=json.loads(data['metadata']),
            cache_type=CacheType(data['cache_type']),
            created_at=float(data['created_at']),
            expires_at=float(data['expires_at']) if data.get('expires_at') else None,
            access_count=int(data.get('access_count', 0)),
            last_accessed=float(data['last_accessed']) if data.get('last_accessed') else None
        )


@dataclass
class CacheSearchResult:
    """Result of cache search operation"""
    hit: bool
    entry: Optional[CacheEntry]
    similarity_score: Optional[float]
    search_time_ms: float


class SemanticCacheService:
    """
    Redis-based semantic cache service using vector similarity.
    
    Features:
    - Vector similarity-based cache lookup
    - Conversation summary caching
    - Automatic cache invalidation
    - Configurable similarity thresholds
    - Cache performance metrics
    - Memory-efficient storage with compression
    """
    
    def __init__(self, redis_manager: Optional[RedisManager] = None):
        """
        Initialize semantic cache service.
        
        Args:
            redis_manager: Redis connection manager
        """
        self.redis_manager = redis_manager or RedisManager()
        
        # Configuration from settings
        self.similarity_threshold = settings.CACHE_SIMILARITY_THRESHOLD
        self.default_ttl_hours = settings.CACHE_TTL_HOURS
        self.max_embedding_dimension = settings.CACHE_MAX_EMBEDDING_DIMENSION
        self.enable_compression = settings.CACHE_ENABLE_COMPRESSION
        
        # Cache key prefixes
        self.key_prefixes = {
            CacheType.CONVERSATION_SUMMARY: "cache:conv_summary:",
            CacheType.QUERY_RESPONSE: "cache:query_resp:",
            CacheType.DOCUMENT_EMBEDDING: "cache:doc_embed:",
            CacheType.SESSION_DATA: "cache:session:",
            CacheType.KNOWLEDGE_BASE: "cache:kb:",
        }
        
        # Performance tracking
        self.performance_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_searches": 0,
            "avg_search_time_ms": 0.0
        }
    
    async def get_cache_entry(
        self,
        query_embedding: List[float],
        cache_type: CacheType,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> CacheSearchResult:
        """
        Search for cache entry using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            cache_type: Type of cache to search
            similarity_threshold: Override default similarity threshold
            metadata_filters: Additional metadata filters
            
        Returns:
            CacheSearchResult with hit status and entry if found
        """
        start_time = time.time()
        threshold = similarity_threshold or self.similarity_threshold
        
        try:
            redis_client = await self.redis_manager.get_connection()
            
            # Get all cache keys for this type
            pattern = f"{self.key_prefixes[cache_type]}*"
            cache_keys = await redis_client.keys(pattern)
            
            if not cache_keys:
                return CacheSearchResult(
                    hit=False,
                    entry=None,
                    similarity_score=None,
                    search_time_ms=(time.time() - start_time) * 1000
                )
            
            best_match = None
            best_similarity = 0.0
            
            # Search through all cache entries
            for cache_key in cache_keys:
                try:
                    # Get cache entry data
                    entry_data = await redis_client.hgetall(cache_key)
                    if not entry_data:
                        continue
                    
                    # Convert to CacheEntry object
                    entry = CacheEntry.from_dict(entry_data)
                    
                    # Check if entry is expired
                    if entry.expires_at and time.time() > entry.expires_at:
                        # Clean up expired entry
                        await redis_client.delete(cache_key)
                        continue
                    
                    # Apply metadata filters
                    if metadata_filters and not self._matches_filters(entry.metadata, metadata_filters):
                        continue
                    
                    # Calculate similarity if embedding exists
                    if entry.embedding:
                        similarity = self._calculate_similarity(query_embedding, entry.embedding)
                        
                        if similarity > best_similarity and similarity >= threshold:
                            best_similarity = similarity
                            best_match = entry
                
                except Exception as e:
                    logger.warning(f"Error processing cache entry {cache_key}: {e}")
                    continue
            
            search_time_ms = (time.time() - start_time) * 1000
            
            if best_match:
                # Update access statistics
                await self._update_access_stats(best_match.key)
                
                # Track performance metrics
                self.performance_metrics["cache_hits"] += 1
                
                logger.debug(
                    f"Cache hit: type={cache_type.value}, "
                    f"similarity={best_similarity:.3f}, "
                    f"search_time={search_time_ms:.2f}ms"
                )
                
                return CacheSearchResult(
                    hit=True,
                    entry=best_match,
                    similarity_score=best_similarity,
                    search_time_ms=search_time_ms
                )
            else:
                # Track cache miss
                self.performance_metrics["cache_misses"] += 1
                
                logger.debug(
                    f"Cache miss: type={cache_type.value}, "
                    f"search_time={search_time_ms:.2f}ms"
                )
                
                return CacheSearchResult(
                    hit=False,
                    entry=None,
                    similarity_score=None,
                    search_time_ms=search_time_ms
                )
        
        except RedisError as e:
            logger.error(f"Redis error in cache lookup: {e}")
            return CacheSearchResult(
                hit=False,
                entry=None,
                similarity_score=None,
                search_time_ms=(time.time() - start_time) * 1000
            )
        
        finally:
            self.performance_metrics["total_searches"] += 1
            self._update_avg_search_time(search_time_ms)
    
    async def store_cache_entry(
        self,
        content: str,
        embedding: List[float],
        cache_type: CacheType,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[float] = None,
        custom_key: Optional[str] = None
    ) -> str:
        """
        Store a new cache entry.
        
        Args:
            content: Content to cache
            embedding: Vector embedding for similarity search
            cache_type: Type of cache entry
            metadata: Additional metadata
            ttl_hours: Time to live in hours
            custom_key: Custom cache key (optional)
            
        Returns:
            Cache key
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            
            # Generate cache key
            if custom_key:
                cache_key = f"{self.key_prefixes[cache_type]}{custom_key}"
            else:
                content_hash = hashlib.md5(content.encode()).hexdigest()
                timestamp = int(time.time())
                cache_key = f"{self.key_prefixes[cache_type]}{timestamp}_{content_hash[:8]}"
            
            # Calculate expiration
            ttl = ttl_hours or self.default_ttl_hours
            expires_at = time.time() + (ttl * 3600) if ttl > 0 else None
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                cache_type=cache_type,
                created_at=time.time(),
                expires_at=expires_at
            )
            
            # Store in Redis
            await redis_client.hset(cache_key, mapping=entry.to_dict())
            
            # Set TTL if specified
            if ttl > 0:
                await redis_client.expire(cache_key, int(ttl * 3600))
            
            logger.debug(f"Stored cache entry: key={cache_key}, type={cache_type.value}")
            
            return cache_key
        
        except Exception as e:
            logger.error(f"Error storing cache entry: {e}")
            raise
    
    async def invalidate_cache(
        self,
        cache_type: Optional[CacheType] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        older_than_hours: Optional[float] = None
    ) -> int:
        """
        Invalidate cache entries based on criteria.
        
        Args:
            cache_type: Specific cache type to invalidate
            metadata_filters: Metadata filters for selective invalidation
            older_than_hours: Invalidate entries older than specified hours
            
        Returns:
            Number of entries invalidated
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            invalidated_count = 0
            
            # Determine patterns to search
            if cache_type:
                patterns = [f"{self.key_prefixes[cache_type]}*"]
            else:
                patterns = [f"{prefix}*" for prefix in self.key_prefixes.values()]
            
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600) if older_than_hours else None
            
            for pattern in patterns:
                cache_keys = await redis_client.keys(pattern)
                
                for cache_key in cache_keys:
                    try:
                        entry_data = await redis_client.hgetall(cache_key)
                        if not entry_data:
                            continue
                        
                        entry = CacheEntry.from_dict(entry_data)
                        
                        # Check age filter
                        if cutoff_time and entry.created_at > cutoff_time:
                            continue
                        
                        # Check metadata filters
                        if metadata_filters and not self._matches_filters(entry.metadata, metadata_filters):
                            continue
                        
                        # Delete the cache entry
                        await redis_client.delete(cache_key)
                        invalidated_count += 1
                    
                    except Exception as e:
                        logger.warning(f"Error invalidating cache entry {cache_key}: {e}")
                        continue
            
            logger.info(f"Invalidated {invalidated_count} cache entries")
            return invalidated_count
        
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            
            stats = {
                "total_entries": 0,
                "entries_by_type": {},
                "memory_usage_bytes": 0,
                "hit_rate": 0.0,
                "performance_metrics": self.performance_metrics.copy()
            }
            
            # Count entries by type
            for cache_type in CacheType:
                pattern = f"{self.key_prefixes[cache_type]}*"
                keys = await redis_client.keys(pattern)
                stats["entries_by_type"][cache_type.value] = len(keys)
                stats["total_entries"] += len(keys)
            
            # Calculate hit rate
            total_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
            if total_requests > 0:
                stats["hit_rate"] = self.performance_metrics["cache_hits"] / total_requests
            
            # Estimate memory usage (approximate)
            for cache_type in CacheType:
                pattern = f"{self.key_prefixes[cache_type]}*"
                async for key in redis_client.scan_iter(match=pattern):
                    try:
                        memory_usage = await redis_client.memory_usage(key)
                        if memory_usage:
                            stats["memory_usage_bytes"] += memory_usage
                    except:
                        # Fallback if memory_usage command not available
                        pass
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_entries(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            cleaned_count = 0
            current_time = time.time()
            
            # Check all cache types
            for cache_type in CacheType:
                pattern = f"{self.key_prefixes[cache_type]}*"
                
                async for cache_key in redis_client.scan_iter(match=pattern):
                    try:
                        entry_data = await redis_client.hgetall(cache_key)
                        if not entry_data:
                            continue
                        
                        entry = CacheEntry.from_dict(entry_data)
                        
                        # Check if expired
                        if entry.expires_at and current_time > entry.expires_at:
                            await redis_client.delete(cache_key)
                            cleaned_count += 1
                    
                    except Exception as e:
                        logger.warning(f"Error cleaning cache entry {cache_key}: {e}")
                        continue
            
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            return cleaned_count
        
        except Exception as e:
            logger.error(f"Error cleaning expired cache entries: {e}")
            return 0
    
    async def update_cache_entry(
        self,
        cache_key: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extend_ttl_hours: Optional[float] = None
    ) -> bool:
        """
        Update an existing cache entry.
        
        Args:
            cache_key: Cache key to update
            content: New content (optional)
            metadata: New metadata (optional)
            extend_ttl_hours: Extend TTL by specified hours
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            
            # Check if entry exists
            if not await redis_client.exists(cache_key):
                return False
            
            # Get current entry
            entry_data = await redis_client.hgetall(cache_key)
            entry = CacheEntry.from_dict(entry_data)
            
            # Update fields
            if content is not None:
                entry.content = content
            
            if metadata is not None:
                entry.metadata.update(metadata)
            
            if extend_ttl_hours is not None:
                if entry.expires_at:
                    entry.expires_at += extend_ttl_hours * 3600
                else:
                    entry.expires_at = time.time() + (extend_ttl_hours * 3600)
            
            # Store updated entry
            await redis_client.hset(cache_key, mapping=entry.to_dict())
            
            # Update TTL if needed
            if extend_ttl_hours is not None and entry.expires_at:
                ttl_seconds = int(entry.expires_at - time.time())
                if ttl_seconds > 0:
                    await redis_client.expire(cache_key, ttl_seconds)
            
            logger.debug(f"Updated cache entry: {cache_key}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating cache entry {cache_key}: {e}")
            return False
    
    async def delete_cache_entry(self, cache_key: str) -> bool:
        """
        Delete a specific cache entry.
        
        Args:
            cache_key: Key of entry to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            redis_client = await self.redis_manager.get_connection()
            result = await redis_client.delete(cache_key)
            
            if result:
                logger.debug(f"Deleted cache entry: {cache_key}")
                return True
            else:
                logger.warning(f"Cache entry not found: {cache_key}")
                return False
        
        except Exception as e:
            logger.error(f"Error deleting cache entry {cache_key}: {e}")
            return False
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1) / 2))
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Args:
            metadata: Entry metadata
            filters: Filter criteria
            
        Returns:
            True if matches, False otherwise
        """
        try:
            for key, value in filters.items():
                if key not in metadata:
                    return False
                
                if isinstance(value, list):
                    if metadata[key] not in value:
                        return False
                elif metadata[key] != value:
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error matching filters: {e}")
            return False
    
    async def _update_access_stats(self, cache_key: str) -> None:
        """Update access statistics for a cache entry."""
        try:
            redis_client = await self.redis_manager.get_connection()
            
            # Update access count and last accessed time
            await redis_client.hincrby(cache_key, 'access_count', 1)
            await redis_client.hset(cache_key, 'last_accessed', time.time())
        
        except Exception as e:
            logger.error(f"Error updating access stats for {cache_key}: {e}")
    
    def _update_avg_search_time(self, search_time_ms: float) -> None:
        """Update average search time metric."""
        try:
            total_searches = self.performance_metrics["total_searches"]
            current_avg = self.performance_metrics["avg_search_time_ms"]
            
            # Calculate new average
            new_avg = ((current_avg * (total_searches - 1)) + search_time_ms) / total_searches
            self.performance_metrics["avg_search_time_ms"] = new_avg
        
        except Exception as e:
            logger.error(f"Error updating average search time: {e}")


class ConversationCacheService:
    """
    Specialized cache service for conversation summaries.
    Built on top of SemanticCacheService with conversation-specific features.
    """
    
    def __init__(self, semantic_cache: Optional[SemanticCacheService] = None):
        """
        Initialize conversation cache service.
        
        Args:
            semantic_cache: Semantic cache service instance
        """
        self.semantic_cache = semantic_cache or SemanticCacheService()
    
    async def store_conversation_summary(
        self,
        conversation_id: str,
        summary: str,
        summary_embedding: List[float],
        context: Dict[str, Any],
        resolved: bool = True
    ) -> str:
        """
        Store a conversation summary in cache.
        
        Args:
            conversation_id: Unique conversation identifier
            summary: Conversation summary text
            summary_embedding: Embedding vector for the summary
            context: Additional context (user_id, session_id, etc.)
            resolved: Whether the conversation was resolved
            
        Returns:
            Cache key
        """
        metadata = {
            "conversation_id": conversation_id,
            "resolved": resolved,
            "timestamp": time.time(),
            **context
        }
        
        return await self.semantic_cache.store_cache_entry(
            content=summary,
            embedding=summary_embedding,
            cache_type=CacheType.CONVERSATION_SUMMARY,
            metadata=metadata,
            custom_key=f"conv_{conversation_id}"
        )
    
    async def find_similar_conversations(
        self,
        query_embedding: List[float],
        similarity_threshold: Optional[float] = None,
        resolved_only: bool = True,
        limit: int = 5
    ) -> List[Tuple[CacheEntry, float]]:
        """
        Find similar resolved conversations.
        
        Args:
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity score
            resolved_only: Only return resolved conversations
            limit: Maximum number of results
            
        Returns:
            List of (CacheEntry, similarity_score) tuples
        """
        metadata_filters = {"resolved": True} if resolved_only else None
        
        # This is a simplified version - in practice, you'd want to
        # implement a more efficient search that can return multiple results
        result = await self.semantic_cache.get_cache_entry(
            query_embedding=query_embedding,
            cache_type=CacheType.CONVERSATION_SUMMARY,
            similarity_threshold=similarity_threshold,
            metadata_filters=metadata_filters
        )
        
        if result.hit and result.entry and result.similarity_score:
            return [(result.entry, result.similarity_score)]
        
        return []


# Global cache service instances
_semantic_cache_service: Optional[SemanticCacheService] = None
_conversation_cache_service: Optional[ConversationCacheService] = None


async def get_semantic_cache_service() -> SemanticCacheService:
    """Get or create global semantic cache service instance."""
    global _semantic_cache_service
    
    if _semantic_cache_service is None:
        _semantic_cache_service = SemanticCacheService()
    
    return _semantic_cache_service


async def get_conversation_cache_service() -> ConversationCacheService:
    """Get or create global conversation cache service instance."""
    global _conversation_cache_service
    
    if _conversation_cache_service is None:
        semantic_cache = await get_semantic_cache_service()
        _conversation_cache_service = ConversationCacheService(semantic_cache)
    
    return _conversation_cache_service


# Dependencies for FastAPI
async def semantic_cache_dependency() -> SemanticCacheService:
    """FastAPI dependency for semantic cache service."""
    return await get_semantic_cache_service()


async def conversation_cache_dependency() -> ConversationCacheService:
    """FastAPI dependency for conversation cache service."""
    return await get_conversation_cache_service()
