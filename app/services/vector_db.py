"""
Vector Database Service - Core Platform

This module provides the main interface for vector database operations,
supporting pluggable adapters for different vector database providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class VectorDBType(Enum):
    """Supported vector database types"""
    PINECONE = "pinecone"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


@dataclass
class VectorDocument:
    """Vector document representation"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Vector search result"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    db_type: VectorDBType
    connection_params: Dict[str, Any]
    index_name: str
    dimension: int
    similarity_metric: str = "cosine"


class VectorDBAdapter(ABC):
    """Abstract base class for vector database adapters"""
    
    @abstractmethod
    async def initialize(self, config: VectorDBConfig) -> bool:
        """Initialize the vector database connection"""
        pass
    
    @abstractmethod
    async def create_index(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a new vector index"""
        pass
    
    @abstractmethod
    async def delete_index(self, name: str) -> bool:
        """Delete a vector index"""
        pass
    
    @abstractmethod
    async def upsert_vectors(self, vectors: List[VectorDocument]) -> bool:
        """Insert or update vectors in the database"""
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[VectorDocument]:
        """Get a specific vector by ID"""
        pass
    
    @abstractmethod
    async def list_vectors(
        self, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[str]:
        """List vector IDs with optional filtering"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass


class VectorDBService:
    """Main vector database service with adapter pattern"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.adapter: Optional[VectorDBAdapter] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the vector database service"""
        try:
            # Import and instantiate the appropriate adapter
            adapter_class = self._get_adapter_class(self.config.db_type)
            self.adapter = adapter_class()
            
            # Initialize the adapter
            success = await self.adapter.initialize(self.config)
            self._initialized = success
            
            if success:
                logger.info(f"Vector database service initialized successfully with {self.config.db_type.value}")
            else:
                logger.error(f"Failed to initialize vector database service with {self.config.db_type.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing vector database service: {str(e)}")
            return False
    
    def _get_adapter_class(self, db_type: VectorDBType):
        """Get the adapter class for the specified database type"""
        if db_type == VectorDBType.PINECONE:
            from .vector_adapters.pinecone_adapter import PineconeAdapter
            return PineconeAdapter
        elif db_type == VectorDBType.CHROMA:
            from .vector_adapters.chroma_adapter import ChromaAdapter
            return ChromaAdapter
        elif db_type == VectorDBType.WEAVIATE:
            from .vector_adapters.weaviate_adapter import WeaviateAdapter
            return WeaviateAdapter
        elif db_type == VectorDBType.QDRANT:
            from .vector_adapters.qdrant_adapter import QdrantAdapter
            return QdrantAdapter
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
    
    def _ensure_initialized(self):
        """Ensure the service is initialized"""
        if not self._initialized or not self.adapter:
            raise RuntimeError("Vector database service not initialized. Call initialize() first.")
    
    async def create_index(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a new vector index"""
        self._ensure_initialized()
        return await self.adapter.create_index(name, dimension, metric)
    
    async def delete_index(self, name: str) -> bool:
        """Delete a vector index"""
        self._ensure_initialized()
        return await self.adapter.delete_index(name)
    
    async def store_document_vectors(self, documents: List[VectorDocument]) -> bool:
        """Store document vectors in the database"""
        self._ensure_initialized()
        
        try:
            # Validate vectors
            for doc in documents:
                if not doc.embedding or len(doc.embedding) != self.config.dimension:
                    raise ValueError(f"Invalid embedding dimension for document {doc.id}")
            
            # Store vectors
            success = await self.adapter.upsert_vectors(documents)
            
            if success:
                logger.info(f"Successfully stored {len(documents)} document vectors")
            else:
                logger.error(f"Failed to store document vectors")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing document vectors: {str(e)}")
            return False
    
    async def delete_document_vectors(self, document_ids: List[str]) -> bool:
        """Delete document vectors by IDs"""
        self._ensure_initialized()
        
        try:
            success = await self.adapter.delete_vectors(document_ids)
            
            if success:
                logger.info(f"Successfully deleted {len(document_ids)} document vectors")
            else:
                logger.error(f"Failed to delete document vectors")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {str(e)}")
            return False
    
    async def search_similar_documents(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[SearchResult]:
        """Search for similar documents"""
        self._ensure_initialized()
        
        try:
            # Validate query vector
            if len(query_vector) != self.config.dimension:
                raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match configured dimension {self.config.dimension}")
            
            # Perform search
            results = await self.adapter.search(
                query_vector=query_vector,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.score >= similarity_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} similar documents (threshold: {similarity_threshold})")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    async def get_document_vector(self, document_id: str) -> Optional[VectorDocument]:
        """Get a specific document vector by ID"""
        self._ensure_initialized()
        
        try:
            return await self.adapter.get_vector(document_id)
        except Exception as e:
            logger.error(f"Error getting document vector {document_id}: {str(e)}")
            return None
    
    async def list_document_ids(
        self, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[str]:
        """List document IDs with optional filtering"""
        self._ensure_initialized()
        
        try:
            return await self.adapter.list_vectors(filter_metadata, limit)
        except Exception as e:
            logger.error(f"Error listing document IDs: {str(e)}")
            return []
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        self._ensure_initialized()
        
        try:
            stats = await self.adapter.get_stats()
            stats["adapter_type"] = self.config.db_type.value
            stats["index_name"] = self.config.index_name
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector database"""
        try:
            if not self._initialized:
                return {"status": "unhealthy", "reason": "not_initialized"}
            
            # Try to get basic stats
            stats = await self.get_database_stats()
            
            if stats:
                return {
                    "status": "healthy",
                    "adapter_type": self.config.db_type.value,
                    "index_name": self.config.index_name,
                    "stats": stats
                }
            else:
                return {"status": "unhealthy", "reason": "unable_to_get_stats"}
                
        except Exception as e:
            return {
                "status": "unhealthy", 
                "reason": "exception",
                "error": str(e)
            }


# Factory function for easy service creation
def create_vector_db_service(
    db_type: str,
    connection_params: Dict[str, Any],
    index_name: str,
    dimension: int,
    similarity_metric: str = "cosine"
) -> VectorDBService:
    """Factory function to create a vector database service"""
    
    # Convert string to enum
    try:
        db_type_enum = VectorDBType(db_type.lower())
    except ValueError:
        raise ValueError(f"Unsupported vector database type: {db_type}")
    
    config = VectorDBConfig(
        db_type=db_type_enum,
        connection_params=connection_params,
        index_name=index_name,
        dimension=dimension,
        similarity_metric=similarity_metric
    )
    
    return VectorDBService(config)