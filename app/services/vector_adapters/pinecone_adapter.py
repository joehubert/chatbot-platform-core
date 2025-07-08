"""
Pinecone Vector Database Adapter

This module implements the Pinecone adapter for the vector database service.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

try:
    import pinecone
    from pinecone import Pinecone
except ImportError:
    pinecone = None
    Pinecone = None

from ..vector_db import VectorDBAdapter, VectorDBConfig, VectorDocument, SearchResult

logger = logging.getLogger(__name__)


class PineconeAdapter(VectorDBAdapter):
    """Pinecone implementation of the vector database adapter"""
    
    def __init__(self):
        self.client: Optional[Pinecone] = None
        self.index = None
        self.config: Optional[VectorDBConfig] = None
        self._initialized = False
    
    async def initialize(self, config: VectorDBConfig) -> bool:
        """Initialize the Pinecone connection"""
        if pinecone is None:
            logger.error("Pinecone package not installed. Install with: pip install pinecone-client")
            return False
        
        try:
            self.config = config
            
            # Get connection parameters
            api_key = config.connection_params.get('api_key')
            environment = config.connection_params.get('environment')
            
            if not api_key:
                logger.error("Pinecone API key not provided in connection_params")
                return False
            
            # Initialize Pinecone client
            self.client = Pinecone(api_key=api_key)
            
            # Connect to the index
            try:
                self.index = self.client.Index(config.index_name)
                # Test the connection by getting index stats
                await self._run_in_thread(self.index.describe_index_stats)
                logger.info(f"Successfully connected to Pinecone index: {config.index_name}")
            except Exception as e:
                logger.warning(f"Index {config.index_name} may not exist: {str(e)}")
                # Index might not exist yet, which is okay
                self.index = None
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone adapter: {str(e)}")
            return False
    
    async def _run_in_thread(self, func, *args, **kwargs):
        """Run synchronous Pinecone operations in a thread"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _ensure_initialized(self):
        """Ensure the adapter is initialized"""
        if not self._initialized or not self.client:
            raise RuntimeError("Pinecone adapter not initialized")
    
    async def create_index(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a new Pinecone index"""
        self._ensure_initialized()
        
        try:
            # Map metric names to Pinecone format
            pinecone_metric = {
                "cosine": "cosine",
                "euclidean": "euclidean", 
                "dotproduct": "dotproduct"
            }.get(metric.lower(), "cosine")
            
            # Create index
            await self._run_in_thread(
                self.client.create_index,
                name=name,
                dimension=dimension,
                metric=pinecone_metric
            )
            
            # Wait for index to be ready
            await asyncio.sleep(5)
            
            # Connect to the newly created index
            self.index = self.client.Index(name)
            
            logger.info(f"Successfully created Pinecone index: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index {name}: {str(e)}")
            return False
    
    async def delete_index(self, name: str) -> bool:
        """Delete a Pinecone index"""
        self._ensure_initialized()
        
        try:
            await self._run_in_thread(self.client.delete_index, name)
            
            # Clear the index reference if it was the current index
            if self.index and hasattr(self.index, '_index_name') and self.index._index_name == name:
                self.index = None
            
            logger.info(f"Successfully deleted Pinecone index: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index {name}: {str(e)}")
            return False
    
    async def upsert_vectors(self, vectors: List[VectorDocument]) -> bool:
        """Insert or update vectors in Pinecone"""
        self._ensure_initialized()
        
        if not self.index:
            logger.error("No Pinecone index available for upsert operation")
            return False
        
        try:
            # Convert VectorDocument objects to Pinecone format
            pinecone_vectors = []
            for doc in vectors:
                # Prepare metadata with additional fields
                metadata = doc.metadata.copy()
                metadata.update({
                    'content': doc.content,
                    'updated_at': datetime.utcnow().isoformat()
                })
                
                pinecone_vectors.append({
                    'id': doc.id,
                    'values': doc.embedding,
                    'metadata': metadata
                })
            
            # Upsert vectors in batches (Pinecone has batch size limits)
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                await self._run_in_thread(self.index.upsert, vectors=batch)
            
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {str(e)}")
            return False
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs from Pinecone"""
        self._ensure_initialized()
        
        if not self.index:
            logger.error("No Pinecone index available for delete operation")
            return False
        
        try:
            # Delete vectors in batches
            batch_size = 1000  # Pinecone delete batch limit
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                await self._run_in_thread(self.index.delete, ids=batch)
            
            logger.info(f"Successfully deleted {len(ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {str(e)}")
            return False
    
    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in Pinecone"""
        self._ensure_initialized()
        
        if not self.index:
            logger.error("No Pinecone index available for search operation")
            return []
        
        try:
            # Prepare query parameters
            query_params = {
                'vector': query_vector,
                'top_k': top_k,
                'include_metadata': True,
                'include_values': False
            }
            
            if filter_metadata:
                query_params['filter'] = filter_metadata
            
            # Perform search
            response = await self._run_in_thread(self.index.query, **query_params)
            
            # Convert response to SearchResult objects
            results = []
            for match in response.matches:
                metadata = match.metadata or {}
                content = metadata.get('content', '')
                
                results.append(SearchResult(
                    id=match.id,
                    content=content,
                    score=float(match.score),
                    metadata=metadata
                ))
            
            logger.info(f"Pinecone search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Pinecone: {str(e)}")
            return []
    
    async def get_vector(self, vector_id: str) -> Optional[VectorDocument]:
        """Get a specific vector by ID from Pinecone"""
        self._ensure_initialized()
        
        if not self.index:
            logger.error("No Pinecone index available for get operation")
            return None
        
        try:
            # Fetch vector by ID
            response = await self._run_in_thread(
                self.index.fetch,
                ids=[vector_id]
            )
            
            if vector_id not in response.vectors:
                return None
            
            vector_data = response.vectors[vector_id]
            metadata = vector_data.metadata or {}
            content = metadata.get('content', '')
            
            return VectorDocument(
                id=vector_id,
                content=content,
                embedding=vector_data.values,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id} from Pinecone: {str(e)}")
            return None
    
    async def list_vectors(
        self, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[str]:
        """List vector IDs with optional filtering"""
        self._ensure_initialized()
        
        if not self.index:
            logger.error("No Pinecone index available for list operation")
            return []
        
        try:
            # Note: Pinecone doesn't have a direct "list all IDs" operation
            # This is a simplified implementation that uses query with a dummy vector
            # In practice, you might want to maintain a separate metadata store
            # or use Pinecone's list_ids method if available in newer versions
            
            # Get index stats to understand the namespace
            stats = await self._run_in_thread(self.index.describe_index_stats)
            
            # For now, return empty list as Pinecone doesn't easily support listing all IDs
            # This would need to be implemented differently based on specific requirements
            logger.warning("Pinecone list_vectors operation not fully implemented - requires vector enumeration strategy")
            return []
            
        except Exception as e:
            logger.error(f"Failed to list vectors from Pinecone: {str(e)}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone database statistics"""
        self._ensure_initialized()
        
        if not self.index:
            logger.error("No Pinecone index available for stats operation")
            return {}
        
        try:
            # Get index statistics
            stats = await self._run_in_thread(self.index.describe_index_stats)
            
            # Convert to dict format
            result = {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': {}
            }
            
            # Add namespace information
            if hasattr(stats, 'namespaces') and stats.namespaces:
                for namespace, ns_stats in stats.namespaces.items():
                    result['namespaces'][namespace] = {
                        'vector_count': ns_stats.vector_count
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {str(e)}")
            return {}