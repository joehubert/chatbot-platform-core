"""
Knowledge Base Management Service

This module handles the high-level knowledge base operations including
document management, RAG queries, and integration with the vector database.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import hashlib

from .vector_db import VectorDBService, VectorDocument, SearchResult
from .document_processor import DocumentProcessor, ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime
    expires_at: Optional[datetime]
    category: Optional[str]
    tags: List[str]
    chunk_count: int
    processed: bool
    vector_ids: List[str]


@dataclass
class QueryResult:
    """RAG query result"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_embedding: List[float]
    processing_time_ms: float


class KnowledgeBaseService:
    """Knowledge base management service with RAG capabilities"""
    
    def __init__(
        self,
        vector_db_service: VectorDBService,
        document_processor: DocumentProcessor,
        default_expiration_days: int = 365
    ):
        self.vector_db = vector_db_service
        self.doc_processor = document_processor
        self.default_expiration_days = default_expiration_days
        self._document_metadata: Dict[str, DocumentMetadata] = {}
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the knowledge base service"""
        try:
            # Initialize vector database
            if not await self.vector_db.initialize():
                logger.error("Failed to initialize vector database")
                return False
            
            # Initialize document processor
            if not await self.doc_processor.initialize():
                logger.error("Failed to initialize document processor")
                return False
            
            # Load existing document metadata (in production, this would come from a database)
            await self._load_document_metadata()
            
            self._initialized = True
            logger.info("Knowledge base service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base service: {str(e)}")
            return False
    
    async def _load_document_metadata(self):
        """Load document metadata from storage"""
        # In a production system, this would load from a database
        # For now, this is a placeholder for the metadata loading logic
        try:
            # Get document IDs from vector database
            document_ids = await self.vector_db.list_document_ids()
            
            # For each document ID, try to reconstruct metadata
            # This is a simplified approach - in production, store metadata in database
            for doc_id in document_ids:
                vector_doc = await self.vector_db.get_document_vector(doc_id)
                if vector_doc and vector_doc.metadata:
                    metadata = self._create_document_metadata_from_vector(doc_id, vector_doc.metadata)
                    if metadata:
                        self._document_metadata[doc_id] = metadata
            
            logger.info(f"Loaded metadata for {len(self._document_metadata)} documents")
            
        except Exception as e:
            logger.warning(f"Failed to load document metadata: {str(e)}")
    
    def _create_document_metadata_from_vector(self, doc_id: str, vector_metadata: Dict[str, Any]) -> Optional[DocumentMetadata]:
        """Create DocumentMetadata from vector metadata"""
        try:
            return DocumentMetadata(
                document_id=doc_id,
                filename=vector_metadata.get('filename', 'unknown'),
                content_type=vector_metadata.get('content_type', 'unknown'),
                size_bytes=vector_metadata.get('size_bytes', 0),
                uploaded_at=datetime.fromisoformat(vector_metadata.get('uploaded_at', datetime.utcnow().isoformat())),
                expires_at=datetime.fromisoformat(vector_metadata['expires_at']) if vector_metadata.get('expires_at') else None,
                category=vector_metadata.get('category'),
                tags=vector_metadata.get('tags', []),
                chunk_count=vector_metadata.get('chunk_count', 1),
                processed=vector_metadata.get('processed', True),
                vector_ids=vector_metadata.get('vector_ids', [doc_id])
            )
        except Exception as e:
            logger.warning(f"Failed to create metadata for document {doc_id}: {str(e)}")
            return None
    
    def _ensure_initialized(self):
        """Ensure the service is initialized"""
        if not self._initialized:
            raise RuntimeError("Knowledge base service not initialized. Call initialize() first.")
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        expiration_days: Optional[int] = None
    ) -> str:
        """Upload and process a document"""
        self._ensure_initialized()
        
        try:
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Set expiration date
            exp_days = expiration_days or self.default_expiration_days
            expires_at = datetime.utcnow() + timedelta(days=exp_days)
            
            # Process the document
            processed_doc = await self.doc_processor.process_document(
                file_content=file_content,
                filename=filename,
                content_type=content_type
            )
            
            if not processed_doc:
                raise ValueError("Document processing failed")
            
            # Create vector documents for each chunk
            vector_documents = []
            vector_ids = []
            
            for i, chunk in enumerate(processed_doc.chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                vector_ids.append(chunk_id)
                
                # Prepare metadata for each chunk
                chunk_metadata = {
                    'document_id': document_id,
                    'filename': filename,
                    'content_type': content_type,
                    'size_bytes': len(file_content),
                    'uploaded_at': datetime.utcnow().isoformat(),
                    'expires_at': expires_at.isoformat(),
                    'category': category,
                    'tags': tags or [],
                    'chunk_index': i,
                    'chunk_count': len(processed_doc.chunks),
                    'processed': True,
                    'vector_ids': vector_ids
                }
                
                vector_doc = VectorDocument(
                    id=chunk_id,
                    content=chunk.content,
                    embedding=chunk.embedding,
                    metadata=chunk_metadata
                )
                vector_documents.append(vector_doc)
            
            # Store vectors in the database
            success = await self.vector_db.store_document_vectors(vector_documents)
            
            if not success:
                raise RuntimeError("Failed to store document vectors")
            
            # Store document metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                content_type=content_type,
                size_bytes=len(file_content),
                uploaded_at=datetime.utcnow(),
                expires_at=expires_at,
                category=category,
                tags=tags or [],
                chunk_count=len(processed_doc.chunks),
                processed=True,
                vector_ids=vector_ids
            )
            
            self._document_metadata[document_id] = metadata
            
            logger.info(f"Successfully uploaded document {document_id} with {len(processed_doc.chunks)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to upload document {filename}: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its vectors"""
        self._ensure_initialized()
        
        try:
            # Get document metadata
            metadata = self._document_metadata.get(document_id)
            if not metadata:
                logger.warning(f"Document {document_id} not found in metadata")
                return False
            
            # Delete vectors from vector database
            success = await self.vector_db.delete_document_vectors(metadata.vector_ids)
            
            if success:
                # Remove from metadata
                del self._document_metadata[document_id]
                logger.info(f"Successfully deleted document {document_id}")
                return True
            else:
                logger.error(f"Failed to delete vectors for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata"""
        self._ensure_initialized()
        return self._document_metadata.get(document_id)
    
    async def list_documents(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        expired_only: bool = False
    ) -> List[DocumentMetadata]:
        """List documents with optional filtering"""
        self._ensure_initialized()
        
        documents = list(self._document_metadata.values())
        
        # Apply filters
        if category:
            documents = [doc for doc in documents if doc.category == category]
        
        if tags:
            documents = [
                doc for doc in documents 
                if any(tag in doc.tags for tag in tags)
            ]
        
        if expired_only:
            now = datetime.utcnow()
            documents = [
                doc for doc in documents 
                if doc.expires_at and doc.expires_at < now
            ]
        
        return documents
    
    async def cleanup_expired_documents(self) -> int:
        """Remove expired documents from the knowledge base"""
        self._ensure_initialized()
        
        try:
            now = datetime.utcnow()
            expired_docs = [
                doc for doc in self._document_metadata.values()
                if doc.expires_at and doc.expires_at < now
            ]
            
            cleanup_count = 0
            for doc in expired_docs:
                if await self.delete_document(doc.document_id):
                    cleanup_count += 1
            
            logger.info(f"Cleaned up {cleanup_count} expired documents")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired documents: {str(e)}")
            return 0
    
    async def query_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        category_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None
    ) -> QueryResult:
        """Query the knowledge base using RAG"""
        self._ensure_initialized()
        
        start_time = datetime.utcnow()
        
        try:
            # Generate embedding for the query
            query_embedding = await self.doc_processor.generate_embedding(query)
            
            if not query_embedding:
                raise ValueError("Failed to generate query embedding")
            
            # Prepare metadata filter
            filter_metadata = {}
            if category_filter:
                filter_metadata['category'] = category_filter
            if tag_filter:
                filter_metadata['tags'] = {"$in": tag_filter}
            
            # Search for similar documents
            search_results = await self.vector_db.search_similar_documents(
                query_vector=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata if filter_metadata else None,
                similarity_threshold=similarity_threshold
            )
            
            # Process search results
            sources = []
            context_chunks = []
            
            for result in search_results:
                sources.append({
                    'document_id': result.metadata.get('document_id'),
                    'filename': result.metadata.get('filename'),
                    'chunk_index': result.metadata.get('chunk_index'),
                    'similarity_score': result.score,
                    'content_preview': result.content[:200] + "..." if len(result.content) > 200 else result.content
                })
                context_chunks.append(result.content)
            
            # Generate answer using context (this would integrate with LLM)
            if context_chunks:
                # For now, return the most relevant chunk as the answer
                # In production, this would be processed by an LLM
                answer = self._generate_answer_from_context(query, context_chunks)
                confidence = max(result.score for result in search_results) if search_results else 0.0
            else:
                answer = "I couldn't find relevant information in the knowledge base to answer your question."
                confidence = 0.0
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return QueryResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
                query_embedding=query_embedding,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to query knowledge base: {str(e)}")
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return QueryResult(
                answer="An error occurred while searching the knowledge base.",
                sources=[],
                confidence=0.0,
                query_embedding=[],
                processing_time_ms=processing_time
            )
    
    def _generate_answer_from_context(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer from context chunks (placeholder for LLM integration)"""
        # This is a simplified implementation
        # In production, this would use an LLM to generate a proper answer
        
        if not context_chunks:
            return "No relevant information found."
        
        # For now, return the most relevant chunk with some basic formatting
        best_chunk = context_chunks[0]  # First chunk is typically most relevant
        
        # Basic answer formatting
        if len(best_chunk) > 500:
            best_chunk = best_chunk[:500] + "..."
        
        return f"Based on the available information: {best_chunk}"
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        self._ensure_initialized()
        
        try:
            # Document statistics
            total_docs = len(self._document_metadata)
            processed_docs = sum(1 for doc in self._document_metadata.values() if doc.processed)
            
            # Expiration statistics
            now = datetime.utcnow()
            expired_docs = sum(
                1 for doc in self._document_metadata.values() 
                if doc.expires_at and doc.expires_at < now
            )
            
            expiring_soon = sum(
                1 for doc in self._document_metadata.values()
                if doc.expires_at and doc.expires_at < now + timedelta(days=7)
            )
            
            # Category statistics
            categories = {}
            total_chunks = 0
            for doc in self._document_metadata.values():
                if doc.category:
                    categories[doc.category] = categories.get(doc.category, 0) + 1
                total_chunks += doc.chunk_count
            
            # Vector database statistics
            vector_stats = await self.vector_db.get_database_stats()
            
            return {
                'documents': {
                    'total': total_docs,
                    'processed': processed_docs,
                    'expired': expired_docs,
                    'expiring_soon': expiring_soon
                },
                'chunks': {
                    'total': total_chunks,
                    'average_per_document': total_chunks / total_docs if total_docs > 0 else 0
                },
                'categories': categories,
                'vector_database': vector_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge base statistics: {str(e)}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the knowledge base service"""
        try:
            if not self._initialized:
                return {"status": "unhealthy", "reason": "not_initialized"}
            
            # Check vector database health
            vector_health = await self.vector_db.health_check()
            
            # Check document processor health
            processor_health = await self.doc_processor.health_check()
            
            # Overall health status
            overall_status = "healthy"
            if vector_health.get("status") != "healthy" or processor_health.get("status") != "healthy":
                overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "components": {
                    "vector_database": vector_health,
                    "document_processor": processor_health
                },
                "metadata": {
                    "total_documents": len(self._document_metadata),
                    "processed_documents": sum(1 for doc in self._document_metadata.values() if doc.processed)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": "exception",
                "error": str(e)
            }
    
    async def rebuild_index(self) -> bool:
        """Rebuild the vector index from stored documents"""
        self._ensure_initialized()
        
        try:
            logger.info("Starting knowledge base index rebuild")
            
            # Get all document metadata
            documents = list(self._document_metadata.values())
            
            # Process each document
            rebuilt_count = 0
            for doc_metadata in documents:
                try:
                    # This would require re-processing the original document
                    # For now, this is a placeholder implementation
                    logger.info(f"Would rebuild document {doc_metadata.document_id}")
                    rebuilt_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to rebuild document {doc_metadata.document_id}: {str(e)}")
            
            logger.info(f"Index rebuild completed. Processed {rebuilt_count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            return False
    
    async def update_document_expiration(self, document_id: str, new_expiration: datetime) -> bool:
        """Update document expiration date"""
        self._ensure_initialized()
        
        try:
            # Update metadata
            if document_id in self._document_metadata:
                self._document_metadata[document_id].expires_at = new_expiration
                
                # Update vector metadata (would need to update all chunks)
                metadata = self._document_metadata[document_id]
                for vector_id in metadata.vector_ids:
                    vector_doc = await self.vector_db.get_document_vector(vector_id)
                    if vector_doc:
                        vector_doc.metadata['expires_at'] = new_expiration.isoformat()
                        await self.vector_db.store_document_vectors([vector_doc])
                
                logger.info(f"Updated expiration for document {document_id}")
                return True
            else:
                logger.warning(f"Document {document_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update document expiration: {str(e)}")
            return False


# Factory function for easy service creation
def create_knowledge_base_service(
    vector_db_service: VectorDBService,
    document_processor: DocumentProcessor,
    default_expiration_days: int = 365
) -> KnowledgeBaseService:
    """Factory function to create a knowledge base service"""
    return KnowledgeBaseService(
        vector_db_service=vector_db_service,
        document_processor=document_processor,
        default_expiration_days=default_expiration_days
    )