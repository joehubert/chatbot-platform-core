from typing import Optional, Dict, Any, Tuple
from enum import Enum
import logging

from app.core.config import ModelType
from app.services.model_factory import ModelFactory, BaseModelClient, ModelResponse

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class ModelRouter:
    """Routes queries to appropriate model types based on analysis"""
    
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the model router"""
        if self._initialized:
            return
        
        # Ensure model factory is initialized
        await self.model_factory.initialize()
        self._initialized = True
        logger.info("Model router initialized")
    
    async def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ModelType, BaseModelClient]:
        """Route a query to the appropriate model type"""
        if not self._initialized:
            await self.initialize()
        
        # Analyze query to determine appropriate model type
        model_type = await self._analyze_query_for_type(query, context)
        
        # Get client for the determined type
        client = await self.model_factory.get_client_for_type(model_type)
        if not client:
            # Fallback to simple query model
            logger.warning(f"Failed to get client for {model_type}, falling back to simple query")
            model_type = ModelType.SIMPLE_QUERY
            client = await self.model_factory.get_client_for_type(model_type)
        
        if not client:
            raise RuntimeError("No model client available")
        
        logger.info(f"Routed query to model type: {model_type}")
        return model_type, client
    
    async def _analyze_query_for_type(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ModelType:
        """Analyze query to determine appropriate model type"""
        context = context or {}
        
        # Check if this is for relevance checking
        if context.get('purpose') == 'relevance_check':
            return ModelType.RELEVANCE
        
        # Check if this is for clarification
        if context.get('purpose') == 'clarification':
            return ModelType.CLARIFICATION
        
        # Check if this is for embedding
        if context.get('purpose') == 'embedding':
            return ModelType.EMBEDDING
        
        # Analyze query complexity for main chat responses
        complexity = self._assess_query_complexity(query)
        
        if complexity == QueryComplexity.SIMPLE:
            return ModelType.SIMPLE_QUERY
        elif complexity == QueryComplexity.COMPLEX:
            return ModelType.COMPLEX_QUERY
        else:
            return ModelType.SIMPLE_QUERY  # Default for medium complexity
    
    def _assess_query_complexity(self, query: str) -> QueryComplexity:
        """Simple heuristic-based complexity assessment"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Complex indicators
        complex_indicators = [
            'analyze', 'compare', 'evaluate', 'synthesize', 'explain in detail',
            'pros and cons', 'advantages and disadvantages', 'step by step',
            'comprehensive', 'thorough analysis'
        ]
        
        # Simple indicators
        simple_indicators = [
            'what is', 'who is', 'when', 'where', 'yes or no', 'true or false'
        ]
        
        if any(indicator in query_lower for indicator in complex_indicators) or word_count > 50:
            return QueryComplexity.COMPLEX
        elif any(indicator in query_lower for indicator in simple_indicators) or word_count < 10:
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MEDIUM
    
    async def get_model_for_type(self, model_type: ModelType) -> Optional[BaseModelClient]:
        """Get model client for a specific type"""
        return await self.model_factory.get_client_for_type(model_type)