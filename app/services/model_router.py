"""
Model Router Service

Intelligent routing of queries to appropriate LLM models based on complexity,
cost optimization, and performance characteristics.
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
import re
from dataclasses import dataclass

from .model_factory import ModelProvider, ModelService, ModelResponse

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class QueryType(Enum):
    """Types of queries"""
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TECHNICAL = "technical"


@dataclass
class RoutingRule:
    """Routing rule for model selection"""
    complexity: QueryComplexity
    query_type: QueryType
    provider: ModelProvider
    model: str
    priority: int = 1
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    complexity: QueryComplexity
    query_type: QueryType
    requires_rag: bool
    word_count: int
    has_code: bool
    has_math: bool
    language: str
    confidence: float


class QueryAnalyzer:
    """Analyzes queries to determine complexity and type"""
    
    def __init__(self):
        # Keywords for different query types
        self.factual_keywords = [
            "what", "when", "where", "who", "which", "how many", "how much",
            "define", "explain", "describe", "tell me about"
        ]
        
        self.analytical_keywords = [
            "analyze", "compare", "evaluate", "assess", "examine", "review",
            "pros and cons", "advantages", "disadvantages", "difference"
        ]
        
        self.creative_keywords = [
            "create", "write", "generate", "design", "compose", "invent",
            "imagine", "story", "poem", "script", "brainstorm"
        ]
        
        self.technical_keywords = [
            "code", "program", "function", "algorithm", "debug", "implement",
            "api", "database", "server", "framework", "library"
        ]
        
        # Patterns for complexity analysis
        self.code_patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',         # Inline code
            r'\b(def|function|class|import|from|if|for|while|try|catch)\b',
            r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\b'
        ]
        
        self.math_patterns = [
            r'[+\-*/=]\s*\d+',   # Mathematical operations
            r'\b\d+\s*[+\-*/]\s*\d+',
            r'\b(calculate|compute|solve|equation|formula)\b'
        ]
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a query to determine its characteristics"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Determine complexity
        complexity = self._determine_complexity(query, word_count)
        
        # Determine query type
        query_type = self._determine_query_type(query_lower)
        
        # Check for special content
        has_code = self._has_code(query)
        has_math = self._has_math(query)
        
        # Determine if RAG is needed
        requires_rag = self._requires_rag(query_lower, query_type)
        
        # Simple language detection (could be enhanced)
        language = self._detect_language(query)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            complexity, query_type, word_count, has_code, has_math
        )
        
        return QueryAnalysis(
            complexity=complexity,
            query_type=query_type,
            requires_rag=requires_rag,
            word_count=word_count,
            has_code=has_code,
            has_math=has_math,
            language=language,
            confidence=confidence
        )
    
    def _determine_complexity(self, query: str, word_count: int) -> QueryComplexity:
        """Determine query complexity"""
        # Simple heuristics for complexity
        if word_count <= 10:
            return QueryComplexity.SIMPLE
        elif word_count <= 50:
            # Check for complex patterns
            if self._has_code(query) or self._has_math(query):
                return QueryComplexity.COMPLEX
            if any(keyword in query.lower() for keyword in self.analytical_keywords):
                return QueryComplexity.MEDIUM
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.COMPLEX
    
    def _determine_query_type(self, query_lower: str) -> QueryType:
        """Determine the type of query"""
        # Check for technical content first
        if any(keyword in query_lower for keyword in self.technical_keywords):
            return QueryType.TECHNICAL
        
        # Check for creative requests
        if any(keyword in query_lower for keyword in self.creative_keywords):
            return QueryType.CREATIVE
        
        # Check for analytical requests
        if any(keyword in query_lower for keyword in self.analytical_keywords):
            return QueryType.ANALYTICAL
        
        # Check for factual questions
        if any(keyword in query_lower for keyword in self.factual_keywords):
            return QueryType.FACTUAL
        
        # Default to conversational
        return QueryType.CONVERSATIONAL
    
    def _has_code(self, query: str) -> bool:
        """Check if query contains code"""
        for pattern in self.code_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _has_math(self, query: str) -> bool:
        """Check if query contains mathematical content"""
        for pattern in self.math_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _requires_rag(self, query_lower: str, query_type: QueryType) -> bool:
        """Determine if query requires RAG (organization-specific knowledge)"""
        # Keywords that suggest organization-specific information is needed
        org_keywords = [
            "company", "organization", "policy", "procedure", "document",
            "our", "we", "us", "here", "this company", "this organization",
            "employee", "staff", "team", "department", "office"
        ]
        
        # Factual queries about the organization likely need RAG
        if query_type == QueryType.FACTUAL:
            if any(keyword in query_lower for keyword in org_keywords):
                return True
        
        # General knowledge questions might not need RAG
        general_keywords = [
            "general", "world", "global", "universal", "common",
            "typical", "standard", "normal", "average"
        ]
        
        if any(keyword in query_lower for keyword in general_keywords):
            return False
        
        # Default to requiring RAG for organization context
        return True
    
    def _detect_language(self, query: str) -> str:
        """Simple language detection (could be enhanced with proper library)"""
        # Very basic detection - could use langdetect library
        return "en"  # Default to English
    
    def _calculate_confidence(
        self, 
        complexity: QueryComplexity, 
        query_type: QueryType,
        word_count: int,
        has_code: bool,
        has_math: bool
    ) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence for clear indicators
        if has_code or has_math:
            confidence += 0.2
        
        if word_count < 5:  # Very short queries are harder to classify
            confidence -= 0.2
        elif word_count > 100:  # Very long queries might be complex
            confidence -= 0.1
        
        # Clamp between 0.1 and 1.0
        return max(0.1, min(1.0, confidence))


class ModelRouter:
    """Routes queries to appropriate models based on analysis and rules"""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.analyzer = QueryAnalyzer()
        self.routing_rules: List[RoutingRule] = []
        self.default_models = {
            QueryComplexity.SIMPLE: (ModelProvider.OPENAI, "gpt-3.5-turbo"),
            QueryComplexity.MEDIUM: (ModelProvider.OPENAI, "gpt-4"),
            QueryComplexity.COMPLEX: (ModelProvider.OPENAI, "gpt-4")
        }
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a routing rule"""
        self.routing_rules.append(rule)
        # Sort by priority (higher priority first)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added routing rule: {rule}")
    
    def set_default_model(
        self, 
        complexity: QueryComplexity, 
        provider: ModelProvider, 
        model: str
    ):
        """Set default model for a complexity level"""
        self.default_models[complexity] = (provider, model)
        logger.info(f"Set default model for {complexity}: {provider.value}/{model}")
    
    async def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ModelProvider, str, QueryAnalysis]:
        """Route a query to the appropriate model"""
        
        # Analyze the query
        analysis = await self.analyzer.analyze_query(query)
        
        # Find matching routing rule
        selected_provider, selected_model = self._find_matching_rule(analysis, context)
        
        if not selected_provider:
            # Use default model for complexity level
            selected_provider, selected_model = self.default_models[analysis.complexity]
        
        logger.info(
            f"Routed query (complexity: {analysis.complexity.value}, "
            f"type: {analysis.query_type.value}) to {selected_provider.value}/{selected_model}"
        )
        
        return selected_provider, selected_model, analysis
    
    def _find_matching_rule(
        self, 
        analysis: QueryAnalysis, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[ModelProvider], Optional[str]]:
        """Find the first matching routing rule"""
        
        for rule in self.routing_rules:
            if self._rule_matches(rule, analysis, context):
                return rule.provider, rule.model
        
        return None, None
    
    def _rule_matches(
        self, 
        rule: RoutingRule, 
        analysis: QueryAnalysis, 
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if a routing rule matches the query analysis"""
        
        # Check complexity match
        if rule.complexity != analysis.complexity:
            return False
        
        # Check query type match
        if rule.query_type != analysis.query_type:
            return False
        
        # Check additional conditions if specified
        if rule.conditions:
            for condition, expected_value in rule.conditions.items():
                if condition == "requires_rag" and analysis.requires_rag != expected_value:
                    return False
                elif condition == "has_code" and analysis.has_code != expected_value:
                    return False
                elif condition == "has_math" and analysis.has_math != expected_value:
                    return False
                elif condition == "min_confidence" and analysis.confidence < expected_value:
                    return False
                elif condition == "language" and analysis.language != expected_value:
                    return False
        
        return True
    
    async def generate_routed_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Optional[ModelResponse], QueryAnalysis]:
        """Generate response using routed model"""
        
        # Route the query
        provider, model, analysis = await self.route_query(query, context)
        
        # Build messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        
        # Generate response
        response = await self.model_service.generate_response(
            messages=messages,
            provider=provider,
            model=model,
            **kwargs
        )
        
        return response, analysis
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            "total_rules": len(self.routing_rules),
            "default_models": {
                complexity.value: f"{provider.value}/{model}"
                for complexity, (provider, model) in self.default_models.items()
            },
            "routing_rules": [
                {
                    "complexity": rule.complexity.value,
                    "query_type": rule.query_type.value,
                    "model": f"{rule.provider.value}/{rule.model}",
                    "priority": rule.priority,
                    "conditions": rule.conditions
                }
                for rule in self.routing_rules
            ]
        }


def create_default_routing_rules() -> List[RoutingRule]:
    """Create a set of default routing rules"""
    rules = [
        # Simple factual questions - use fast, cheap model
        RoutingRule(
            complexity=QueryComplexity.SIMPLE,
            query_type=QueryType.FACTUAL,
            provider=ModelProvider.OPENAI,
            model="gpt-3.5-turbo",
            priority=10
        ),
        
        # Technical questions with code - use capable model
        RoutingRule(
            complexity=QueryComplexity.MEDIUM,
            query_type=QueryType.TECHNICAL,
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            priority=20,
            conditions={"has_code": True}
        ),
        
        # Complex analytical tasks - use most capable model
        RoutingRule(
            complexity=QueryComplexity.COMPLEX,
            query_type=QueryType.ANALYTICAL,
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            priority=30
        ),
        
        # Creative tasks - use creative model
        RoutingRule(
            complexity=QueryComplexity.MEDIUM,
            query_type=QueryType.CREATIVE,
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            priority=15
        ),
        
        # Math problems - use capable model
        RoutingRule(
            complexity=QueryComplexity.MEDIUM,
            query_type=QueryType.ANALYTICAL,
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            priority=25,
            conditions={"has_math": True}
        )
    ]
    
    return rules


def setup_default_router(model_service: ModelService) -> ModelRouter:
    """Set up a model router with default configuration"""
    router = ModelRouter(model_service)

    # Add default routing rules
    for rule in create_default_routing_rules():
        router.add_routing_rule(rule)
    
    logger.info("Model router set up with default configuration")
    return router
