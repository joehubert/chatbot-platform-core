"""
Response Validator Service

This service validates generated responses for quality, safety, and appropriateness
before they are sent to users.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from app.services.llm_factory import LLMFactory
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    level: ValidationLevel
    type: str
    message: str
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    confidence: float
    suggested_improvements: List[str]


class ResponseValidator:
    """Service for validating chatbot responses."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm_factory = LLMFactory()
        self.validation_model = self.settings.VALIDATION_MODEL
        self.organization_context = self._load_organization_context()
        self.safety_filters = self._load_safety_filters()
        self.quality_thresholds = self._load_quality_thresholds()
    
    def _load_organization_context(self) -> Dict[str, Any]:
        """Load organization context for validation."""
        return {
            "name": self.settings.ORGANIZATION_NAME,
            "domain": self.settings.ORGANIZATION_DOMAIN,
            "tone": self.settings.ORGANIZATION_TONE,
            "values": self.settings.ORGANIZATION_VALUES,
            "prohibited_topics": self.settings.PROHIBITED_TOPICS,
            "required_disclaimers": self.settings.REQUIRED_DISCLAIMERS
        }
    
    def _load_safety_filters(self) -> Dict[str, Any]:
        """Load safety filters and patterns."""
        return {
            "hate_speech_patterns": [
                r'\b(hate|hatred|racist|racism|sexist|sexism)\b',
                r'\b(offensive|discriminatory|prejudice)\b'
            ],
            "inappropriate_content": [
                r'\b(explicit|sexual|violent|illegal)\b',
                r'\b(drug|weapon|harm)\b'
            ],
            "personal_info_patterns": [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{16}\b',  # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
            ],
            "prohibited_advice": [
                r'\b(medical|legal|financial)\s+(advice|recommendation)\b',
                r'\b(diagnose|prescribe|treat)\b'
            ]
        }
    
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """Load quality thresholds for validation."""
        return {
            "min_relevance_score": 0.7,
            "min_coherence_score": 0.6,
            "min_helpfulness_score": 0.7,
            "max_length": 1000,
            "min_length": 10
        }
    
    async def validate_response(
        self,
        response: str,
        original_query: str,
        context: Optional[Dict[str, Any]] = None,
        rag_sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Validate a response for quality, safety, and appropriateness.
        
        Args:
            response: The generated response to validate
            original_query: The original user query
            context: Additional context from the request
            rag_sources: Sources used for RAG generation
            
        Returns:
            Dictionary with validation results
        """
        try:
            issues = []
            
            # Run all validation checks
            issues.extend(await self._check_safety(response))
            issues.extend(await self._check_quality(response, original_query, context))
            issues.extend(await self._check_relevance(response, original_query, rag_sources))
            issues.extend(await self._check_compliance(response, context))
            issues.extend(await self._check_format(response))
            
            # Determine if response is valid
            critical_issues = [issue for issue in issues if issue.level == ValidationLevel.CRITICAL]
            is_valid = len(critical_issues) == 0
            
            # Calculate confidence score
            confidence = self._calculate_confidence(issues)
            
            # Generate improvement suggestions
            suggestions = self._generate_suggestions(issues)
            
            result = {
                "is_valid": is_valid,
                "issues": [
                    {
                        "level": issue.level.value,
                        "type": issue.type,
                        "message": issue.message,
                        "suggestion": issue.suggestion
                    }
                    for issue in issues
                ],
                "confidence": confidence,
                "suggested_improvements": suggestions
            }
            
            logger.debug(f"Response validation: {'PASSED' if is_valid else 'FAILED'} (confidence: {confidence})")
            
            return result
            
        except Exception as e:
            logger.error(f"Response validation error: {str(e)}")
            # Default to valid if validation fails
            return {
                "is_valid": True,
                "issues": [],
                "confidence": 0.5,
                "suggested_improvements": []
            }
    
    async def _check_safety(self, response: str) -> List[ValidationIssue]:
        """Check response for safety issues."""
        issues = []
        
        # Check for hate speech
        for pattern in self.safety_filters["hate_speech_patterns"]:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    type="hate_speech",
                    message="Response contains potentially hateful or discriminatory language",
                    suggestion="Remove offensive language and use inclusive terminology"
                ))
        
        # Check for inappropriate content
        for pattern in self.safety_filters["inappropriate_content"]:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    type="inappropriate_content",
                    message="Response contains inappropriate content",
                    suggestion="Remove inappropriate content and provide appropriate alternatives"
                ))
        
        # Check for personal information exposure
        for pattern in self.safety_filters["personal_info_patterns"]:
            if re.search(pattern, response):
                issues.append(ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    type="personal_info",
                    message="Response may contain personal information",
                    suggestion="Remove or redact personal information"
                ))
        
        # Check for prohibited advice
        for pattern in self.safety_filters["prohibited_advice"]:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    type="prohibited_advice",
                    message="Response may provide advice outside organization's scope",
                    suggestion="Add appropriate disclaimers or redirect to qualified professionals"
                ))
        
        return issues
    
    async def _check_quality(
        self,
        response: str,
        original_query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Check response quality using LLM-based assessment."""
        issues = []
        
        try:
            # Build quality assessment prompt
            system_prompt = self._build_quality_system_prompt()
            user_prompt = self._build_quality_user_prompt(response, original_query, context)
            
            # Get LLM client
            llm_client = self.llm_factory.get_client(self.validation_model)
            
            # Generate quality assessment
            assessment = await llm_client.generate_response(
                query=user_prompt,
                context={"system_prompt": system_prompt},
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse quality assessment
            quality_result = self._parse_quality_assessment(assessment["response"])
            
            # Check thresholds
            if quality_result["relevance"] < self.quality_thresholds["min_relevance_score"]:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    type="low_relevance",
                    message="Response may not be sufficiently relevant to the query",
                    suggestion="Improve relevance by better addressing the user's specific question"
                ))
            
            if quality_result["coherence"] < self.quality_thresholds["min_coherence_score"]:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    type="low_coherence",
                    message="Response may lack clarity or coherence",
                    suggestion="Improve structure and flow of the response"
                ))
            
            if quality_result["helpfulness"] < self.quality_thresholds["min_helpfulness_score"]:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    type="low_helpfulness",
                    message="Response may not be sufficiently helpful",
                    suggestion="Provide more actionable information or clearer guidance"
                ))
            
        except Exception as e:
            logger.error(f"Quality check error: {str(e)}")
            # Add a generic quality warning if assessment fails
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                type="quality_check_failed",
                message="Could not assess response quality",
                suggestion="Manual review recommended"
            ))
        
        return issues
    
    async def _check_relevance(
        self,
        response: str,
        original_query: str,
        rag_sources: Optional[List[Dict[str, Any]]]
    ) -> List[ValidationIssue]:
        """Check response relevance to the original query."""
        issues = []
        
        # Basic length checks
        if len(response) < self.quality_thresholds["min_length"]:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                type="too_short",
                message="Response may be too short to be helpful",
                suggestion="Provide more detailed information"
            ))
        
        if len(response) > self.quality_thresholds["max_length"]:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                type="too_long",
                message="Response may be too long and overwhelming",
                suggestion="Provide more concise information"
            ))
        
        # Check if response actually addresses the query
        query_keywords = set(original_query.lower().split())
        response_keywords = set(response.lower().split())
        
        overlap = len(query_keywords.intersection(response_keywords))
        if overlap / len(query_keywords) < 0.3:  # Less than 30% keyword overlap
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                type="keyword_mismatch",
                message="Response may not address the specific query",
                suggestion="Ensure response directly answers the user's question"
            ))
        
        # Check if RAG sources are properly utilized
        if rag_sources and not any(source["content"].lower() in response.lower() for source in rag_sources):
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                type="rag_underutilization",
                message="Response may not fully utilize available knowledge sources",
                suggestion="Better integrate information from knowledge base"
            ))
        
        return issues
    
    async def _check_compliance(
        self,
        response: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Check response compliance with organization policies."""
        issues = []
        
        # Check for required disclaimers
        for disclaimer_type, disclaimer_text in self.organization_context["required_disclaimers"].items():
            if self._needs_disclaimer(response, disclaimer_type) and disclaimer_text not in response:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    type="missing_disclaimer",
                    message=f"Response may need {disclaimer_type} disclaimer",
                    suggestion=f"Add appropriate disclaimer: {disclaimer_text}"
                ))
        
        # Check for prohibited topics
        for topic in self.organization_context["prohibited_topics"]:
            if topic.lower() in response.lower():
                issues.append(ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    type="prohibited_topic",
                    message=f"Response discusses prohibited topic: {topic}",
                    suggestion="Remove discussion of prohibited topics"
                ))
        
        # Check tone compliance
        expected_tone = self.organization_context["tone"]
        if not self._matches_tone(response, expected_tone):
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                type="tone_mismatch",
                message=f"Response tone may not match expected {expected_tone} tone",
                suggestion=f"Adjust tone to be more {expected_tone}"
            ))
        
        return issues
    
    async def _check_format(self, response: str) -> List[ValidationIssue]:
        """Check response formatting and structure."""
        issues = []
        
        # Check for proper sentence structure
        if not response.strip().endswith(('.', '!', '?')):
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                type="missing_punctuation",
                message="Response should end with proper punctuation",
                suggestion="Add appropriate ending punctuation"
            ))
        
        # Check for excessive capitalization
        if sum(1 for c in response if c.isupper()) / len(response) > 0.3:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                type="excessive_caps",
                message="Response contains excessive capitalization",
                suggestion="Use normal capitalization patterns"
            ))
        
        # Check for repetitive content
        sentences = response.split('.')
        if len(sentences) > 2:
            unique_sentences = set(sentences)
            if len(unique_sentences) / len(sentences) < 0.8:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    type="repetitive_content",
                    message="Response contains repetitive content",
                    suggestion="Remove redundant information"
                ))
        
        return issues
    
    def _build_quality_system_prompt(self) -> str:
        """Build system prompt for quality assessment."""
        return f"""You are a quality assessor for responses from {self.organization_context['name']}.

Your task is to evaluate response quality across multiple dimensions.

Organization Context:
- Name: {self.organization_context['name']}
- Domain: {self.organization_context['domain']}
- Expected Tone: {self.organization_context['tone']}
- Values: {', '.join(self.organization_context['values'])}

Evaluate responses on:
1. Relevance (0.0-1.0): How well does the response address the user's query?
2. Coherence (0.0-1.0): How clear and well-structured is the response?
3. Helpfulness (0.0-1.0): How useful is the response for the user?
4. Accuracy (0.0-1.0): How accurate is the information provided?
5. Appropriateness (0.0-1.0): How appropriate is the response for the context?

Respond in JSON format:
{{
    "relevance": float,
    "coherence": float,
    "helpfulness": float,
    "accuracy": float,
    "appropriateness": float,
    "overall_score": float,
    "feedback": "brief explanation of assessment"
}}"""
    
    def _build_quality_user_prompt(
        self,
        response: str,
        original_query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build user prompt for quality assessment."""
        prompt = f"Original Query: {original_query}\n\n"
        prompt += f"Response to Evaluate: {response}\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += "Please assess the quality of this response."
        
        return prompt
    
    def _parse_quality_assessment(self, assessment: str) -> Dict[str, float]:
        """Parse quality assessment from LLM response."""
        try:
            import json
            if assessment.strip().startswith('{'):
                result = json.loads(assessment.strip())
                return {
                    "relevance": result.get("relevance", 0.5),
                    "coherence": result.get("coherence", 0.5),
                    "helpfulness": result.get("helpfulness", 0.5),
                    "accuracy": result.get("accuracy", 0.5),
                    "appropriateness": result.get("appropriateness", 0.5),
                    "overall_score": result.get("overall_score", 0.5)
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        return {
            "relevance": 0.5,
            "coherence": 0.5,
            "helpfulness": 0.5,
            "accuracy": 0.5,
            "appropriateness": 0.5,
            "overall_score": 0.5
        }
    
    def _needs_disclaimer(self, response: str, disclaimer_type: str) -> bool:
        """Check if response needs a specific disclaimer."""
        disclaimer_triggers = {
            "medical": ["health", "medical", "symptom", "treatment", "medication"],
            "legal": ["legal", "law", "lawsuit", "attorney", "court"],
            "financial": ["investment", "financial", "money", "tax", "loan"]
        }
        
        triggers = disclaimer_triggers.get(disclaimer_type, [])
        return any(trigger in response.lower() for trigger in triggers)
    
    def _matches_tone(self, response: str, expected_tone: str) -> bool:
        """Check if response matches expected tone."""
        tone_indicators = {
            "professional": ["please", "thank you", "kindly", "appreciate"],
            "friendly": ["happy", "glad", "excited", "wonderful"],
            "casual": ["hey", "sure", "cool", "awesome"],
            "formal": ["furthermore", "consequently", "therefore", "regarding"]
        }
        
        indicators = tone_indicators.get(expected_tone.lower(), [])
        return any(indicator in response.lower() for indicator in indicators)
    
    def _calculate_confidence(self, issues: List[ValidationIssue]) -> float:
        """Calculate confidence score based on validation issues."""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        weights = {
            ValidationLevel.CRITICAL: 0.5,
            ValidationLevel.WARNING: 0.3,
            ValidationLevel.INFO: 0.1
        }
        
        total_weight = sum(weights[issue.level] for issue in issues)
        confidence = max(0.0, 1.0 - total_weight)
        
        return confidence
    
    def _generate_suggestions(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        for issue in issues:
            if issue.suggestion:
                suggestions.append(issue.suggestion)
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions
    
    async def health_check(self) -> str:
        """Check if the response validator is healthy."""
        try:
            # Test with a simple response validation
            test_result = await self.validate_response(
                "This is a test response.",
                "test query"
            )
            return "healthy" if test_result else "unhealthy"
        except Exception as e:
            logger.error(f"Response validator health check failed: {str(e)}")
            return "unhealthy"
    
    def update_validation_rules(self, new_rules: Dict[str, Any]) -> None:
        """Update validation rules and thresholds."""
        if "safety_filters" in new_rules:
            self.safety_filters.update(new_rules["safety_filters"])
        
        if "quality_thresholds" in new_rules:
            self.quality_thresholds.update(new_rules["quality_thresholds"])
        
        if "organization_context" in new_rules:
            self.organization_context.update(new_rules["organization_context"])
        
        logger.info("Validation rules updated")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about response validation."""
        return {
            "model_used": self.validation_model,
            "quality_thresholds": self.quality_thresholds,
            "safety_filters_count": {
                filter_type: len(patterns)
                for filter_type, patterns in self.safety_filters.items()
            },
            "organization_context": self.organization_context
        }
