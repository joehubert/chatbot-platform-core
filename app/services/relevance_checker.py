"""
Relevance Checker Service

This service determines if user queries are relevant to the organization's domain
and generates appropriate clarification requests when needed.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.services.llm_factory import LLMFactory
from app.core.config import get_settings

logger = logging.getLogger(__name__)

@dataclass
class RelevanceResult:
    """Result of relevance checking."""
    is_relevant: bool
    confidence: float
    reasoning: str
    suggested_clarification: Optional[str] = None


class RelevanceChecker:
    """Service for checking query relevance and generating clarifications."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm_factory = LLMFactory()
        self.relevance_model = self.settings.RELEVANCE_MODEL
        self.confidence_threshold = self.settings.RELEVANCE_CONFIDENCE_THRESHOLD
        self.organization_context = self._load_organization_context()
    
    def _load_organization_context(self) -> Dict[str, Any]:
        """Load organization context for relevance checking."""
        return {
            "name": self.settings.ORGANIZATION_NAME,
            "domain": self.settings.ORGANIZATION_DOMAIN,
            "services": self.settings.ORGANIZATION_SERVICES,
            "scope": self.settings.ORGANIZATION_SCOPE,
            "out_of_scope_topics": self.settings.OUT_OF_SCOPE_TOPICS,
            "clarification_prompts": self.settings.CLARIFICATION_PROMPTS
        }
    
    async def check_relevance(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Check if a query is relevant to the organization's domain.
        
        Args:
            query: The user's query
            context: Additional context from the request
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with relevance results
        """
        try:
            # Build the relevance checking prompt
            system_prompt = self._build_relevance_system_prompt()
            user_prompt = self._build_relevance_user_prompt(
                query, context, conversation_history
            )
            
            # Get LLM client
            llm_client = self.llm_factory.get_client(self.relevance_model)
            
            # Generate relevance assessment
            response = await llm_client.generate_response(
                query=user_prompt,
                context={"system_prompt": system_prompt},
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse the response
            result = self._parse_relevance_response(response["response"])
            
            logger.debug(f"Relevance check: {result['is_relevant']} (confidence: {result['confidence']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Relevance checking error: {str(e)}")
            # Default to relevant if service fails
            return {
                "is_relevant": True,
                "confidence": 0.5,
                "reasoning": "Relevance checking service unavailable"
            }
    
    async def generate_clarification(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        attempt_number: int = 1
    ) -> str:
        """
        Generate a clarification request for ambiguous queries.
        
        Args:
            query: The user's query
            context: Additional context from the request
            attempt_number: Which clarification attempt this is
            
        Returns:
            Clarification message
        """
        try:
            # Build the clarification prompt
            system_prompt = self._build_clarification_system_prompt()
            user_prompt = self._build_clarification_user_prompt(
                query, context, attempt_number
            )
            
            # Get LLM client
            llm_client = self.llm_factory.get_client(self.relevance_model)
            
            # Generate clarification
            response = await llm_client.generate_response(
                query=user_prompt,
                context={"system_prompt": system_prompt},
                max_tokens=150,
                temperature=0.3
            )
            
            return response["response"]
            
        except Exception as e:
            logger.error(f"Clarification generation error: {str(e)}")
            return self._get_default_clarification(attempt_number)
    
    async def get_out_of_scope_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an out-of-scope response with helpful redirection.
        
        Args:
            query: The user's query
            context: Additional context from the request
            
        Returns:
            Out-of-scope response message
        """
        try:
            # Build the out-of-scope response prompt
            system_prompt = self._build_out_of_scope_system_prompt()
            user_prompt = self._build_out_of_scope_user_prompt(query, context)
            
            # Get LLM client
            llm_client = self.llm_factory.get_client(self.relevance_model)
            
            # Generate out-of-scope response
            response = await llm_client.generate_response(
                query=user_prompt,
                context={"system_prompt": system_prompt},
                max_tokens=200,
                temperature=0.2
            )
            
            return response["response"]
            
        except Exception as e:
            logger.error(f"Out-of-scope response generation error: {str(e)}")
            return self._get_default_out_of_scope_response()
    
    def _build_relevance_system_prompt(self) -> str:
        """Build system prompt for relevance checking."""
        return f"""You are a relevance checker for {self.organization_context['name']}.

Your task is to determine if user queries are relevant to our organization's domain and services.

Organization Context:
- Name: {self.organization_context['name']}
- Domain: {self.organization_context['domain']}
- Services: {', '.join(self.organization_context['services'])}
- Scope: {self.organization_context['scope']}

Out of Scope Topics: {', '.join(self.organization_context['out_of_scope_topics'])}

Respond in JSON format with:
{{
    "is_relevant": boolean,
    "confidence": float (0.0 to 1.0),
    "reasoning": "explanation of decision"
}}

A query is relevant if it:
1. Relates to our services or domain
2. Is a reasonable question someone might ask our organization
3. Could be answered with our knowledge base or resources

A query is NOT relevant if it:
1. Is about topics explicitly out of scope
2. Is completely unrelated to our domain
3. Is inappropriate or off-topic

Use confidence scores:
- 0.9-1.0: Very clearly relevant/irrelevant
- 0.7-0.9: Probably relevant/irrelevant
- 0.5-0.7: Somewhat relevant/irrelevant
- 0.3-0.5: Unclear, needs clarification
- 0.0-0.3: Very unclear or ambiguous"""
    
    def _build_relevance_user_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Build user prompt for relevance checking."""
        prompt = f"Query: {query}\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        if conversation_history:
            prompt += "Conversation History:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "\n"
        
        prompt += "Please assess the relevance of this query to our organization."
        
        return prompt
    
    def _build_clarification_system_prompt(self) -> str:
        """Build system prompt for clarification generation."""
        return f"""You are a helpful assistant for {self.organization_context['name']}.

Your task is to generate clarification requests for ambiguous or unclear queries.

Organization Context:
- Name: {self.organization_context['name']}
- Domain: {self.organization_context['domain']}
- Services: {', '.join(self.organization_context['services'])}

Guidelines for clarification:
1. Be polite and helpful
2. Acknowledge what you understood from their query
3. Ask specific questions to clarify their intent
4. Suggest relevant services or topics they might be interested in
5. Keep responses concise and focused

Example clarification approaches:
- "I understand you're asking about [topic]. Are you specifically interested in [option A] or [option B]?"
- "To better help you with [topic], could you clarify whether you need information about [specific aspect]?"
- "I'd be happy to help! Are you looking for information about our [service 1], [service 2], or something else?"

Generate a helpful clarification request that guides the user toward relevant topics."""
    
    def _build_clarification_user_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        attempt_number: int
    ) -> str:
        """Build user prompt for clarification generation."""
        prompt = f"User Query: {query}\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        if attempt_number > 1:
            prompt += f"This is clarification attempt #{attempt_number}. "
            prompt += "The user may be getting frustrated, so be extra helpful and direct.\n\n"
        
        prompt += "Generate a clarification request to help understand what the user needs."
        
        return prompt
    
    def _build_out_of_scope_system_prompt(self) -> str:
        """Build system prompt for out-of-scope responses."""
        return f"""You are a helpful assistant for {self.organization_context['name']}.

Your task is to politely explain that a query is outside your scope and provide helpful alternatives.

Organization Context:
- Name: {self.organization_context['name']}
- Domain: {self.organization_context['domain']}
- Services: {', '.join(self.organization_context['services'])}

Guidelines for out-of-scope responses:
1. Politely acknowledge their question
2. Explain that it's outside your organization's scope
3. Suggest relevant alternatives or resources
4. Offer to help with related topics you CAN assist with
5. Maintain a helpful and professional tone

Example structure:
"I understand you're asking about [topic]. While that's outside our area of expertise at {self.organization_context['name']}, I'd be happy to help you with [relevant alternative] or you might want to try [suggestion]."

Generate a helpful out-of-scope response."""
    
    def _build_out_of_scope_user_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build user prompt for out-of-scope response generation."""
        prompt = f"User Query: {query}\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += "Generate a polite out-of-scope response with helpful alternatives."
        
        return prompt
    
    def _parse_relevance_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response for relevance checking."""
        try:
            import json
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                result = json.loads(response.strip())
                return {
                    "is_relevant": result.get("is_relevant", False),
                    "confidence": max(0.0, min(1.0, result.get("confidence", 0.5))),
                    "reasoning": result.get("reasoning", "")
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing for non-JSON responses
        response_lower = response.lower()
        
        # Look for relevance indicators
        relevant_indicators = ["relevant", "yes", "true", "related", "appropriate"]
        irrelevant_indicators = ["irrelevant", "no", "false", "unrelated", "inappropriate"]
        
        is_relevant = any(indicator in response_lower for indicator in relevant_indicators)
        
        # Estimate confidence based on language strength
        confidence = 0.5
        if any(word in response_lower for word in ["very", "clearly", "definitely", "obviously"]):
            confidence = 0.9
        elif any(word in response_lower for word in ["probably", "likely", "seems"]):
            confidence = 0.7
        elif any(word in response_lower for word in ["maybe", "unclear", "ambiguous"]):
            confidence = 0.4
        
        return {
            "is_relevant": is_relevant,
            "confidence": confidence,
            "reasoning": response
        }
    
    def _get_default_clarification(self, attempt_number: int) -> str:
        """Get default clarification message when generation fails."""
        if attempt_number == 1:
            return f"I'd like to help you, but I need a bit more information. Could you please clarify what specific aspect of {self.organization_context['domain']} you're interested in?"
        elif attempt_number == 2:
            return f"I'm still not quite sure how to help. Could you please be more specific about what information you need regarding {self.organization_context['name']}?"
        else:
            return f"I'm having trouble understanding your request. You might want to contact our support team directly for personalized assistance."
    
    def _get_default_out_of_scope_response(self) -> str:
        """Get default out-of-scope response when generation fails."""
        return f"I understand you're looking for information, but that topic is outside the scope of what I can help with at {self.organization_context['name']}. Is there anything else related to {self.organization_context['domain']} that I can assist you with?"
    
    async def health_check(self) -> str:
        """Check if the relevance checker is healthy."""
        try:
            # Test with a simple query
            test_result = await self.check_relevance("test query")
            return "healthy" if test_result else "unhealthy"
        except Exception as e:
            logger.error(f"Relevance checker health check failed: {str(e)}")
            return "unhealthy"
    
    def update_organization_context(self, new_context: Dict[str, Any]) -> None:
        """Update organization context for relevance checking."""
        self.organization_context.update(new_context)
        logger.info("Organization context updated for relevance checker")
    
    def get_relevance_stats(self) -> Dict[str, Any]:
        """Get statistics about relevance checking."""
        # In a real implementation, this would track metrics
        return {
            "model_used": self.relevance_model,
            "confidence_threshold": self.confidence_threshold,
            "organization_context": self.organization_context
        }
