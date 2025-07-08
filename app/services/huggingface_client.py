"""
HuggingFace Client Service

Implements HuggingFace Inference API integration for the LLM factory.
Supports both Inference API and local model deployment.
"""

import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator

import aiohttp
from transformers import AutoTokenizer

from .llm_factory import BaseLLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseLLMClient):
    """HuggingFace API client implementation."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_inference_api: bool = True,
        **kwargs
    ):
        super().__init__(model_name, api_key, **kwargs)
        
        self.use_inference_api = use_inference_api
        self.base_url = base_url or "https://api-inference.huggingface.co"
        self.inference_url = f"{self.base_url}/models/{self.model_name}"
        
        # Initialize tokenizer for token counting
        self.tokenizer = None
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Initialize tokenizer for the model."""
        try:
            if self.use_inference_api:
                # For inference API, we'll use a fallback tokenizer
                # In production, you might want to download the actual tokenizer
                self.tokenizer = None
                logger.info(f"Using inference API for model: {self.model_name}")
            else:
                # For local models, load the tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info(f"Loaded tokenizer for model: {self.model_name}")
                
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {self.model_name}: {e}")
            self.tokenizer = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate content using HuggingFace's API.

        Args:
            request: The LLM request

        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()
        
        try:
            if self.use_inference_api:
                response_data = await self._call_inference_api(request)
            else:
                response_data = await self._call_local_api(request)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract content
            content = self._extract_content(response_data, request)
            
            # Estimate token usage
            usage = self._estimate_token_usage(request.prompt, content)
            
            # Calculate cost estimate
            cost_estimate = self._calculate_cost(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)
            )
            
            return LLMResponse(
                content=content,
                provider="huggingface",
                model=self.model_name,
                usage=usage,
                metadata={
                    "model": self.model_name,
                    "inference_api": self.use_inference_api,
                    "raw_response": response_data if isinstance(response_data, dict) else {}
                },
                cost_estimate=cost_estimate,
                latency_ms=latency_ms
            )

        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                logger.warning(f"HuggingFace rate limit exceeded: {e}")
                raise
            else:
                logger.error(f"HuggingFace API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in HuggingFace generate: {e}")
            raise

    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """
        Stream generate content using HuggingFace's API.
        
        Note: HuggingFace Inference API doesn't always support streaming,
        so this implementation may fall back to non-streaming behavior.

        Args:
            request: The LLM request

        Yields:
            Content chunks as they are generated
        """
        try:
            if self.use_inference_api:
                # For inference API, we might not have true streaming
                # Fall back to generating complete response and yielding in chunks
                response = await self.generate(request)
                content = response.content
                
                # Yield content in chunks to simulate streaming
                chunk_size = 10  # characters per chunk
                for i in range(0, len(content), chunk_size):
                    yield content[i:i + chunk_size]
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.1)
            else:
                # For local models, implement actual streaming if supported
                await self._stream_local_api(request)

        except Exception as e:
            logger.error(f"Unexpected error in HuggingFace stream_generate: {e}")
            raise

    async def _call_inference_api(self, request: LLMRequest) -> Dict[str, Any]:
        """Call HuggingFace Inference API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload based on model type
        payload = self._prepare_inference_payload(request)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.inference_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _call_local_api(self, request: LLMRequest) -> Dict[str, Any]:
        """Call local HuggingFace model deployment."""
        # This would be implemented for local model deployments
        # For now, raise an error as it's not implemented
        raise NotImplementedError("Local HuggingFace model deployment not yet implemented")

    async def _stream_local_api(self, request: LLMRequest):
        """Stream from local HuggingFace model deployment."""
        # This would be implemented for local model deployments with streaming
        raise NotImplementedError("Streaming from local HuggingFace models not yet implemented")

    def _prepare_inference_payload(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare payload for HuggingFace Inference API."""
        # Base payload
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "return_full_text": False
            }
        }
        
        # Add system prompt if provided (some models support it)
        if request.system_prompt:
            # For chat models, combine system prompt with user prompt
            combined_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
            payload["inputs"] = combined_prompt
        
        # Handle chat messages format
        if request.messages:
            payload["inputs"] = self._format_messages_for_inference(request.messages)
        
        # Add model-specific parameters
        for key, value in request.model_params.items():
            if key in [
                "top_p", "top_k", "repetition_penalty", "num_return_sequences",
                "do_sample", "early_stopping", "use_cache"
            ]:
                payload["parameters"][key] = value
        
        return payload

    def _format_messages_for_inference(self, messages: list) -> str:
        """Format chat messages for inference API."""
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts)

    def _extract_content(self, response_data: Any, request: LLMRequest) -> str:
        """Extract content from HuggingFace API response."""
        if isinstance(response_data, list) and len(response_data) > 0:
            # Standard inference API response format
            result = response_data[0]
            if isinstance(result, dict):
                return result.get("generated_text", "")
            elif isinstance(result, str):
                return result
        elif isinstance(response_data, dict):
            # Alternative response format
            return response_data.get("generated_text", "")
        
        logger.warning(f"Unexpected response format from HuggingFace: {type(response_data)}")
        return str(response_data) if response_data else ""

    def _estimate_token_usage(self, prompt: str, completion: str) -> Dict[str, int]:
        """Estimate token usage for prompt and completion."""
        if self.tokenizer:
            # Use actual tokenizer if available
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = len(self.tokenizer.encode(completion))
        else:
            # Fallback estimation (roughly 4 characters per token)
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(completion) // 4)
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for HuggingFace API usage."""
        # HuggingFace Inference API pricing varies by model
        # This is a rough estimation - actual pricing should be configured
        
        if self.use_inference_api:
            # Inference API is generally free for limited usage
            # or has different pricing tiers
            return 0.0
        else:
            # For local deployment, cost would be compute-based
            # Placeholder calculation
            return (input_tokens + output_tokens) * 0.00001

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the HuggingFace model."""
        capabilities = ["text_generation"]
        
        if self._is_chat_model():
            capabilities.append("chat")
        
        if self._supports_streaming():
            capabilities.append("streaming")
        
        # Filter out None values
        capabilities = [cap for cap in capabilities if cap is not None]
        
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "capabilities": capabilities,
            "max_tokens": self._get_model_max_tokens(),
            "supports_system_prompt": self._supports_system_prompt(),
            "supports_functions": False,  # HuggingFace models don't support function calling
            "supports_tools": False,
            "inference_api": self.use_inference_api,
            "context_window": self._get_model_context_window()
        }

    def _is_chat_model(self) -> bool:
        """Check if this is a chat/instruction-tuned model."""
        chat_indicators = [
            "chat", "instruct", "assistant", "conversational",
            "dialogue", "alpaca", "vicuna", "llama-2-chat"
        ]
        model_name_lower = self.model_name.lower()
        return any(indicator in model_name_lower for indicator in chat_indicators)

    def _supports_streaming(self) -> bool:
        """Check if the model supports streaming."""
        # Most HuggingFace models can support streaming in theory,
        # but Inference API doesn't always support it
        return not self.use_inference_api

    def _supports_system_prompt(self) -> bool:
        """Check if the model supports system prompts."""
        # Chat models typically support system prompts
        return self._is_chat_model()

    def _get_model_max_tokens(self) -> int:
        """Get maximum tokens for the model."""
        # Common HuggingFace model token limits
        max_tokens_map = {
            "gpt2": 1024,
            "distilgpt2": 1024,
            "microsoft/DialoGPT": 1024,
            "facebook/blenderbot": 128,
            "google/flan-t5": 512,
            "bigscience/bloom": 2048,
            "meta-llama/Llama-2": 4096,
            "mistralai/Mistral": 8192,
            "codellama": 16384
        }
        
        model_lower = self.model_name.lower()
        for key, max_tokens in max_tokens_map.items():
            if key.lower() in model_lower:
                return max_tokens
        
        # Default for unknown models
        return 2048

    def _get_model_context_window(self) -> int:
        """Get context window size for the model."""
        # This is often the same as max tokens for most models
        context_window_map = {
            "gpt2": 1024,
            "distilgpt2": 1024,
            "microsoft/DialoGPT": 1024,
            "facebook/blenderbot": 128,
            "google/flan-t5": 512,
            "bigscience/bloom": 2048,
            "meta-llama/Llama-2": 4096,
            "mistralai/Mistral": 8192,
            "codellama": 16384
        }
        
        model_lower = self.model_name.lower()
        for key, context_size in context_window_map.items():
            if key.lower() in model_lower:
                return context_size
        
        return 2048  # Default

    async def health_check(self) -> bool:
        """Check if the HuggingFace API/model is accessible."""
        try:
            if self.use_inference_api:
                # Check inference API health
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Make a minimal request
                payload = {
                    "inputs": "Hi",
                    "parameters": {
                        "max_new_tokens": 1,
                        "temperature": 0
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.inference_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        # Even if we get a rate limit, the API is working
                        if response.status in [200, 429]:
                            logger.debug(f"Health check passed for HuggingFace model: {self.model_name}")
                            return True
                        else:
                            return False
            else:
                # For local models, implement appropriate health check
                # This is a placeholder
                return True
                
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                logger.warning(f"HuggingFace rate limited during health check: {self.model_name}")
                return True  # Rate limit means the API is working
            else:
                logger.error(f"HuggingFace API error during health check: {e}")
                return False
        except Exception as e:
            logger.error(f"HuggingFace health check failed for {self.model_name}: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the client."""
        return f"HuggingFaceClient(model={self.model_name}, inference_api={self.use_inference_api})"

    def __repr__(self) -> str:
        """Detailed string representation of the client."""
        return (
            f"HuggingFaceClient("
            f"model_name='{self.model_name}', "
            f"use_inference_api={self.use_inference_api}, "
            f"base_url='{self.base_url}'"
            f")"
        )
