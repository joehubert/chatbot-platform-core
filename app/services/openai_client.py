"""
OpenAI LLM Client

Implementation of OpenAI LLM client following the BaseLLMClient interface.
Supports both chat completion and streaming responses.
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
import logging
import asyncio

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .model_factory import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI implementation of BaseLLMClient"""
    
    def __init__(self, config: Dict[str, Any]):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        super().__init__(config)
        
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")  # For custom OpenAI endpoints
        self.organization = config.get("organization")
        self.timeout = config.get("timeout", 30)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.organization:
            client_kwargs["organization"] = self.organization
        
        self.client = AsyncOpenAI(**client_kwargs)
        
        # Model-specific settings
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1000)
        self.default_top_p = config.get("top_p", 1.0)
        self.default_frequency_penalty = config.get("frequency_penalty", 0.0)
        self.default_presence_penalty = config.get("presence_penalty", 0.0)
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Optional[LLMResponse]:
        """Generate a response using OpenAI's chat completion"""
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": self.format_messages(messages),
                "temperature": kwargs.get("temperature", self.default_temperature),
                "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
                "top_p": kwargs.get("top_p", self.default_top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.default_frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.default_presence_penalty),
                "timeout": kwargs.get("timeout", self.timeout)
            }
            
            # Add additional parameters if provided
            if "stop" in kwargs:
                request_params["stop"] = kwargs["stop"]
            if "seed" in kwargs:
                request_params["seed"] = kwargs["seed"]
            if "tools" in kwargs:
                request_params["tools"] = kwargs["tools"]
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response content
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                content = choice.message.content or ""
                
                # Extract usage information
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                
                # Extract metadata
                metadata = {
                    "finish_reason": choice.finish_reason,
                    "response_id": response.id,
                    "model": response.model,
                    "created": response.created
                }
                
                # Add tool calls if present
                if choice.message.tool_calls:
                    metadata["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        for tool_call in choice.message.tool_calls
                    ]
                
                return LLMResponse(
                    content=content,
                    model=response.model,
                    provider=self.provider,
                    usage=usage,
                    metadata=metadata
                )
            else:
                logger.error("No choices returned from OpenAI API")
                return None
                
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return None
    
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using OpenAI's chat completion"""
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": self.format_messages(messages),
                "temperature": kwargs.get("temperature", self.default_temperature),
                "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
                "top_p": kwargs.get("top_p", self.default_top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.default_frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.default_presence_penalty),
                "stream": True,
                "timeout": kwargs.get("timeout", self.timeout)
            }
            
            # Add additional parameters if provided
            if "stop" in kwargs:
                request_params["stop"] = kwargs["stop"]
            if "seed" in kwargs:
                request_params["seed"] = kwargs["seed"]
            
            # Make streaming API call
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        yield choice.delta.content
                        
        except Exception as e:
            logger.error(f"Error generating OpenAI streaming response: {e}")
            yield f"Error: {str(e)}"
    
    async def health_check(self) -> bool:
        """Check if OpenAI service is healthy"""
        try:
            # Make a simple API call to test connectivity
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                timeout=10
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for OpenAI API (already in correct format)"""
        # OpenAI expects messages in the format we're already using
        formatted_messages = []
        
        for message in messages:
            # Ensure required fields
            if "role" not in message or "content" not in message:
                logger.warning(f"Skipping malformed message: {message}")
                continue
            
            # Validate role
            if message["role"] not in ["system", "user", "assistant", "tool"]:
                logger.warning(f"Unknown role: {message['role']}, defaulting to 'user'")
                message = {"role": "user", "content": message["content"]}
            
            formatted_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        return formatted_messages
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Very rough estimation: ~4 characters per token for English
        # For more accurate counting, use tiktoken library
        return len(text) // 4
    
    def get_max_context_length(self) -> int:
        """Get maximum context length for the model"""
        # Context lengths for common OpenAI models
        context_lengths = {
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000
        }
        
        return context_lengths.get(self.model, 4096)  # Default to 4k
    
    def truncate_conversation(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Truncate conversation to fit within context limits"""
        if max_tokens is None:
            max_tokens = self.get_max_context_length() - self.default_max_tokens - 500  # Buffer
        
        # Start from the end and work backwards
        truncated_messages = []
        current_tokens = 0
        
        # Always keep system message if present
        system_message = None
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]
            current_tokens += self.estimate_tokens(system_message["content"])
            messages = messages[1:]
        
        # Add messages from most recent backwards
        for message in reversed(messages):
            message_tokens = self.estimate_tokens(message["content"])
            if current_tokens + message_tokens > max_tokens:
                break
            truncated_messages.insert(0, message)
            current_tokens += message_tokens
        
        # Add system message back if it exists
        if system_message:
            truncated_messages.insert(0, system_message)
        
        return truncated_messages
