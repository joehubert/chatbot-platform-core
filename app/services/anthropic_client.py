"""
Anthropic Claude LLM Client

Implementation of Anthropic Claude LLM client following the BaseLLMClient interface.
Supports both chat completion and streaming responses.
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
import logging

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .model_factory import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude implementation of BaseLLMClient"""
    
    def __init__(self, config: Dict[str, Any]):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        
        super().__init__(config)
        
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 30)
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Initialize Anthropic client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = anthropic.AsyncAnthropic(**client_kwargs)
        
        # Model-specific settings
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1000)
        self.default_top_p = config.get("top_p", 1.0)
        self.default_top_k = config.get("top_k", 250)
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Optional[LLMResponse]:
        """Generate a response using Anthropic's messages API"""
        try:
            # Format messages for Anthropic (separate system message)
            formatted_messages, system_prompt = self._format_messages_for_anthropic(messages)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
                "temperature": kwargs.get("temperature", self.default_temperature),
                "timeout": kwargs.get("timeout", self.timeout)
            }
            
            # Add system prompt if present
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Add optional parameters
            if "top_p" in kwargs:
                request_params["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                request_params["top_k"] = kwargs["top_k"]
            if "stop_sequences" in kwargs:
                request_params["stop_sequences"] = kwargs["stop_sequences"]
            
            # Make API call
            response = await self.client.messages.create(**request_params)
            
            # Extract response content
            if response.content and len(response.content) > 0:
                # Anthropic returns content as a list of content blocks
                content = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        content += content_block.text
                
                # Extract usage information
                usage = {}
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    }
                
                # Extract metadata
                metadata = {
                    "stop_reason": response.stop_reason,
                    "response_id": response.id,
                    "model": response.model,
                    "role": response.role
                }
                
                return LLMResponse(
                    content=content,
                    model=response.model,
                    provider=self.provider,
                    usage=usage,
                    metadata=metadata
                )
            else:
                logger.error("No content returned from Anthropic API")
                return None
                
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            return None
    
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Anthropic's messages API"""
        try:
            # Format messages for Anthropic
            formatted_messages, system_prompt = self._format_messages_for_anthropic(messages)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
                "temperature": kwargs.get("temperature", self.default_temperature),
                "stream": True,
                "timeout": kwargs.get("timeout", self.timeout)
            }
            
            # Add system prompt if present
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Add optional parameters
            if "top_p" in kwargs:
                request_params["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                request_params["top_k"] = kwargs["top_k"]
            if "stop_sequences" in kwargs:
                request_params["stop_sequences"] = kwargs["stop_sequences"]
            
            # Make streaming API call
            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text
                        
        except Exception as e:
            logger.error(f"Error generating Anthropic streaming response: {e}")
            yield f"Error: {str(e)}"
    
    async def health_check(self) -> bool:
        """Check if Anthropic service is healthy"""
        try:
            # Make a simple API call to test connectivity
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                timeout=10
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False
    
    def _format_messages_for_anthropic(
        self, 
        messages: List[Dict[str, str]]
    ) -> tuple[List[Dict[str, str]], Optional[str]]:
        """Format messages for Anthropic API"""
        formatted_messages = []
        system_prompt = None
        
        for message in messages:
            # Handle system messages separately
            if message["role"] == "system":
                # Anthropic uses a separate system parameter
                if system_prompt is None:
                    system_prompt = message["content"]
                else:
                    # Concatenate multiple system messages
                    system_prompt += "\n\n" + message["content"]
            elif message["role"] in ["user", "assistant"]:
                formatted_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
            else:
                # Convert unknown roles to user
                logger.warning(f"Unknown role: {message['role']}, converting to user")
                formatted_messages.append({
                    "role": "user",
                    "content": message["content"]
                })
        
        # Ensure alternating user/assistant pattern
        formatted_messages = self._ensure_alternating_pattern(formatted_messages)
        
        return formatted_messages, system_prompt
    
    def _ensure_alternating_pattern(
        self, 
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Ensure messages alternate between user and assistant"""
        if not messages:
            return messages
        
        fixed_messages = []
        last_role = None
        
        for message in messages:
            current_role = message["role"]
            
            # If same role as previous, merge the content
            if last_role == current_role and fixed_messages:
                fixed_messages[-1]["content"] += "\n\n" + message["content"]
            else:
                # If we have two user messages in a row, insert a brief assistant response
                if last_role == "user" and current_role == "user" and fixed_messages:
                    fixed_messages.append({
                        "role": "assistant",
                        "content": "I understand. Please continue."
                    })
                
                fixed_messages.append(message)
                last_role = current_role
        
        # Ensure we start with a user message
        if fixed_messages and fixed_messages[0]["role"] == "assistant":
            fixed_messages.insert(0, {
                "role": "user",
                "content": "Hello, I have a question."
            })
        
        return fixed_messages
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Claude's tokenization is roughly 3.5-4 characters per token
        return len(text) // 4
    
    def get_max_context_length(self) -> int:
        """Get maximum context length for the model"""
        # Context lengths for Anthropic models
        context_lengths = {
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000
        }
        
        return context_lengths.get(self.model, 100000)  # Default to 100k
    
    def truncate_conversation(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Truncate conversation to fit within context limits"""
        if max_tokens is None:
            max_tokens = self.get_max_context_length() - self.default_max_tokens - 1000  # Buffer
        
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
        
        # Ensure proper format for Anthropic
        if truncated_messages:
            truncated_messages, _ = self._format_messages_for_anthropic(truncated_messages)
        
        return truncated_messages
    
    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling"""
        # Claude 3 models support function calling (tools)
        function_calling_models = [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022"
        ]
        return self.model in function_calling_models
    
    def get_model_family(self) -> str:
        """Get the model family (haiku, sonnet, opus)"""
        if "haiku" in self.model:
            return "haiku"
        elif "sonnet" in self.model:
            return "sonnet"
        elif "opus" in self.model:
            return "opus"
        elif "claude-2" in self.model:
            return "claude-2"
        elif "instant" in self.model:
            return "instant"
        else:
            return "unknown"
