"""
LLM Factory Service

Factory pattern implementation for creating and managing different LLM clients.
Supports multiple providers with unified interface and fallback mechanisms.
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


class LLMResponse:
    """Unified response format for all LLM providers"""
    
    def __init__(
        self,
        content: str,
        model: str,
        provider: str,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.model = model
        self.provider = provider
        self.usage = usage or {}
        self.metadata = metadata or {}


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get("provider")
        self.model = config.get("model")
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Optional[LLMResponse]:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy"""
        pass
    
    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for the specific provider (override if needed)"""
        return messages


class LLMFactory:
    """Factory for creating and managing LLM clients"""
    
    def __init__(self):
        self._clients = {}
        self._provider_configs = {}
    
    def register_provider_config(self, provider: LLMProvider, config: Dict[str, Any]):
        """Register configuration for a provider"""
        self._provider_configs[provider] = config
        logger.info(f"Registered configuration for provider: {provider.value}")
    
    def create_client(self, provider: LLMProvider, model: str, **kwargs) -> Optional[BaseLLMClient]:
        """Create an LLM client for the specified provider and model"""
        client_key = f"{provider.value}_{model}"
        
        if client_key in self._clients:
            return self._clients[client_key]
        
        try:
            # Get provider config
            provider_config = self._provider_configs.get(provider, {})
            
            # Merge with specific model config
            config = {
                **provider_config,
                "provider": provider.value,
                "model": model,
                **kwargs
            }
            
            # Create client based on provider
            if provider == LLMProvider.OPENAI:
                from .openai_client import OpenAIClient
                client = OpenAIClient(config)
            elif provider == LLMProvider.ANTHROPIC:
                from .anthropic_client import AnthropicClient
                client = AnthropicClient(config)
            elif provider == LLMProvider.HUGGINGFACE:
                from .huggingface_client import HuggingFaceClient
                client = HuggingFaceClient(config)
            elif provider == LLMProvider.OLLAMA:
                from .ollama_client import OllamaClient
                client = OllamaClient(config)
            elif provider == LLMProvider.AZURE_OPENAI:
                from .azure_openai_client import AzureOpenAIClient
                client = AzureOpenAIClient(config)
            else:
                logger.error(f"Unsupported provider: {provider}")
                return None
            
            self._clients[client_key] = client
            logger.info(f"Created LLM client: {client_key}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create LLM client for {provider.value}/{model}: {e}")
            return None
    
    def get_client(self, provider: LLMProvider, model: str) -> Optional[BaseLLMClient]:
        """Get an existing LLM client"""
        client_key = f"{provider.value}_{model}"
        return self._clients.get(client_key)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Health check all registered clients"""
        results = {}
        
        for client_key, client in self._clients.items():
            try:
                results[client_key] = await client.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {client_key}: {e}")
                results[client_key] = False
        
        return results
    
    def list_clients(self) -> List[str]:
        """List all registered clients"""
        return list(self._clients.keys())
    
    def remove_client(self, provider: LLMProvider, model: str):
        """Remove a client from the factory"""
        client_key = f"{provider.value}_{model}"
        if client_key in self._clients:
            del self._clients[client_key]
            logger.info(f"Removed LLM client: {client_key}")


class LLMService:
    """High-level service for LLM operations with fallback support"""
    
    def __init__(self, factory: LLMFactory):
        self.factory = factory
        self.fallback_chains = {}
    
    def configure_fallback_chain(
        self, 
        primary_provider: LLMProvider, 
        primary_model: str,
        fallbacks: List[tuple]  # List of (provider, model) tuples
    ):
        """Configure fallback chain for a primary model"""
        chain_key = f"{primary_provider.value}_{primary_model}"
        self.fallback_chains[chain_key] = fallbacks
        logger.info(f"Configured fallback chain for {chain_key}: {fallbacks}")
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        provider: LLMProvider,
        model: str,
        use_fallback: bool = True,
        **kwargs
    ) -> Optional[LLMResponse]:
        """Generate response with optional fallback support"""
        
        # Try primary model
        client = self.factory.get_client(provider, model)
        if not client:
            client = self.factory.create_client(provider, model)
        
        if client:
            try:
                response = await client.generate_response(messages, **kwargs)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Primary model {provider.value}/{model} failed: {e}")
        
        # Try fallback chain if enabled
        if use_fallback:
            chain_key = f"{provider.value}_{model}"
            fallbacks = self.fallback_chains.get(chain_key, [])
            
            for fallback_provider_str, fallback_model in fallbacks:
                try:
                    fallback_provider = LLMProvider(fallback_provider_str)
                    logger.info(f"Trying fallback: {fallback_provider.value}/{fallback_model}")
                    
                    fallback_client = self.factory.get_client(fallback_provider, fallback_model)
                    if not fallback_client:
                        fallback_client = self.factory.create_client(fallback_provider, fallback_model)
                    
                    if fallback_client:
                        response = await fallback_client.generate_response(messages, **kwargs)
                        if response:
                            logger.info(f"Fallback successful: {fallback_provider.value}/{fallback_model}")
                            return response
                            
                except Exception as e:
                    logger.warning(f"Fallback {fallback_provider_str}/{fallback_model} failed: {e}")
                    continue
        
        logger.error(f"All models failed for request to {provider.value}/{model}")
        return None
    
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        provider: LLMProvider,
        model: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response (no fallback for streaming)"""
        
        client = self.factory.get_client(provider, model)
        if not client:
            client = self.factory.create_client(provider, model)
        
        if client:
            try:
                async for chunk in client.generate_streaming_response(messages, **kwargs):
                    yield chunk
                return
            except Exception as e:
                logger.error(f"Streaming failed for {provider.value}/{model}: {e}")
        
        # If streaming fails, yield error message
        yield f"Error: Streaming not available for {provider.value}/{model}"
    
    async def health_check(self, provider: LLMProvider, model: str) -> bool:
        """Health check for specific model"""
        client = self.factory.get_client(provider, model)
        if not client:
            return False
        
        try:
            return await client.health_check()
        except Exception as e:
            logger.error(f"Health check failed for {provider.value}/{model}: {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Health check all models in the factory"""
        return await self.factory.health_check_all()


# Utility functions for common LLM operations
def format_system_message(content: str) -> Dict[str, str]:
    """Format a system message"""
    return {"role": "system", "content": content}


def format_user_message(content: str) -> Dict[str, str]:
    """Format a user message"""
    return {"role": "user", "content": content}


def format_assistant_message(content: str) -> Dict[str, str]:
    """Format an assistant message"""
    return {"role": "assistant", "content": content}


def build_conversation(
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_message: Optional[str] = None
) -> List[Dict[str, str]]:
    """Build a conversation message list"""
    messages = []
    
    if system_prompt:
        messages.append(format_system_message(system_prompt))
    
    if conversation_history:
        messages.extend(conversation_history)
    
    if user_message:
        messages.append(format_user_message(user_message))
    
    return messages


# Global factory instance
_global_factory = LLMFactory()
_global_service = LLMService(_global_factory)


def get_llm_factory() -> LLMFactory:
    """Get the global LLM factory instance"""
    return _global_factory


def get_llm_service() -> LLMService:
    """Get the global LLM service instance"""
    return _global_service


def initialize_llm_providers(config: Dict[str, Any]):
    """Initialize LLM providers from configuration"""
    factory = get_llm_factory()
    
    # Register provider configurations
    if "openai" in config:
        factory.register_provider_config(LLMProvider.OPENAI, config["openai"])
    
    if "anthropic" in config:
        factory.register_provider_config(LLMProvider.ANTHROPIC, config["anthropic"])
    
    if "huggingface" in config:
        factory.register_provider_config(LLMProvider.HUGGINGFACE, config["huggingface"])
    
    if "ollama" in config:
        factory.register_provider_config(LLMProvider.OLLAMA, config["ollama"])
    
    if "azure_openai" in config:
        factory.register_provider_config(LLMProvider.AZURE_OPENAI, config["azure_openai"])
    
    # Configure fallback chains if specified
    service = get_llm_service()
    fallback_config = config.get("fallback_chains", {})
    
    for chain_config in fallback_config:
        primary_provider = LLMProvider(chain_config["primary_provider"])
        primary_model = chain_config["primary_model"]
        fallbacks = [(fb["provider"], fb["model"]) for fb in chain_config["fallbacks"]]
        
        service.configure_fallback_chain(primary_provider, primary_model, fallbacks)
    
    logger.info("LLM providers initialized successfully")
