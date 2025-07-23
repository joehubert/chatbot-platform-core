from typing import Dict, Any, Optional, List, AsyncGenerator, Protocol
from enum import Enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from app.core.model_enums import ModelType, ModelProvider
from app.core.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Unified response format for all model providers"""
    content: str
    model: str
    provider: str
    model_type: ModelType
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseModelClient(ABC):
    """Abstract base class for model clients"""
    
    def __init__(self, provider: ModelProvider, model: str, config: Dict[str, Any]):
        self.provider = provider
        self.model = model
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model client"""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model_type: ModelType,
        **kwargs
    ) -> Optional[ModelResponse]:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]], 
        model_type: ModelType,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the model"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model service is healthy"""
        pass

class ModelConfiguration:
    """Configuration container for a specific model type"""
    
    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        self.model_type = model_type
        self.provider = ModelProvider(config['provider'])
        self.model = config['model']
        self.config = config
        
    def get_client_key(self) -> str:
        """Generate unique key for this model configuration"""
        return f"{self.provider.value}_{self.model}_{self.model_type.value}"

class ModelFactory:
    """Factory for creating and managing model clients by type"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._clients: Dict[str, BaseModelClient] = {}
        self._configurations: Dict[ModelType, ModelConfiguration] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the model factory with configurations"""
        if self._initialized:
            return
            
        # Load model type configurations
        model_configs = self.settings.get_model_type_configs()
        
        for model_type_str, config in model_configs.items():
            model_type = ModelType(model_type_str)
            self._configurations[model_type] = ModelConfiguration(model_type, config)
        
        logger.info(f"Model factory initialized with {len(self._configurations)} model types")
        self._initialized = True
    
    async def get_client_for_type(self, model_type: ModelType) -> Optional[BaseModelClient]:
        """Get or create a model client for the specified type"""
        if not self._initialized:
            await self.initialize()
        
        if model_type not in self._configurations:
            logger.error(f"No configuration found for model type: {model_type}")
            return None
        
        config = self._configurations[model_type]
        client_key = config.get_client_key()
        
        # Return existing client if available
        if client_key in self._clients:
            return self._clients[client_key]
        
        # Create new client
        client = await self._create_client(config)
        if client:
            self._clients[client_key] = client
            await client.initialize()
        
        return client
    
    async def _create_client(self, config: ModelConfiguration) -> Optional[BaseModelClient]:
        """Create a model client based on provider"""
        try:
            if config.provider == ModelProvider.OPENAI:
                return await self._create_openai_client(config)
            elif config.provider == ModelProvider.ANTHROPIC:
                return await self._create_anthropic_client(config)
            elif config.provider == ModelProvider.OLLAMA:
                return await self._create_ollama_client(config)
            elif config.provider == ModelProvider.HUGGINGFACE:
                return await self._create_huggingface_client(config)
            elif config.provider == ModelProvider.AZURE_OPENAI:
                return await self._create_azure_openai_client(config)
            else:
                logger.error(f"Unsupported provider: {config.provider}")
                return None
        except Exception as e:
            logger.error(f"Failed to create client for {config.provider}/{config.model}: {e}")
            return None
    
    async def _create_openai_client(self, config: ModelConfiguration) -> Optional[BaseModelClient]:
        """Create OpenAI client"""
        from app.services.model_clients.openai_client import OpenAIClient
        return OpenAIClient(
            provider=config.provider,
            model=config.model,
            config={
                **config.config,
                'api_key': self.settings.OPENAI_API_KEY
            }
        )
    
    async def _create_anthropic_client(self, config: ModelConfiguration) -> Optional[BaseModelClient]:
        """Create Anthropic client"""
        from app.services.model_clients.anthropic_client import AnthropicClient
        return AnthropicClient(
            provider=config.provider,
            model=config.model,
            config={
                **config.config,
                'api_key': self.settings.ANTHROPIC_API_KEY
            }
        )
    
    async def _create_ollama_client(self, config: ModelConfiguration) -> Optional[BaseModelClient]:
        """Create Ollama client"""
        from app.services.model_clients.ollama_client import OllamaClient
        return OllamaClient(
            provider=config.provider,
            model=config.model,
            config={
                **config.config,
                'base_url': self.settings.OLLAMA_BASE_URL
            }
        )
    
    # Add other provider client creation methods...
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Health check all model clients"""
        results = {}
        for client_key, client in self._clients.items():
            try:
                results[client_key] = await client.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {client_key}: {e}")
                results[client_key] = False
        return results
    
    def get_configuration(self, model_type: ModelType) -> Optional[ModelConfiguration]:
        """Get configuration for a model type"""
        return self._configurations.get(model_type)
    
    def list_configured_types(self) -> List[ModelType]:
        """List all configured model types"""
        return list(self._configurations.keys())