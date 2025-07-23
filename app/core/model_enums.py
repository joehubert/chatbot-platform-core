from enum import Enum

class ModelType(Enum):
    RELEVANCE = "relevance"
    SIMPLE_QUERY = "simple_query"
    COMPLEX_QUERY = "complex_query"
    CLARIFICATION = "clarification"
    EMBEDDING = "embedding"

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"