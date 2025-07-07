# Chatbot Platform Core - Project Structure

## Directory Layout

```
chatbot-platform-core/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── docker-compose.dev.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   │
│   ├── api/                       # API routes and endpoints
│   │   ├── __init__.py
│   │   ├── deps.py               # Dependencies and dependency injection
│   │   ├── v1/                   # API version 1
│   │   │   ├── __init__.py
│   │   │   ├── api.py            # Main API router
│   │   │   ├── endpoints/        # Individual endpoint modules
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chat.py       # Chat message processing
│   │   │   │   ├── auth.py       # Authentication endpoints
│   │   │   │   ├── knowledge.py  # Knowledge base management
│   │   │   │   ├── config.py     # Configuration management
│   │   │   │   ├── analytics.py  # Analytics and metrics
│   │   │   │   └── health.py     # Health checks
│   │   │   └── middleware/       # Custom middleware
│   │   │       ├── __init__.py
│   │   │       ├── rate_limit.py # Rate limiting middleware
│   │   │       ├── auth.py       # Authentication middleware
│   │   │       └── logging.py    # Request logging middleware
│   │   └── exceptions/           # Custom exception handlers
│   │       ├── __init__.py
│   │       ├── handlers.py       # Exception handler functions
│   │       └── exceptions.py     # Custom exception classes
│   │
│   ├── core/                     # Business logic and services
│   │   ├── __init__.py
│   │   ├── config.py            # Application configuration
│   │   ├── security.py          # Security utilities
│   │   ├── logging.py           # Logging configuration
│   │   │
│   │   ├── pipeline/            # LangGraph processing pipeline
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Base pipeline classes
│   │   │   ├── graph.py         # LangGraph implementation
│   │   │   ├── nodes/           # Pipeline processing nodes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── rate_limit.py       # Rate limiting node
│   │   │   │   ├── relevance_check.py  # Relevance checking node
│   │   │   │   ├── semantic_cache.py   # Semantic cache node
│   │   │   │   ├── model_routing.py    # Model routing node
│   │   │   │   ├── auth_check.py       # Authentication node
│   │   │   │   ├── question_processing.py # RAG and MCP processing
│   │   │   │   ├── response_validation.py # Response validation
│   │   │   │   ├── resolution_confirmation.py # Resolution tracking
│   │   │   │   ├── conversation_recording.py # Conversation storage
│   │   │   │   └── cache_update.py     # Cache update node
│   │   │   └── utils/           # Pipeline utilities
│   │   │       ├── __init__.py
│   │   │       ├── state.py     # Pipeline state management
│   │   │       └── metrics.py   # Pipeline metrics collection
│   │   │
│   │   ├── llm/                 # LLM provider integration
│   │   │   ├── __init__.py
│   │   │   ├── factory.py       # LLM factory pattern
│   │   │   ├── base.py          # Base LLM interface
│   │   │   ├── providers/       # LLM provider implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── openai.py    # OpenAI integration
│   │   │   │   ├── anthropic.py # Anthropic (Claude) integration
│   │   │   │   ├── huggingface.py # HuggingFace integration
│   │   │   │   └── local.py     # Local model integration
│   │   │   └── utils/           # LLM utilities
│   │   │       ├── __init__.py
│   │   │       ├── embeddings.py # Embedding generation
│   │   │       ├── prompts.py   # Prompt templates
│   │   │       └── routing.py   # Model routing logic
│   │   │
│   │   ├── vector/              # Vector database integration
│   │   │   ├── __init__.py
│   │   │   ├── factory.py       # Vector DB factory
│   │   │   ├── base.py          # Base vector DB interface
│   │   │   ├── providers/       # Vector DB implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── pinecone.py  # Pinecone implementation
│   │   │   │   ├── chroma.py    # Chroma implementation
│   │   │   │   ├── weaviate.py  # Weaviate implementation
│   │   │   │   └── pgvector.py  # PostgreSQL pgvector
│   │   │   └── utils/           # Vector utilities
│   │   │       ├── __init__.py
│   │   │       ├── chunking.py  # Document chunking
│   │   │       └── similarity.py # Similarity calculations
│   │   │
│   │   ├── knowledge/           # Knowledge base management
│   │   │   ├── __init__.py
│   │   │   ├── processor.py     # Document processing
│   │   │   ├── extractor.py     # Text extraction
│   │   │   ├── chunker.py       # Document chunking
│   │   │   └── embedder.py      # Embedding generation
│   │   │
│   │   ├── auth/                # Authentication system
│   │   │   ├── __init__.py
│   │   │   ├── token_manager.py # OTP token management
│   │   │   ├── session_manager.py # Session management
│   │   │   ├── providers/       # Auth providers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── sms.py       # SMS provider (Twilio)
│   │   │   │   └── email.py     # Email provider (SendGrid)
│   │   │   └── utils/           # Auth utilities
│   │   │       ├── __init__.py
│   │   │       └── validators.py # Input validation
│   │   │
│   │   ├── cache/               # Caching system
│   │   │   ├── __init__.py
│   │   │   ├── semantic.py      # Semantic cache implementation
│   │   │   ├── redis_client.py  # Redis client wrapper
│   │   │   └── utils/           # Cache utilities
│   │   │       ├── __init__.py
│   │   │       └── keys.py      # Cache key generation
│   │   │
│   │   └── mcp/                 # Model Context Protocol integration
│   │       ├── __init__.py
│   │       ├── client.py        # MCP client implementation
│   │       ├── server_manager.py # MCP server management
│   │       └── protocol/        # MCP protocol definitions
│   │           ├── __init__.py
│   │           ├── types.py     # MCP type definitions
│   │           └── handlers.py  # Message handlers
│   │
│   ├── models/                  # Database models (SQLAlchemy)
│   │   ├── __init__.py
│   │   ├── base.py             # Base model class
│   │   ├── conversation.py      # Conversation model
│   │   ├── message.py          # Message model
│   │   ├── document.py         # Document model
│   │   ├── user.py             # User model
│   │   ├── auth_token.py       # Auth token model
│   │   ├── organization.py     # Organization model
│   │   └── analytics.py        # Analytics models
│   │
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   ├── base.py            # Base schema classes
│   │   ├── chat.py            # Chat request/response schemas
│   │   ├── auth.py            # Authentication schemas
│   │   ├── knowledge.py       # Knowledge base schemas
│   │   ├── config.py          # Configuration schemas
│   │   ├── analytics.py       # Analytics schemas
│   │   └── common.py          # Common shared schemas
│   │
│   ├── services/              # External service integrations
│   │   ├── __init__.py
│   │   ├── database.py        # Database service
│   │   ├── redis.py          # Redis service
│   │   ├── file_storage.py   # File storage service
│   │   └── notification/     # Notification services
│   │       ├── __init__.py
│   │       ├── sms.py        # SMS service (Twilio)
│   │       └── email.py      # Email service (SendGrid)
│   │
│   └── utils/                 # Utilities and helpers
│       ├── __init__.py
│       ├── dependencies.py    # FastAPI dependencies
│       ├── exceptions.py      # Custom exceptions
│       ├── validators.py      # Input validators
│       ├── formatters.py      # Data formatters
│       ├── constants.py       # Application constants
│       └── helpers.py         # Helper functions
│
├── alembic/                   # Database migrations
│   ├── versions/              # Migration versions
│   ├── env.py                # Alembic environment
│   ├── script.py.mako        # Migration template
│   └── alembic.ini           # Alembic configuration
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── test_main.py          # Main application tests
│   │
│   ├── api/                  # API endpoint tests
│   │   ├── __init__.py
│   │   ├── test_chat.py      # Chat endpoint tests
│   │   ├── test_auth.py      # Auth endpoint tests
│   │   ├── test_knowledge.py # Knowledge endpoint tests
│   │   └── test_health.py    # Health check tests
│   │
│   ├── core/                 # Core business logic tests
│   │   ├── __init__.py
│   │   ├── test_pipeline.py  # Pipeline tests
│   │   ├── test_llm.py       # LLM integration tests
│   │   ├── test_vector.py    # Vector DB tests
│   │   ├── test_auth.py      # Auth system tests
│   │   └── test_cache.py     # Cache system tests
│   │
│   ├── models/               # Database model tests
│   │   ├── __init__.py
│   │   └── test_models.py    # Model validation tests
│   │
│   ├── integration/          # Integration tests
│   │   ├── __init__.py
│   │   ├── test_e2e.py       # End-to-end tests
│   │   └── test_pipeline_integration.py
│   │
│   └── fixtures/             # Test fixtures and data
│       ├── __init__.py
│       ├── test_data.py      # Test data fixtures
│       └── sample_files/     # Sample documents for testing
│
├── scripts/                   # Utility scripts
│   ├── init_db.py            # Database initialization
│   ├── seed_data.py          # Seed development data
│   ├── migrate.py            # Migration helper
│   ├── health_check.py       # Health check script
│   └── backup.py             # Backup utilities
│
├── docs/                      # Documentation
│   ├── api/                  # API documentation
│   ├── deployment/           # Deployment guides
│   ├── development/          # Development setup
│   └── architecture/         # Architecture documentation
│
└── config/                    # Configuration files
    ├── logging.yaml          # Logging configuration
    ├── prompts/              # LLM prompt templates
    │   ├── relevance_check.txt
    │   ├── clarification.txt
    │   └── response_generation.txt
    └── examples/             # Example configurations
        ├── docker-compose.example.yml
        └── env.example
```

## Key Design Principles

### 1. Modular Architecture
- Clear separation of concerns with distinct modules for each major component
- Pluggable architecture for LLM providers and vector databases
- Factory patterns for easy switching between implementations

### 2. Scalability
- Stateless application design for horizontal scaling
- Async/await patterns throughout for better concurrency
- Efficient database design with proper indexing

### 3. Maintainability
- Comprehensive type hints and docstrings
- Structured logging with correlation IDs
- Extensive test coverage with clear test organization

### 4. Configuration Management
- Environment-based configuration with validation
- Separate configurations for development, staging, and production
- Centralized configuration with proper validation

### 5. Security
- JWT-based authentication with secure token management
- Input validation at multiple layers
- Secure handling of sensitive data and API keys

## Development Workflow

1. **Core Infrastructure**: Database models, basic API structure
2. **Pipeline Implementation**: LangGraph-based processing pipeline
3. **LLM Integration**: Multi-provider LLM support
4. **Vector Database**: Knowledge base and semantic search
5. **Authentication**: OTP-based authentication system
6. **Caching**: Semantic cache implementation
7. **MCP Integration**: Model Context Protocol support
8. **Analytics**: Metrics collection and reporting
9. **Testing**: Comprehensive test suite
10. **Documentation**: API docs and deployment guides