# Chatbot Platform Core

Enterprise-grade, self-hosted AI chatbot platform designed for small and medium enterprises (SMEs). Built with Python/FastAPI, this platform provides intelligent conversation management, RAG capabilities, multi-LLM support, and comprehensive analytics.

## 🚀 Overview

The Chatbot Platform Core is a single-tenant, self-hosted solution that organizations deploy and customize for their specific needs. It features an intelligent LangGraph-based processing pipeline, configurable LLM backends, organization-specific knowledge base integration, and professional-grade monitoring and analytics.

### Key Features

- **🧠 Intelligent Processing Pipeline** - LangGraph-based request processing with rate limiting, relevance checking, semantic caching, and model routing
- **📚 RAG Knowledge Base** - Document upload, processing, chunking, and vector storage with expiration management
- **🤖 Multi-LLM Support** - Pluggable architecture supporting OpenAI, Anthropic, HuggingFace, and local models
- **🔐 Authentication System** - SMS/Email OTP authentication with secure session management
- **📊 Analytics & Monitoring** - Comprehensive conversation tracking, performance metrics, and cost analysis
- **🔧 MCP Integration** - Model Context Protocol support for external system integration
- **⚡ Semantic Caching** - Vector-based response caching for improved performance
- **🛡️ Enterprise Security** - End-to-end encryption, secure token management, and audit logging

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Widget (JS)  │  Admin UI (React)  │  Mobile App  │  API Clients │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway (FastAPI)                     │
├─────────────────────────────────────────────────────────────────┤
│  Authentication │  Rate Limiting  │  Request Validation  │  CORS │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Processing Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│ Rate Limit → Relevance → Cache Check → Model Route → Auth → RAG │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   LLM Providers │ │   Vector Store  │ │   MCP Servers   │
│                 │ │                 │ │                 │
│ • OpenAI        │ │ • Pinecone      │ │ • CRM Systems   │
│ • Anthropic     │ │ • Chroma        │ │ • Databases     │
│ • HuggingFace   │ │ • Weaviate      │ │ • APIs          │
│ • Local Models  │ │ • pgvector      │ │ • Tools         │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data & Cache Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL  │  Redis Cache  │  File Storage  │  Audit Logs    │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Core Components

### 1. API Layer (`app/api/`)
- **Chat Endpoints** - Message processing, conversation management
- **Knowledge Base** - Document upload, management, search
- **Authentication** - OTP generation, verification, session management
- **Configuration** - System settings, model configuration
- **Analytics** - Metrics, reporting, performance data

### 2. Processing Pipeline (`app/core/pipeline/`)
- **Rate Limiter** - Token bucket algorithm with Redis backend
- **Relevance Checker** - LLM-based query classification
- **Semantic Cache** - Vector similarity-based response caching
- **Model Router** - Intelligent routing based on query complexity
- **Auth Handler** - Conditional authentication flow
- **RAG Engine** - Knowledge base query and context injection

### 3. LLM Integration (`app/services/llm/`)
- **Provider Factory** - Pluggable LLM provider architecture
- **Model Manager** - Dynamic model selection and configuration
- **Cost Tracker** - Token usage and cost monitoring
- **Fallback Handler** - Automatic failover between providers

### 4. Knowledge Management (`app/services/knowledge/`)
- **Document Processor** - File parsing, chunking, metadata extraction
- **Vector Manager** - Embedding generation and storage
- **Expiration Handler** - Automated content lifecycle management
- **Search Engine** - Semantic search and retrieval

### 5. Data Layer (`app/models/`)
- **Conversation Models** - Chat sessions, messages, context
- **User Models** - Authentication, sessions, preferences  
- **Document Models** - Knowledge base content, metadata
- **Analytics Models** - Metrics, usage tracking, performance

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **PostgreSQL 15+**
- **Redis 7+**
- **LLM API Keys** (OpenAI, Anthropic, etc.)

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/chatbot-platform-core.git
cd chatbot-platform-core

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://chatbot:password@localhost:5432/chatbot_db
REDIS_URL=redis://localhost:6379/0

# LLM Provider Configuration
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
HUGGINGFACE_API_KEY=hf_your-huggingface-key

# Vector Database
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env

# Pipeline Configuration
RATE_LIMIT_PER_USER_PER_MINUTE=60
RATE_LIMIT_GLOBAL_PER_MINUTE=1000
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_TTL_HOURS=24

# Model Configuration
RELEVANCE_MODEL=gpt-3.5-turbo
SIMPLE_QUERY_MODEL=gpt-3.5-turbo
COMPLEX_QUERY_MODEL=gpt-4
CLARIFICATION_MODEL=gpt-3.5-turbo

# Authentication
AUTH_SESSION_TIMEOUT_MINUTES=30
OTP_EXPIRY_MINUTES=5
SMS_PROVIDER=twilio
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=your-sendgrid-key

# Security
SECRET_KEY=your-super-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# File Upload
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf,txt,docx,md
UPLOAD_STORAGE_PATH=./uploads

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=chatbot-platform
```

### 3. Database Setup

```bash
# Start PostgreSQL and Redis (if using Docker)
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Create initial admin user (optional)
python scripts/create_admin.py
```

### 4. Start the Platform

#### Development Mode
```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode with Docker
```bash
# Build and start all services
docker-compose up -d

# Check service health
docker-compose ps
curl http://localhost:8000/health
```

### 5. Verify Installation

```bash
# Test the chat endpoint
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, can you help me?"}'

# Check system health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

## ⚙️ Configuration Guide

### LLM Provider Setup

#### OpenAI Configuration
```bash
OPENAI_API_KEY=sk-your-key
OPENAI_ORG_ID=org-your-org-id  # Optional
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
```

#### Anthropic Configuration
```bash
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_BASE_URL=https://api.anthropic.com  # Optional
```

#### Local Model Configuration (Ollama)
```bash
LOCAL_MODEL_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_MODEL_NAME=llama2:7b
```

### Vector Database Configuration

#### Pinecone Setup
```bash
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=your-environment
PINECONE_INDEX_NAME=chatbot-knowledge
```

#### Chroma Setup (Local)
```bash
VECTOR_DB_TYPE=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=chatbot-docs
```

#### pgvector Setup (PostgreSQL Extension)
```bash
VECTOR_DB_TYPE=pgvector
# Uses same DATABASE_URL as main database
PGVECTOR_DIMENSION=1536  # Match your embedding model
```

### Pipeline Configuration

#### Rate Limiting
```bash
# Per-user limits
RATE_LIMIT_PER_USER_PER_MINUTE=60
RATE_LIMIT_BURST_CAPACITY=10

# Global limits
RATE_LIMIT_GLOBAL_PER_MINUTE=1000
RATE_LIMIT_WINDOW_SIZE=60
```

#### Semantic Cache
```bash
CACHE_SIMILARITY_THRESHOLD=0.85  # 0.0-1.0
CACHE_TTL_HOURS=24
CACHE_MAX_ENTRIES=10000
CACHE_CLEANUP_INTERVAL_HOURS=6
```

#### Model Routing
```bash
# Complexity thresholds (0.0-1.0)
SIMPLE_QUERY_THRESHOLD=0.3
COMPLEX_QUERY_THRESHOLD=0.7

# Model assignments
SIMPLE_QUERY_MODEL=gpt-3.5-turbo
COMPLEX_QUERY_MODEL=gpt-4
RELEVANCE_MODEL=gpt-3.5-turbo
CLARIFICATION_MODEL=gpt-3.5-turbo
```

### Authentication Configuration

#### SMS Provider (Twilio)
```bash
SMS_PROVIDER=twilio
TWILIO_ACCOUNT_SID=ACxxxxx
TWILIO_AUTH_TOKEN=your-token
TWILIO_PHONE_NUMBER=+1234567890
```

#### Email Provider (SendGrid)
```bash
EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=SG.xxxxx
FROM_EMAIL=noreply@yourdomain.com
FROM_NAME=Your Chatbot
```

## 🐳 Docker Deployment

### Development Environment
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://chatbot:password@postgres:5432/chatbot_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=chatbot_db
      - POSTGRES_USER=chatbot
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  postgres_data:
  redis_data:
  chroma_data:
```

### Production Environment
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Health check
docker-compose exec app python scripts/health_check.py

# View logs
docker-compose logs -f app

# Scale services
docker-compose up -d --scale app=3
```

## 📊 API Reference

### Chat API

#### Send Message
```http
POST /api/v1/chat/message
Content-Type: application/json

{
  "message": "string",
  "session_id": "string (optional)",
  "user_id": "string (optional)",
  "context": {
    "page_url": "string",
    "user_agent": "string"
  }
}
```

**Response:**
```json
{
  "response": "string",
  "session_id": "string",
  "requires_auth": false,
  "auth_methods": ["sms", "email"],
  "conversation_id": "string",
  "cached": false,
  "model_used": "gpt-3.5-turbo",
  "processing_time_ms": 1250,
  "sources": ["doc1.pdf", "doc2.txt"]
}
```

#### Request Authentication
```http
POST /api/v1/chat/auth/request
Content-Type: application/json

{
  "session_id": "string",
  "contact_method": "sms",
  "contact_value": "+1234567890"
}
```

#### Verify Authentication
```http
POST /api/v1/chat/auth/verify
Content-Type: application/json

{
  "session_id": "string",
  "token": "123456"
}
```

### Knowledge Base API

#### Upload Document
```http
POST /api/v1/knowledge/documents
Content-Type: multipart/form-data

file: (binary)
expiration_date: 2024-12-31T23:59:59Z
category: general
metadata: {"source": "manual", "version": "1.0"}
```

#### List Documents
```http
GET /api/v1/knowledge/documents?category=general&limit=50&offset=0
```

#### Delete Document
```http
DELETE /api/v1/knowledge/documents/{document_id}
```

### Configuration API

#### Get Configuration
```http
GET /api/v1/config
Authorization: Bearer {admin_token}
```

#### Update Configuration
```http
PUT /api/v1/config
Authorization: Bearer {admin_token}
Content-Type: application/json

{
  "rate_limits": {
    "per_user_per_minute": 60,
    "global_per_minute": 1000
  },
  "models": {
    "simple_query": "gpt-3.5-turbo",
    "complex_query": "gpt-4"
  }
}
```

### Analytics API

#### Conversation Metrics
```http
GET /api/v1/analytics/conversations?start_date=2024-01-01&end_date=2024-01-31
```

#### Performance Metrics
```http
GET /api/v1/analytics/performance?metric=response_time&granularity=hour
```

## 🔧 Development Guide

### Project Structure
```
chatbot-platform-core/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── chat.py        # Chat endpoints
│   │   │   ├── knowledge.py   # Knowledge base endpoints
│   │   │   ├── auth.py        # Authentication endpoints
│   │   │   ├── config.py      # Configuration endpoints
│   │   │   └── analytics.py   # Analytics endpoints
│   ├── core/                   # Business logic
│   │   ├── __init__.py
│   │   ├── pipeline/          # LangGraph processing pipeline
│   │   │   ├── __init__.py
│   │   │   ├── rate_limiter.py
│   │   │   ├── relevance_checker.py
│   │   │   ├── semantic_cache.py
│   │   │   ├── model_router.py
│   │   │   ├── auth_handler.py
│   │   │   └── rag_engine.py
│   │   ├── security/          # Security utilities
│   │   └── config.py          # Configuration management
│   ├── models/                # Database models
│   │   ├── __init__.py
│   │   ├── conversation.py
│   │   ├── user.py
│   │   ├── document.py
│   │   └── analytics.py
│   ├── schemas/               # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── auth.py
│   │   ├── knowledge.py
│   │   └── config.py
│   ├── services/              # External service integrations
│   │   ├── __init__.py
│   │   ├── llm/              # LLM provider services
│   │   │   ├── __init__.py
│   │   │   ├── factory.py
│   │   │   ├── openai.py
│   │   │   ├── anthropic.py
│   │   │   └── local.py
│   │   ├── vector/           # Vector database services
│   │   │   ├── __init__.py
│   │   │   ├── factory.py
│   │   │   ├── pinecone.py
│   │   │   ├── chroma.py
│   │   │   └── pgvector.py
│   │   ├── knowledge/        # Knowledge management
│   │   ├── auth/            # Authentication services
│   │   └── mcp/             # MCP server integration
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── logging.py
│       ├── metrics.py
│       └── helpers.py
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── alembic/                   # Database migrations
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
├── docker-compose.yml
├── docker-compose.prod.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/unit/test_pipeline.py -v
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# View migration history
alembic history
```

### Code Quality
```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/

# Security scan
bandit -r app/
```

## 📈 Monitoring and Observability

### Health Checks
The platform provides comprehensive health monitoring:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health with dependencies
curl http://localhost:8000/health/detailed
```

### Metrics Collection
- **Request Metrics**: Response times, error rates, throughput
- **LLM Metrics**: Token usage, costs, model performance
- **Cache Metrics**: Hit rates, eviction rates, memory usage
- **Database Metrics**: Query performance, connection pool status

### Logging
```python
# Structured logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
            "level": "INFO"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

## 🔒 Security Considerations

### Authentication & Authorization
- **JWT Tokens**: Secure session management with configurable expiration
- **OTP Verification**: SMS/Email one-time passwords for sensitive operations
- **Role-Based Access**: Admin, user, and service account roles
- **API Key Management**: Secure storage and rotation of LLM provider keys

### Data Protection
- **Encryption at Rest**: Database and file storage encryption
- **Encryption in Transit**: TLS 1.3 for all communications
- **PII Handling**: Automatic detection and masking of personal information
- **Audit Logging**: Comprehensive audit trail for all operations

### Security Headers
```python
# Security middleware configuration
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}
```

## 📚 Knowledge Base Management

### Document Processing Pipeline
1. **Upload** - Multi-format file upload with validation
2. **Parsing** - Extract text content from various file formats
3. **Chunking** - Intelligent text segmentation with overlap
4. **Embedding** - Generate vector embeddings for semantic search
5. **Storage** - Store in vector database with metadata
6. **Indexing** - Build search indexes for fast retrieval

### Supported Formats
- **PDF** - Adobe PDF documents
- **DOCX** - Microsoft Word documents
- **TXT** - Plain text files
- **MD** - Markdown files
- **HTML** - Web pages and HTML documents

### Content Lifecycle
```python
# Document expiration management
@scheduled_task(cron="0 2 * * *")  # Daily at 2 AM
async def cleanup_expired_documents():
    """Remove expired documents from knowledge base"""
    expired_docs = await get_expired_documents()
    for doc in expired_docs:
        await remove_from_vector_store(doc.vector_ids)
        await delete_document(doc.id)
        logger.info(f"Removed expired document: {doc.filename}")
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -r requirements-dev.txt`)
4. Make your changes
5. Run tests (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards
- **Python**: Follow PEP 8 style guide
- **Type Hints**: Use type hints for all functions and classes
- **Documentation**: Docstrings for all public methods
- **Testing**: Minimum 80% test coverage for new code
- **Security**: Run security scans before committing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Documentation](https://docs.yourdomain.com/api)
- [Deployment Guide](https://docs.yourdomain.com/deployment)
- [Configuration Reference](https://docs.yourdomain.com/configuration)

### Getting Help
- [GitHub Issues](https://github.com/your-org/chatbot-platform-core/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/your-org/chatbot-platform-core/discussions) - Questions and community support
- [Professional Support](https://yourdomain.com/support) - Enterprise support and consulting

### Community
- [Discord Server](https://discord.gg/your-server) - Real-time community chat
- [Newsletter](https://yourdomain.com/newsletter) - Updates and announcements

---

**Built with ❤️ for SMEs who need powerful, customizable chatbot solutions.**