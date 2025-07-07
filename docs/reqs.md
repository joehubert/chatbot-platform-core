# Turnkey AI Chatbot - Core Platform Requirements

## Project Overview

**Project Name**: Turnkey AI Chatbot - Core Platform  
**Repository**: `chatbot-platform-core`  
**Target Market**: Small and Medium Enterprises (SMEs)  
**Architecture**: Single-tenant, self-hosted platform with pluggable components

## System Architecture

### Technology Stack
- **Backend Framework**: Python 3.11+ with FastAPI
- **Primary Database**: PostgreSQL 15+
- **Vector Database**: Pluggable architecture, default Pinecone implementation
- **Cache/Queue**: Redis 7+
- **Deployment**: Docker Compose
- **API Documentation**: OpenAPI/Swagger auto-generated

### Core Components
```
chatbot-platform-core/
├── app/
│   ├── api/              # FastAPI routes and endpoints
│   ├── core/             # Business logic and services
│   ├── models/           # Database models (SQLAlchemy)
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # External service integrations
│   └── utils/            # Utilities and helpers
├── tests/
├── alembic/              # Database migrations
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Functional Requirements

### 1. Request Processing Pipeline (LangGraph Implementation)

The core platform must implement the following sequential processing pipeline:

#### 1.1 Rate Limiting
- **Per-user rate limiting**: Configurable requests per minute (default: 60)
- **Global rate limiting**: Configurable global requests per minute (default: 1000)
- **Implementation**: Redis-based token bucket algorithm
- **Response**: HTTP 429 when limits exceeded
- **Configuration**: Environment variables for thresholds

#### 1.2 Relevance Check
- **Purpose**: Determine if query relates to organization's domain
- **Implementation**: Configurable LLM model for classification
- **Clarification Loop**: 
  - Maximum attempts: Configurable (default: 3)
  - Uses same configurable model for clarification requests
  - Iteration affects threshold count
- **Configuration**: Model selection, confidence threshold, max attempts

#### 1.3 Semantic Cache Check
- **Cache Key Strategy**: Vector embedding similarity
- **Similarity Threshold**: Configurable (default: 0.85)
- **Cache Content**: Conversation summaries of resolved interactions
- **Cache Invalidation**: Automatic removal when content expires
- **Storage**: Redis with embedding vectors

#### 1.4 Model Routing
- **Trigger**: After cache miss
- **Logic**: Route based on query complexity analysis
- **Model Types**: 
  - Simple queries: Faster/cheaper models
  - Complex queries: More capable models
- **Configuration**: Model selection rules, complexity thresholds

#### 1.5 Authentication Check
- **Trigger**: When MCP server or tool requires authentication
- **Method**: One-time token via SMS/email
- **User Identification**: Mobile number or email address
- **Session Management**: 
  - Timeout: Configurable (default: 30-60 minutes)
  - Context preservation across authentication
- **Token Generation**: Secure random tokens with expiration

#### 1.6 Question Processing
- **RAG Integration**: Query organization-specific knowledge base
- **MCP Server Integration**: Connect to external systems via Model Context Protocol
- **Tool Integration**: Execute registered tools as needed
- **Context Management**: Maintain conversation context throughout process

#### 1.7 Response Validation
- **Quality Check**: Validate response appropriateness
- **Safety Check**: Filter inappropriate content
- **Fallback**: Graceful failure with configurable error messages

#### 1.8 Resolution Confirmation
- **User Feedback**: "Did this answer your question?"
- **Resolution Tracking**: Record success/failure for analytics

#### 1.9 Conversation Recording
- **Metrics**: Number of attempts, success rate, conversation summary
- **Storage**: PostgreSQL for audit and analytics
- **Purpose**: Future analysis and system improvement

#### 1.10 Cache Update
- **Action**: Add successful Q&A pairs to semantic cache
- **Format**: Conversation summaries with embedding vectors

### 2. Knowledge Base Management

#### 2.1 Document Upload
- **File Types**: PDF, TXT, DOCX, Markdown
- **Size Limits**: Configurable maximum file size
- **Processing**: Automatic chunking and embedding generation
- **Metadata**: Title, upload date, expiration date, category
- **Admin Control**: Expiration dates set by admin user

#### 2.2 Vector Database Integration
- **Architecture**: Pluggable adapter pattern
- **Default Implementation**: Pinecone
- **Operations**: 
  - Store document embeddings
  - Similarity search for RAG
  - Index management and updates
- **Expiration Handling**: Remove expired content from vector store

#### 2.3 Document Processing Pipeline
- **Text Extraction**: Support for multiple file formats
- **Chunking Strategy**: Configurable chunk size and overlap
- **Embedding Generation**: Integration with embedding models
- **Metadata Preservation**: Track source documents and sections

### 3. LLM Provider Integration

#### 3.1 Multi-Provider Support
- **Supported Providers**: OpenAI, Anthropic (Claude), Hugging Face
- **Architecture**: Factory pattern for model instantiation
- **Configuration**: Environment-based provider selection
- **Fallback Chains**: Automatic failover between providers

#### 3.2 Model Configuration
- **Model Selection**: Per-use-case model configuration
- **Usage Types**:
  - Relevance checking
  - Query processing
  - Response generation
  - Clarification requests
- **Cost Optimization**: Route to appropriate model based on complexity

### 4. MCP Server Integration

#### 4.1 Protocol Implementation
- **Standard**: Model Context Protocol for external system integration
- **Discovery**: Runtime discovery of available MCP servers
- **Authentication**: MCP servers declare authentication requirements
- **Error Handling**: Graceful fallback when MCP servers unavailable

#### 4.2 Server Management
- **Registration**: Dynamic registration of MCP servers
- **Health Monitoring**: Track MCP server availability
- **Load Balancing**: Distribute requests across multiple servers

### 5. User Authentication System

#### 5.1 One-Time Token Authentication
- **Trigger**: Authentication required by MCP server or tool
- **Delivery Methods**: 
  - SMS to registered mobile number
  - Email to registered email address
- **Token Properties**:
  - Cryptographically secure random generation
  - Configurable expiration time (default: 5 minutes)
  - Single-use tokens
- **User Identification**: Lookup by mobile number or email

#### 5.2 Session Management
- **Session Duration**: Configurable timeout (default: 30-60 minutes)
- **Context Preservation**: Maintain conversation context across authentication
- **Security**: Secure session token generation and validation

## API Specifications

### 1. Chat API

#### POST /api/v1/chat/message
**Purpose**: Process user chat messages through the complete pipeline

**Request Schema**:
```json
{
  "message": "string",
  "session_id": "string (optional)",
  "user_id": "string (optional)",
  "context": {
    "page_url": "string (optional)",
    "user_agent": "string (optional)"
  }
}
```

**Response Schema**:
```json
{
  "response": "string",
  "session_id": "string",
  "requires_auth": "boolean",
  "auth_methods": ["sms", "email"],
  "conversation_id": "string",
  "cached": "boolean",
  "model_used": "string"
}
```

#### POST /api/v1/chat/auth/request
**Purpose**: Request authentication token

**Request Schema**:
```json
{
  "session_id": "string",
  "contact_method": "sms | email",
  "contact_value": "string"
}
```

#### POST /api/v1/chat/auth/verify
**Purpose**: Verify authentication token

**Request Schema**:
```json
{
  "session_id": "string",
  "token": "string"
}
```

### 2. Knowledge Base API

#### POST /api/v1/knowledge/documents
**Purpose**: Upload and process documents

**Request**: Multipart form data with file and metadata

**Response Schema**:
```json
{
  "document_id": "string",
  "status": "processing | completed | failed",
  "chunks_created": "integer",
  "expiration_date": "datetime"
}
```

#### GET /api/v1/knowledge/documents
**Purpose**: List uploaded documents

#### DELETE /api/v1/knowledge/documents/{document_id}
**Purpose**: Remove document and associated embeddings

### 3. Configuration API

#### GET /api/v1/config
**Purpose**: Retrieve current system configuration

#### PUT /api/v1/config
**Purpose**: Update system configuration (admin only)

### 4. Analytics API

#### GET /api/v1/analytics/conversations
**Purpose**: Retrieve conversation metrics

#### GET /api/v1/analytics/performance
**Purpose**: Retrieve system performance metrics

## Data Models

### 1. Conversation
```python
class Conversation(Base):
    id: UUID
    session_id: str
    user_identifier: str (optional)
    started_at: datetime
    ended_at: datetime (optional)
    messages: List[Message]
    resolved: bool
    resolution_attempts: int
    authenticated: bool
```

### 2. Message
```python
class Message(Base):
    id: UUID
    conversation_id: UUID
    content: str
    role: str  # user, assistant, system
    timestamp: datetime
    model_used: str (optional)
    cached: bool
    processing_time_ms: int
```

### 3. Document
```python
class Document(Base):
    id: UUID
    filename: str
    content_type: str
    uploaded_at: datetime
    expires_at: datetime
    processed: bool
    chunk_count: int
    vector_ids: List[str]
```

### 4. User
```python
class User(Base):
    id: UUID
    mobile_number: str (optional)
    email: str (optional)
    created_at: datetime
    last_authenticated: datetime (optional)
```

### 5. AuthToken
```python
class AuthToken(Base):
    id: UUID
    user_id: UUID
    token: str
    expires_at: datetime
    used: bool
    session_id: str
```

## Configuration Management

### Environment Variables

#### Core Configuration
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/chatbot
REDIS_URL=redis://localhost:6379

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=ant-...
HUGGINGFACE_API_KEY=hf_...

# Vector Database
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=...

# Rate Limiting
RATE_LIMIT_PER_USER_PER_MINUTE=60
RATE_LIMIT_GLOBAL_PER_MINUTE=1000

# Cache Configuration
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_TTL_HOURS=24

# Authentication
AUTH_SESSION_TIMEOUT_MINUTES=30
OTP_EXPIRY_MINUTES=5
SMS_PROVIDER=twilio
EMAIL_PROVIDER=sendgrid

# Model Configuration
RELEVANCE_MODEL=gpt-3.5-turbo
SIMPLE_QUERY_MODEL=gpt-3.5-turbo
COMPLEX_QUERY_MODEL=gpt-4
CLARIFICATION_MODEL=gpt-3.5-turbo

# File Upload
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf,txt,docx,md

# Error Handling
FALLBACK_ERROR_MESSAGE="I'm having trouble right now. Please contact support at support@company.com"
```

#### Environment-Specific Configuration
- **Development**: Debug logging, relaxed rate limits
- **Staging**: Production-like settings with test data
- **Production**: Strict security, monitoring enabled

## Security Requirements

### 1. Authentication Security
- **Token Generation**: Cryptographically secure random tokens
- **Token Storage**: Hashed tokens in database
- **Session Security**: Secure session management with timeout
- **Rate Limiting**: Prevent brute force attacks

### 2. Data Protection
- **Encryption**: Encrypt sensitive data at rest
- **Input Validation**: Validate and sanitize all inputs
- **SQL Injection Prevention**: Use parameterized queries
- **XSS Prevention**: Proper output encoding

### 3. API Security
- **HTTPS Only**: All API communications over HTTPS
- **CORS Configuration**: Proper cross-origin settings
- **Request Validation**: Comprehensive input validation
- **Error Handling**: No sensitive information in error messages

## Performance Requirements

### 1. Response Times
- **Cached Responses**: < 500ms
- **Simple Queries**: < 2 seconds
- **Complex RAG Queries**: < 5 seconds
- **Authentication Flow**: < 30 seconds total

### 2. Scalability
- **Concurrent Users**: Support 100+ concurrent conversations
- **Document Processing**: Handle 1000+ documents in knowledge base
- **Request Volume**: 10,000+ requests per day
- **Database Performance**: Optimized queries with proper indexing

### 3. Availability
- **Uptime Target**: 99.5%
- **Failover**: Automatic failover for LLM providers
- **Health Monitoring**: Comprehensive health checks
- **Graceful Degradation**: Partial functionality during outages

## Deployment Requirements

### 1. Docker Configuration
```yaml
# docker-compose.yml structure
version: '3.8'
services:
  api:
    build: .
    environment:
      - DATABASE_URL
      - REDIS_URL
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB
      - POSTGRES_USER
      - POSTGRES_PASSWORD
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7
    volumes:
      - redis_data:/data
```

### 2. Health Checks
- **API Health**: `/health` endpoint with dependency checks
- **Database Connectivity**: PostgreSQL connection validation
- **Cache Availability**: Redis connection validation
- **External Services**: LLM provider and vector database status

### 3. Monitoring
- **Logging**: Structured logging with correlation IDs
- **Metrics**: Request counts, response times, error rates
- **Alerting**: Critical error notifications
- **Performance Tracking**: Database query performance, cache hit rates

## Testing Requirements

### 1. Unit Tests
- **Coverage**: Minimum 80% code coverage
- **Framework**: pytest with appropriate fixtures
- **Mocking**: Mock external services for isolated testing

### 2. Integration Tests
- **Database**: Test database operations with real PostgreSQL
- **Cache**: Test Redis operations
- **API**: Full API endpoint testing

### 3. End-to-End Tests
- **Complete Pipeline**: Test full conversation flow
- **Authentication**: Test complete auth flow
- **Document Processing**: Test file upload and processing

## Development Guidelines

### 1. Code Organization
- **Modular Architecture**: Clear separation of concerns
- **Dependency Injection**: Use FastAPI dependency injection
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging throughout application

### 2. Database Design
- **Migrations**: Alembic for database schema management
- **Indexing**: Proper database indexing for performance
- **Relationships**: Clear foreign key relationships
- **Constraints**: Database-level data integrity constraints

### 3. API Design
- **RESTful Principles**: Follow REST conventions
- **Versioning**: URL-based API versioning (/api/v1/)
- **Documentation**: Auto-generated OpenAPI documentation
- **Error Responses**: Consistent error response format