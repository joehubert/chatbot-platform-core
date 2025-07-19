Based on my review of the project structure, here are some of the key software architecture concepts it employs:


   * Layered Architecture: The code is organized into distinct layers with separate responsibilities:
       * api: Handles incoming HTTP requests, routing, and data validation (the presentation layer).
       * services: Contains the core business logic, orchestrating tasks and interacting with other components.
       * models: Defines the database schema using an ORM (the data access layer).
       * schemas: Defines the data transfer objects (DTOs) for API requests and responses.
       * core: Manages application-wide concerns like configuration, database sessions, and security.


   * Dependency Injection: The presence of app/api/dependencies.py suggests the use of a dependency injection framework (common in FastAPI) to decouple components, making the
     application more modular and easier to test.


   * Service-Oriented Architecture (SOA) / Microservices Integration: The extensive services directory with clients for various external services (openai_client.py, anthropic_client.py,
     huggingface_client.py) indicates that the application acts as a central hub or orchestrator that communicates with multiple specialized services.

   * API Versioning: The API is explicitly versioned under /api/v1/, which is a best practice for managing API evolution without breaking client integrations.


   * Object-Relational Mapping (ORM): The app/models directory, along with alembic.ini, points to the use of an ORM (like SQLAlchemy) to abstract database interactions, allowing
     developers to work with Python objects instead of raw SQL.


   * Database Migration Management: The alembic/ directory shows that the project uses Alembic to manage and version database schema changes systematically.

   * Containerization: The Dockerfile and docker-compose.yml files indicate that the application is designed to be built and run as a container, ensuring consistent deployment
     environments.


   * Caching: The existence of app/services/cache.py and app/core/redis.py points to a caching layer, likely using Redis, to improve performance by storing frequently accessed data in
     memory.


   * Vector Database Integration: The files app/services/vector_db.py and vector_adapters/pinecone_adapter.py show that the architecture includes a vector database, a key component for
     modern AI applications involving semantic search or Retrieval-Augmented Generation (RAG).

> quit

# Software Architecture Guide - Chatbot Platform Core

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Dependency Injection Architecture](#3-dependency-injection-architecture)
4. [Data Models & Database Schema](#4-data-models--database-schema)
5. [Database Migration Strategy (Alembic)](#5-database-migration-strategy-alembic)
6. [API Design & Architecture](#6-api-design--architecture)
7. [Business Logic Architecture](#7-business-logic-architecture)
8. [External Service Integration](#8-external-service-integration)
9. [Security Architecture](#9-security-architecture)
10. [Testing Strategy](#10-testing-strategy)
11. [Performance & Scalability](#11-performance--scalability)
12. [Monitoring & Observability](#12-monitoring--observability)
13. [Deployment & DevOps](#13-deployment--devops)
14. [SME-Specific Design Decisions](#14-sme-specific-design-decisions)
15. [Future Architecture Considerations](#15-future-architecture-considerations)

---

## 1. Executive Summary

### 1.1 Project Overview & Target Market

The Chatbot Platform Core is an enterprise-grade, open-source chatbot solution specifically designed for small and medium-sized enterprises (SMEs), including local businesses, schools, clubs, and community organizations. The platform addresses the unique constraints of organizations with limited technical resources and constrained budgets (typically $10K-$50K annual technology spending).

### 1.2 Architecture Philosophy

The architecture follows these core principles:

- **Single-Tenant by Design**: Each deployment serves one organization, ensuring data isolation and simplicity
- **SME-First Approach**: Optimized for ease of deployment, cost-effectiveness, and minimal technical overhead
- **Modular Extensibility**: Pluggable components allow customization without core changes
- **Production-Ready**: Enterprise-grade reliability with comprehensive monitoring and error handling

### 1.3 Key Technology Decisions

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Backend Framework** | FastAPI | Async support, automatic API docs, excellent typing, dependency injection |
| **Database** | PostgreSQL 15+ | ACID compliance, JSON support, vector extensions (pgvector) |
| **ORM** | SQLAlchemy | Mature ecosystem, async support, migration tools |
| **Cache/Queue** | Redis 7+ | High performance, persistence options, pub/sub capabilities |
| **Migrations** | Alembic | Industry standard, rollback support, team collaboration |
| **Containerization** | Docker | Consistent deployments, isolation, resource management |

### 1.4 Document Purpose & Audience

This document serves software architects, senior developers, and technical decision-makers who need to understand the platform's technical foundation, design decisions, and implementation patterns for:

- **System Integration**: Understanding how to extend or integrate with the platform
- **Deployment Planning**: Making informed infrastructure and scaling decisions
- **Development Guidelines**: Maintaining consistency with established patterns
- **Technical Assessment**: Evaluating the platform's suitability for specific use cases

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           Client Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  Web Chat Widget  │  Mobile Apps  │  API Integrations          │
└─────────────────────┬───────────────┬───────────────────────────┘
                      │               │
                      v               v
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                             │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application Server                                    │
│  ├── Authentication & Authorization                            │
│  ├── Rate Limiting & Security                                  │
│  ├── Request Validation & Serialization                        │
│  └── API Versioning (/api/v1/)                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  LangGraph Pipeline Engine                                     │
│  ├── Query Analysis & Classification                           │
│  ├── Relevance Filtering                                       │
│  ├── Model Routing & Selection                                 │
│  ├── RAG Context Retrieval                                     │
│  ├── Response Generation                                       │
│  └── Response Validation                                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────────┐
│                     Service Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   LLM Services  │  │  Vector DB      │  │  Cache Services │  │
│  │  ├── OpenAI     │  │  ├── Pinecone   │  │  ├── Semantic   │  │
│  │  ├── Anthropic  │  │  ├── Chroma     │  │  ├── Session    │  │
│  │  ├── Local/     │  │  └── pgvector   │  │  └── Response   │  │
│  │  │   Ollama     │  │                 │  │                 │  │
│  │  └── Azure/AWS  │  └─────────────────┘  └─────────────────┘  │
│  └─────────────────┘                                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   PostgreSQL    │  │     Redis       │  │  File Storage   │  │
│  │  ├── Users      │  │  ├── Sessions   │  │  ├── Documents  │  │
│  │  ├── Conversations│ │  ├── Cache     │  │  ├── Uploads    │  │
│  │  ├── Messages   │  │  ├── Rate Limits│  │  └── Temp Files │  │
│  │  ├── Documents  │  │  └── Pub/Sub    │  │                 │  │
│  │  └── Analytics  │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack Rationale

**Backend Framework**: FastAPI was chosen for its automatic API documentation, excellent async support, built-in dependency injection, and strong typing system that reduces runtime errors.

**Database**: PostgreSQL provides ACID compliance, excellent performance for both relational and JSON data, and native vector support through pgvector extension.

**ORM**: SQLAlchemy offers mature async support, comprehensive migration tools through Alembic, and excellent performance optimization capabilities.

**Caching**: Redis provides both traditional caching and advanced features like pub/sub for real-time notifications and session management.

### 2.3 Core Components & Responsibilities

| Component | Responsibility | Location |
|-----------|---------------|----------|
| **API Layer** | Request handling, validation, authentication | `app/api/` |
| **Business Logic** | LangGraph pipeline, chat processing | `app/core/` |
| **Data Models** | Database schema, relationships | `app/models/` |
| **Services** | External integrations, business services | `app/services/` |
| **Schemas** | Request/response validation | `app/schemas/` |
| **Utilities** | Helpers, dependencies, logging | `app/utils/` |

### 2.4 Data Flow Architecture

```
User Message → API Validation → Rate Limiting → Pipeline Processing → Response
     ↓              ↓              ↓              ↓                   ↓
  Session        Request         Redis         LangGraph           Database
 Management     Validation      Check         Pipeline            Storage
     ↓              ↓              ↓              ↓                   ↓
Authentication  Schema           Global       Query Analysis      Conversation
  Check         Validation       Limits       ↓                   History
     ↓              ↓              ↓         RAG Retrieval            ↓
   OTP            Error           Per-User   ↓                   Analytics
Verification     Handling        Limits     LLM Generation       Tracking
                                  ↓              ↓                   ↓
                               Continue      Response              Monitoring
                               Pipeline      Validation            & Metrics
```

### 2.5 Deployment Architecture (Docker Compose)

The platform uses a multi-service Docker Compose architecture:

```yaml
services:
  app:          # FastAPI application server
  postgres:     # Primary database
  redis:        # Cache and session store
  nginx:        # Reverse proxy (production)
  vector-db:    # Vector database (if not using cloud)
```

---

## 3. Dependency Injection Architecture

### 3.1 Why Dependency Injection is Used

The platform extensively uses FastAPI's dependency injection system for several critical architectural benefits:

#### Testability and Mock Integration
Dependency injection enables comprehensive testing by allowing easy substitution of real services with mocks:

```python
# Production code uses real database
@app.get("/conversations")
async def get_conversations(db: Session = Depends(get_db)):
    return conversation_service.get_all(db)

# Test code injects mock database
app.dependency_overrides[get_db] = lambda: mock_db_session
```

#### Service Decoupling & Modularity
Services are loosely coupled, making the system more maintainable and allowing independent development:

- LLM providers can be swapped without changing API code
- Vector databases can be changed through configuration
- Cache implementations can be replaced transparently

#### Configuration Management
Environment-specific configurations are injected at runtime:

```python
async def get_semantic_cache_service() -> SemanticCacheService:
    settings = get_settings()
    return SemanticCacheService(
        redis_url=settings.REDIS_URL,
        ttl_hours=settings.CACHE_TTL_HOURS,
        similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD
    )
```

#### Database Session Management
Proper session lifecycle management prevents memory leaks and ensures transactional integrity:

```python
def get_db() -> Session:
    """Database session with automatic cleanup"""
    db = get_sync_db()
    try:
        yield db
    finally:
        db.close()
```

### 3.2 FastAPI Dependency System Implementation

The dependency system is centralized in `app/api/dependencies.py` and provides these key dependencies:

#### Core Dependencies (`app/api/dependencies.py`)

```python
# Database access
def get_db() -> Session:
    """Provides database session with automatic cleanup"""

# Redis access  
def get_redis() -> redis.Redis:
    """Provides Redis client for caching and sessions"""

# Settings injection
def get_settings() -> Settings:
    """Provides application configuration"""
```

#### Database Session Injection

The database dependency ensures proper session management:

- **Automatic Creation**: New session for each request
- **Transaction Management**: Automatic commit/rollback
- **Resource Cleanup**: Sessions are always closed
- **Connection Pooling**: Efficient database connection reuse

```python
# Usage in API endpoints
@router.post("/message")
async def send_message(
    request: ChatMessage,
    db: Session = Depends(get_db)  # Automatic session injection
):
    # Database operations use the injected session
    conversation = await conversation_service.get_or_create(db, session_id)
```

#### Authentication Dependencies

Authentication is handled through dependency chains:

```python
async def get_current_user(
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[User]:
    """Extract and validate user from bearer token"""
    
# Usage creates dependency chain
@router.get("/profile")
async def get_profile(user: User = Depends(get_current_user)):
    return user.profile
```

#### Rate Limiting Dependencies

Rate limiting is implemented as a dependency that can be applied to any endpoint:

```python
async def check_rate_limit(
    request: Request,
    db: Session = Depends(get_db),
    redis: redis.Redis = Depends(get_redis)
) -> bool:
    """Check if request exceeds rate limits"""

# Applied to endpoints that need rate limiting
@router.post("/message")
async def send_message(
    _: bool = Depends(check_rate_limit)  # Rate limit check
):
```

### 3.3 Service Layer Dependency Pattern

Services are injected using factory functions that handle initialization and configuration:

#### LLM Service Factory Pattern

```python
async def get_model_factory() -> ModelFactory:
    """Dependency injection for model factory"""
    settings = get_settings()
    
    model_factory = ModelFactory(
        default_provider=settings.DEFAULT_LLM_PROVIDER,
        model_configs=settings.MODEL_CONFIGS
    )
    
    if not model_factory._initialized:
        await model_factory.initialize()
    
    return model_factory
```

This pattern provides:
- **Lazy Initialization**: Services are created only when needed
- **Configuration Injection**: Settings are provided from environment
- **Singleton Behavior**: Services are reused across requests
- **Async Support**: Proper async initialization for I/O operations

#### Vector Database Abstraction

```python
async def get_vector_db_service() -> VectorDBService:
    """Factory function for vector database service"""
    settings = get_settings()
    
    return create_vector_db_service(
        db_type=settings.VECTOR_DB_TYPE,  # pinecone|chroma|pgvector
        connection_params={
            "api_key": settings.VECTOR_DB_API_KEY,
            "environment": settings.VECTOR_DB_ENVIRONMENT,
        },
        index_name=settings.VECTOR_DB_INDEX_NAME
    )
```

#### Cache Service Dependencies

Multiple cache services are injected based on use case:

```python
# Semantic similarity caching
async def get_semantic_cache_service() -> SemanticCacheService:
    """For caching similar queries and responses"""

# Session data caching  
async def get_conversation_cache_service() -> ConversationCacheService:
    """For temporary conversation state"""
```

### 3.4 Testing with Dependency Overrides

The dependency system enables comprehensive testing through overrides:

```python
# Test configuration (conftest.py)
@pytest.fixture
def client(db_session: AsyncSession):
    """Test client with mocked dependencies"""
    
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()
```

**Benefits for Testing**:
- **Isolated Tests**: Each test gets fresh mocked dependencies
- **Predictable State**: Test databases are created/destroyed per test
- **Service Mocking**: External services (LLMs, email) can be mocked
- **Performance**: Tests run against in-memory/local services

---

## 4. Data Models & Database Schema

### 4.1 Core Entity Definitions

The platform uses a carefully designed database schema optimized for conversational AI workloads and analytics.

#### User Model (`app/models/user.py`)

Manages user identity and authentication preferences:

```python
class User(Base):
    __tablename__ = "users"
    
    # Identity
    id: UUID = Column(UUID(as_uuid=True), primary_key=True)
    mobile_number: Optional[str] = Column(String(20), index=True)
    email: Optional[str] = Column(String(255), index=True)
    
    # Lifecycle tracking
    created_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_authenticated: Optional[datetime] = Column(DateTime(timezone=True))
    last_seen: Optional[datetime] = Column(DateTime(timezone=True))
    authentication_count: int = Column(Integer, default=0)
    
    # Configuration
    is_active: bool = Column(Boolean, default=True)
    preferred_contact_method: Optional[str] = Column(String(10))  # 'sms'|'email'
    timezone: Optional[str] = Column(String(50))
    language_preference: Optional[str] = Column(String(10))
```

**Design Decisions**:
- **UUID Primary Keys**: Prevents enumeration attacks and allows distributed generation
- **Optional Contact Methods**: Users can authenticate via SMS or email
- **Timezone Support**: Critical for scheduling and proper timestamp display
- **Activity Tracking**: Enables user engagement analytics

#### Conversation Model (`app/models/conversation.py`)

Represents chat sessions with comprehensive metadata:

```python
class Conversation(Base):
    __tablename__ = "conversations"
    
    # Identity and session
    id: UUID = Column(UUID(as_uuid=True), primary_key=True)
    session_id: str = Column(String(255), unique=True, index=True)
    user_identifier: Optional[str] = Column(String(255), index=True)
    user_id: Optional[UUID] = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Lifecycle
    started_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    ended_at: Optional[datetime] = Column(DateTime(timezone=True))
    last_activity: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Status and resolution
    resolved: bool = Column(Boolean, default=False)
    resolution_attempts: int = Column(Integer, default=0)
    authenticated: bool = Column(Boolean, default=False)
    
    # Relationships
    messages: List["Message"] = relationship("Message", back_populates="conversation", 
                                           cascade="all, delete-orphan")
    user: Optional["User"] = relationship("User", lazy="select")
```

**Key Features**:
- **Session Management**: Links anonymous sessions to authenticated users
- **Resolution Tracking**: Monitors conversation completion and quality
- **Cascade Deletion**: Maintains referential integrity
- **Performance Indexes**: Optimized for common query patterns

#### Message Model (`app/models/message.py`)

Captures individual messages with rich metadata:

```python
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(Base):
    __tablename__ = "messages"
    
    # Identity and relationships
    id: UUID = Column(UUID(as_uuid=True), primary_key=True)
    conversation_id: UUID = Column(UUID(as_uuid=True), 
                                  ForeignKey("conversations.id", ondelete="CASCADE"))
    
    # Content
    content: str = Column(Text, nullable=False)
    role: MessageRole = Column(SQLEnum(MessageRole), nullable=False, index=True)
    timestamp: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Processing metadata
    model_used: Optional[str] = Column(String(100))  # For assistant messages
    cached: bool = Column(Boolean, default=False)
    processing_time_ms: Optional[int] = Column(Integer)
    tokens_used: Optional[int] = Column(Integer)
    confidence_score: Optional[float] = Column(Float)
    requires_clarification: bool = Column(Boolean, default=False)
    
    # Additional data
    data: Optional[dict] = Column(JSON)  # Flexible metadata storage
```

**Analytics Features**:
- **Performance Tracking**: Response times and token usage
- **Quality Metrics**: Confidence scores and clarification needs
- **Cost Management**: Token usage tracking for billing
- **Caching Analytics**: Cache hit/miss rates

#### Document Model (`app/models/document.py`)

Manages knowledge base content with versioning:

```python
class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing" 
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"

class Document(Base):
    __tablename__ = "documents"
    
    # Identity
    id: UUID = Column(UUID(as_uuid=True), primary_key=True)
    filename: str = Column(String(255), nullable=False)
    content_type: str = Column(String(100), nullable=False)
    content_hash: str = Column(String(64), unique=True)  # SHA-256
    
    # Processing status
    status: DocumentStatus = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED)
    uploaded_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    processed_at: Optional[datetime] = Column(DateTime(timezone=True))
    expires_at: Optional[datetime] = Column(DateTime(timezone=True))
    
    # Content metadata
    file_size: int = Column(Integer, nullable=False)
    chunk_count: Optional[int] = Column(Integer)
    embedding_model: Optional[str] = Column(String(100))
    
    # Processing results
    processing_error: Optional[str] = Column(Text)
    metadata: Optional[dict] = Column(JSON)
```

**Content Management Features**:
- **Deduplication**: Content hash prevents duplicate processing
- **Status Tracking**: Complete processing lifecycle monitoring
- **Expiration Management**: Automatic cleanup of temporary content
- **Error Handling**: Detailed error tracking for debugging

#### AuthToken Model (`app/models/auth_token.py`)

Manages OTP tokens for authentication:

```python
class AuthToken(Base):
    __tablename__ = "auth_tokens"
    
    # Identity
    id: UUID = Column(UUID(as_uuid=True), primary_key=True)
    user_id: UUID = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    session_id: str = Column(String(255), nullable=False)
    
    # Token data
    token: str = Column(String(10), unique=True, nullable=False)  # 6-digit OTP
    created_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at: datetime = Column(DateTime(timezone=True), nullable=False)
    used_at: Optional[datetime] = Column(DateTime(timezone=True))
    used: bool = Column(Boolean, default=False)
    
    # Delivery information
    delivery_method: str = Column(String(10), nullable=False)  # 'sms'|'email'
    delivery_address: str = Column(String(255), nullable=False)
    
    # Security
    attempts: int = Column(Integer, default=0)
    max_attempts: int = Column(Integer, default=3)
```

**Security Features**:
- **Limited Attempts**: Prevents brute force attacks
- **Expiration Management**: Time-limited token validity
- **Usage Tracking**: Single-use tokens with audit trail
- **Multi-Channel Support**: SMS and email delivery options

### 4.2 Entity Relationships & Foreign Keys

The schema implements a hierarchical relationship structure:

```
User (1) ←→ (0..n) Conversation ←→ (1..n) Message
  ↓
  └─→ (0..n) AuthToken

Document (independent entity for knowledge base)
```

**Relationship Characteristics**:
- **Users → Conversations**: One-to-many with optional relationship (supports anonymous users)
- **Conversations → Messages**: One-to-many with cascade delete
- **Users → AuthTokens**: One-to-many with cascade delete
- **Documents**: Independent entities for knowledge base content

### 4.3 Database Design Patterns

#### UUID vs Integer Primary Keys

**Decision**: UUIDs are used for all primary keys

**Rationale**:
- **Security**: Prevents enumeration attacks
- **Distribution**: Allows ID generation without database coordination  
- **Integration**: Easier integration with external systems
- **Analytics**: Prevents user counting/tracking

#### Soft Delete vs Hard Delete Strategy

**Decision**: Hard delete with cascade for most entities

**Rationale for SME Context**:
- **Storage Efficiency**: SMEs have limited storage budgets
- **Compliance Simplicity**: GDPR compliance through actual deletion
- **Performance**: No need to filter soft-deleted records
- **Analytics Retention**: Important analytics copied to separate tables

#### JSON Metadata Storage Pattern

Strategic use of JSON columns for flexible data:

```python
# Message metadata
data: Optional[dict] = Column(JSON)  # Flexible per-message data

# Document metadata  
metadata: Optional[dict] = Column(JSON)  # Processing results, source info
```

**Use Cases**:
- **Extension Points**: Adding fields without schema changes
- **Integration Data**: Storing third-party service responses
- **Processing Results**: Complex processing outputs
- **Experiment Data**: A/B testing and feature flags

#### Indexing Strategy for Performance

Composite indexes optimize common query patterns:

```python
# Conversation queries
Index("idx_conversation_session_started", "session_id", "started_at")
Index("idx_conversation_user_identifier", "user_identifier")

# Message queries  
Index("idx_message_conversation_timestamp", "conversation_id", "timestamp")
Index("idx_message_role_timestamp", "role", "timestamp")

# Analytics queries
Index("idx_message_cached", "cached")
Index("idx_message_model_used", "model_used")
```

### 4.4 SQLAlchemy ORM Implementation

#### Model Base Classes

All models inherit from a common base:

```python
# app/models/base.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """Base class for all database models"""
    pass
```

#### Relationship Configurations

Relationships are configured for optimal loading:

```python
# Lazy loading for large collections
messages: List["Message"] = relationship("Message", 
                                       back_populates="conversation",
                                       lazy="select",  # Load when accessed
                                       cascade="all, delete-orphan")

# Eager loading for frequently accessed data
user: Optional["User"] = relationship("User", lazy="select")
```

#### Query Optimization Patterns

Models include helper methods for common queries:

```python
class Conversation(Base):
    @classmethod
    def get_active_conversations(cls, db: Session) -> Query:
        """Get conversations that haven't ended"""
        return db.query(cls).filter(cls.ended_at.is_(None))
    
    @classmethod  
    def get_by_session(cls, db: Session, session_id: str) -> Optional['Conversation']:
        """Find conversation by session ID"""
        return db.query(cls).filter(cls.session_id == session_id).first()
```

---

## 5. Database Migration Strategy (Alembic)

### 5.1 Why Alembic is Used

Alembic provides essential database schema management capabilities for enterprise deployment:

#### Schema Evolution Management
- **Version Control**: Database schema changes are tracked in source control
- **Team Collaboration**: Multiple developers can work on schema changes safely
- **Environment Consistency**: Development, staging, and production schemas stay synchronized
- **Change Documentation**: Each migration documents what changed and why

#### Production Deployment Safety
- **Rollback Capability**: Every migration can be reversed safely
- **Incremental Updates**: Changes are applied in small, tested increments
- **Zero-Downtime Migrations**: Careful migration design enables online schema changes
- **Validation**: Migrations can be tested in staging before production

#### SME-Specific Benefits
- **Simplified Deployment**: Database updates are automated and repeatable
- **Reduced Risk**: Migrations are tested and documented before deployment
- **Audit Trail**: Complete history of all database changes
- **Recovery Planning**: Clear rollback procedures for any migration

### 5.2 Migration Architecture

#### Alembic Configuration (`alembic.ini`)

The configuration file manages migration behavior and database connections:

```ini
[alembic]
script_location = alembic
file_template = %%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s
prepend_sys_path = .

# Database URL - overridden by environment variable  
sqlalchemy.url = postgresql://user:password@localhost/chatbot_platform

[loggers]
keys = root,sqlalchemy,alembic

[handlers]  
keys = console

[formatters]
keys = generic
```

**Key Features**:
- **Timestamped Files**: Migration files include creation timestamps
- **Environment Override**: Production URLs come from environment variables
- **Comprehensive Logging**: All migration activity is logged
- **Path Management**: Ensures proper module imports

#### Environment Setup (`alembic/env.py`)

The environment file configures migration context and model imports:

```python
# Import all models to ensure metadata registration
from app.models import Conversation, Message, Document, User, AuthToken
from app.models.base import Base

# Set target metadata for autogenerate
target_metadata = Base.metadata

def get_database_url() -> str:
    """Get database URL with environment variable override"""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # Fallback to config file
    return config.get_main_option("sqlalchemy.url")

def run_migrations_online() -> None:
    """Run migrations in 'online' mode with live database connection"""
    database_url = get_database_url()
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = database_url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=include_name,
            compare_type=compare_type,
            render_as_batch=True
        )
        
        with context.begin_transaction():
            context.run_migrations()
```

**Configuration Features**:
- **Model Discovery**: Automatically imports all models for metadata
- **Environment Flexibility**: Works in development and production
- **Type Comparison**: Custom logic for PostgreSQL-specific types
- **Connection Management**: Proper connection lifecycle management

#### Migration File Structure

Migrations follow a consistent naming and structure pattern:

```
alembic/versions/
├── 001_initial_database_schema.py    # Initial schema creation
├── 002_add_user_preferences.py       # Add user preference fields  
├── 003_add_analytics_indexes.py      # Performance optimization
└── 004_add_document_metadata.