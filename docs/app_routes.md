# Complete API Route Paths for Chatbot Platform

Based on the project files and API structure, here are all the route paths supported by the chatbot platform API:

## Root Application Routes

```
GET    /                           # Root endpoint (platform info)
GET    /health                     # Basic health check
GET    /health/detailed            # Detailed health check with dependencies
```

## API v1 Routes

### Chat API (`/api/v1/chat`)

```
POST   /api/v1/chat/message        # Send chat message through pipeline
GET    /api/v1/chat/conversations  # Get conversation history (paginated)
GET    /api/v1/chat/conversations/{conversation_id}  # Get specific conversation
POST   /api/v1/chat/feedback       # Submit feedback on responses
```

### Authentication API (`/api/v1/auth`)

```
GET    /api/v1/auth/health         # Auth service health check
POST   /api/v1/auth/request        # Request authentication token (OTP)
POST   /api/v1/auth/verify         # Verify authentication token
GET    /api/v1/auth/session/{session_id}/status  # Get session authentication status
```

### Knowledge Base API (`/api/v1/knowledge`)

```
POST   /api/v1/knowledge/documents              # Upload document
GET    /api/v1/knowledge/documents              # List documents (with pagination/filtering)
GET    /api/v1/knowledge/documents/{document_id}  # Get specific document details
PUT    /api/v1/knowledge/documents/{document_id}  # Update document metadata
DELETE /api/v1/knowledge/documents/{document_id}  # Delete document
GET    /api/v1/knowledge/search                 # Search knowledge base
GET    /api/v1/knowledge/stats                  # Get knowledge base statistics
POST   /api/v1/knowledge/rebuild-index          # Rebuild vector index (admin)
DELETE /api/v1/knowledge/expired                # Cleanup expired documents
```

### Configuration API (`/api/v1/config`)

```
GET    /api/v1/config              # Get current system configuration
PUT    /api/v1/config              # Update system configuration (admin)
GET    /api/v1/config/validate     # Validate configuration
POST   /api/v1/config/reload       # Reload configuration
GET    /api/v1/config/llm          # Get LLM provider configurations
PUT    /api/v1/config/llm          # Update LLM provider settings
GET    /api/v1/config/vector-db    # Get vector database configuration
PUT    /api/v1/config/vector-db    # Update vector database settings
GET    /api/v1/config/cache        # Get cache configuration
PUT    /api/v1/config/cache        # Update cache settings
POST   /api/v1/config/cache/clear  # Clear cache (admin)
GET    /api/v1/config/health       # Get system health status
```

### Analytics API (`/api/v1/analytics`)

```
GET    /api/v1/analytics/conversations     # Get conversation metrics
GET    /api/v1/analytics/performance       # Get performance metrics
GET    /api/v1/analytics/models            # Get model usage analytics
GET    /api/v1/analytics/cache             # Get cache performance metrics
GET    /api/v1/analytics/errors            # Get error analysis
GET    /api/v1/analytics/costs             # Get cost analysis
GET    /api/v1/analytics/users/engagement  # Get user engagement analysis
GET    /api/v1/analytics/knowledge-base/usage  # Get knowledge base analytics
```

### Health Check API (`/api/v1/health`)

```
GET    /api/v1/health              # Service health status
GET    /api/v1/health/detailed     # Detailed health with all dependencies
GET    /api/v1/health/database     # Database connectivity check
GET    /api/v1/health/redis        # Redis connectivity check
GET    /api/v1/health/vector-db    # Vector database connectivity check
GET    /api/v1/health/llm-providers  # LLM provider availability check
```

## Nested Authentication Routes

The auth endpoints are also accessible through chat for integrated authentication:

```
POST   /api/v1/chat/auth/request   # Request auth token (nested under chat)
POST   /api/v1/chat/auth/verify    # Verify auth token (nested under chat)
```

## Query Parameters Summary

### Common Parameters
- `skip` / `offset`: Pagination offset
- `limit`: Number of items to return
- `start_date` / `end_date`: Date range filtering

### Chat Specific
- `user_id`: Filter by specific user
- `session_id`: Filter by session

### Knowledge Base Specific
- `category`: Filter documents by category
- `status_filter`: Filter by processing status
- `similarity_threshold`: Search similarity threshold
- `query`: Search query string

### Analytics Specific
- `metric`: Specific metric to retrieve
- `granularity`: Time granularity (hour, day, week, month)
- `groupby`: Group results by field

## WebSocket Routes (if implemented)

```
WS     /ws                         # WebSocket connection for real-time chat
WS     /ws/chat/{session_id}       # Session-specific WebSocket
```

## Admin-Only Routes

The following routes require admin privileges:
- `PUT /api/v1/config/*` (all config updates)
- `POST /api/v1/config/cache/clear`
- `POST /api/v1/knowledge/rebuild-index`
- Most analytics endpoints
- System health endpoints

## Total Route Count: ~45 Routes

This comprehensive API provides full functionality for:
- **Chat Operations**: Message processing, conversation management
- **Authentication**: OTP-based auth with session management
- **Knowledge Management**: Document upload, processing, search
- **System Configuration**: LLM providers, cache, vector DB settings
- **Analytics & Monitoring**: Performance metrics, usage tracking
- **Health Checks**: System status and dependency monitoring

All routes follow RESTful conventions and include proper error handling, validation, and authentication where required.