# Chatbot Platform API - OpenAPI Specification v3.0.3

## Overview

This OpenAPI specification defines the REST API for the Enterprise Chatbot Platform Core. The platform provides intelligent conversation management, RAG (Retrieval-Augmented Generation) capabilities, multi-LLM support, and comprehensive analytics for enterprise deployments.

**Base URL**: `https://your-domain.com/api`  
**API Version**: v1  
**Authentication**: Bearer Token (JWT) with OTP verification  
**Content Type**: `application/json`

## API Specification

```yaml
openapi: 3.0.3
info:
  title: Enterprise Chatbot Platform API
  description: |
    Enterprise-grade chatbot platform with intelligent processing pipeline, RAG capabilities, 
    multi-LLM support, and comprehensive analytics. This API enables organizations to integrate 
    AI-powered conversations into their applications with advanced features like semantic caching, 
    intelligent routing, and real-time monitoring.

    ## Key Features
    - **Intelligent Processing Pipeline**: LangGraph-based message processing with rate limiting, 
      relevance checking, semantic caching, and intelligent model routing
    - **RAG Knowledge Base**: Document upload, processing, chunking, vector storage with expiration management
    - **Multi-LLM Support**: Pluggable architecture supporting OpenAI, Anthropic, HuggingFace, and local models
    - **OTP Authentication**: SMS/Email based authentication with secure session management
    - **Comprehensive Analytics**: Conversation tracking, performance metrics, cost analysis
    - **Semantic Caching**: Vector-based response caching for improved performance and cost optimization

    ## Authentication Flow
    1. Request OTP via SMS or email using `/auth/request-token`
    2. Verify OTP and receive session token using `/auth/verify-token`  
    3. Include session token in Authorization header for authenticated requests
    4. Optionally check authentication status using `/auth/status`

    ## Rate Limiting
    The API implements intelligent rate limiting with different tiers:
    - **Guest Users**: 10 requests per minute
    - **Authenticated Users**: 60 requests per minute
    - **Premium Users**: 300 requests per minute
    
    Rate limit headers are included in responses:
    - `X-RateLimit-Limit`: Maximum requests allowed
    - `X-RateLimit-Remaining`: Remaining requests in current window
    - `X-RateLimit-Reset`: UTC timestamp when limit resets

  version: "1.0.0"
  contact:
    name: Chatbot Platform Support
    email: support@chatbot-platform.com
    url: https://chatbot-platform.com/support
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT
  termsOfService: https://chatbot-platform.com/terms

servers:
  - url: https://api.chatbot-platform.com/api
    description: Production server
  - url: https://staging-api.chatbot-platform.com/api
    description: Staging server
  - url: http://localhost:8000/api
    description: Local development server

security:
  - BearerAuth: []

paths:
  # ==========================================
  # CHAT ENDPOINTS
  # ==========================================
  
  /v1/chat/message:
    post:
      tags:
        - Chat
      summary: Send chat message
      description: |
        Process a chat message through the complete intelligent pipeline including:
        
        1. **Rate Limiting**: Token bucket algorithm prevents abuse
        2. **Relevance Checking**: LLM-based classification filters irrelevant queries
        3. **Semantic Caching**: Vector similarity search for cached responses
        4. **Authentication**: Conditional authentication for sensitive topics
        5. **Model Routing**: Intelligent routing based on query complexity
        6. **RAG Processing**: Knowledge base search and context injection
        7. **Response Generation**: Multi-LLM response generation with fallbacks
        8. **Response Validation**: Quality checks and safety filtering
        
        The endpoint automatically handles conversation context, maintains message history,
        and provides detailed metadata about the processing pipeline performance.
      operationId: sendMessage
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatMessage'
            examples:
              simple_query:
                summary: Simple customer support query
                value:
                  session_id: "session_12345"
                  message: "What are your business hours?"
                  conversation_id: null
                  user_context:
                    timezone: "America/New_York"
                    user_agent: "Mozilla/5.0..."
              complex_query:
                summary: Complex technical query requiring authentication
                value:
                  session_id: "session_67890"
                  message: "Can you help me reset my account password and explain the security implications?"
                  conversation_id: "conv_123e4567-e89b-12d3-a456-426614174000"
                  user_context:
                    timezone: "UTC"
                    device_type: "mobile"
      responses:
        '200':
          description: Message processed successfully
          headers:
            X-RateLimit-Limit:
              description: Maximum requests allowed per time window
              schema:
                type: integer
            X-RateLimit-Remaining:
              description: Remaining requests in current window
              schema:
                type: integer
            X-RateLimit-Reset:
              description: UTC timestamp when rate limit resets
              schema:
                type: integer
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatResponse'
              examples:
                cached_response:
                  summary: Response served from semantic cache
                  value:
                    message_id: "msg_123e4567-e89b-12d3-a456-426614174000"
                    response: "Our business hours are Monday-Friday 9 AM to 6 PM EST."
                    conversation_id: "conv_123e4567-e89b-12d3-a456-426614174000"
                    session_id: "session_12345"
                    cached: true
                    processing_time_ms: 45
                    pipeline_steps:
                      - step: "rate_limit"
                        duration_ms: 5
                        status: "passed"
                      - step: "semantic_cache"
                        duration_ms: 35
                        status: "hit"
                        cache_similarity: 0.95
                    suggested_actions: []
                    requires_authentication: false
                    timestamp: "2024-01-15T10:30:00Z"
                authentication_required:
                  summary: Authentication required for sensitive query
                  value:
                    message_id: "msg_987f6543-a21b-43c5-d678-901234567890"
                    response: "I'd be happy to help with your account security. For your protection, please verify your identity first."
                    conversation_id: "conv_987f6543-a21b-43c5-d678-901234567890"
                    session_id: "session_67890"
                    cached: false
                    processing_time_ms: 1250
                    pipeline_steps:
                      - step: "rate_limit"
                        duration_ms: 5
                        status: "passed"
                      - step: "relevance_check"
                        duration_ms: 800
                        status: "passed"
                        relevance_score: 0.92
                      - step: "auth_check"
                        duration_ms: 10
                        status: "required"
                    suggested_actions:
                      - action: "authenticate"
                        description: "Complete OTP verification to continue"
                        endpoint: "/api/v1/auth/request-token"
                    requires_authentication: true
                    auth_methods: ["sms", "email"]
                    timestamp: "2024-01-15T10:30:00Z"
        '400':
          description: Invalid request format or missing required fields
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: "validation_error"
                message: "Message content cannot be empty"
                details:
                  field: "message"
                  constraint: "min_length_1"
                request_id: "req_12345"
                timestamp: "2024-01-15T10:30:00Z"
        '429':
          description: Rate limit exceeded
          headers:
            Retry-After:
              description: Seconds to wait before retrying
              schema:
                type: integer
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: "rate_limit_exceeded"
                message: "Too many requests. Please wait before sending another message."
                details:
                  limit: 10
                  window_seconds: 60
                  retry_after: 45
                request_id: "req_67890"
                timestamp: "2024-01-15T10:30:00Z"
        '500':
          description: Internal server error during message processing
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/chat/conversations/{conversation_id}/history:
    get:
      tags:
        - Chat
      summary: Get conversation history
      description: |
        Retrieve the complete message history for a specific conversation including:
        
        - All messages in chronological order
        - Message metadata (timestamps, processing info, etc.)
        - Conversation context and user information
        - Pipeline processing details for each message
        - Authentication status changes throughout conversation
        
        Results are paginated and can be filtered by date range or message type.
        This endpoint is useful for conversation analysis, debugging, and user support.
      operationId: getConversationHistory
      parameters:
        - name: conversation_id
          in: path
          required: true
          description: Unique identifier for the conversation
          schema:
            type: string
            format: uuid
          example: "123e4567-e89b-12d3-a456-426614174000"
        - name: limit
          in: query
          description: Maximum number of messages to return (default 50, max 200)
          schema:
            type: integer
            minimum: 1
            maximum: 200
            default: 50
        - name: offset
          in: query
          description: Number of messages to skip for pagination (default 0)
          schema:
            type: integer
            minimum: 0
            default: 0
        - name: start_date
          in: query
          description: Filter messages after this timestamp (ISO 8601 format)
          schema:
            type: string
            format: date-time
          example: "2024-01-01T00:00:00Z"
        - name: end_date
          in: query
          description: Filter messages before this timestamp (ISO 8601 format)
          schema:
            type: string
            format: date-time
          example: "2024-01-31T23:59:59Z"
        - name: include_system
          in: query
          description: Include system messages in response (default false)
          schema:
            type: boolean
            default: false
      responses:
        '200':
          description: Conversation history retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ConversationHistory'
        '404':
          description: Conversation not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '403':
          description: Access denied to conversation
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/chat/feedback:
    post:
      tags:
        - Chat
      summary: Submit message feedback
      description: |
        Submit feedback for a specific chat response to improve the system's performance.
        Feedback is used for:
        
        - **Model Fine-tuning**: Improve response quality over time
        - **Cache Optimization**: Remove poor responses from semantic cache
        - **Analytics**: Track user satisfaction and identify improvement areas
        - **Knowledge Base**: Flag documents that need updates or removal
        
        Feedback types include thumbs up/down, detailed ratings, and free-form comments.
        This data is anonymized and used solely for system improvement.
      operationId: submitFeedback
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/FeedbackRequest'
      responses:
        '200':
          description: Feedback submitted successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FeedbackResponse'
        '400':
          description: Invalid feedback format
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  # ==========================================
  # AUTHENTICATION ENDPOINTS
  # ==========================================

  /v1/auth/request-token:
    post:
      tags:
        - Authentication
      summary: Request OTP authentication token
      description: |
        Request a one-time password (OTP) for authentication via SMS or email.
        
        **Process Flow:**
        1. Submit contact method (SMS/email) and contact value
        2. System generates and sends OTP to specified contact
        3. OTP expires after configurable time (default 5 minutes)
        4. Rate limiting prevents abuse (max 3 requests per 15 minutes per contact)
        
        **Security Features:**
        - Rate limiting per contact method and IP address
        - OTP expiration and single-use enforcement
        - Encrypted transmission and storage
        - Audit logging for security monitoring
        
        This endpoint does not require authentication but is subject to rate limiting.
      operationId: requestAuthToken
      security: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AuthRequest'
            examples:
              sms_request:
                summary: Request OTP via SMS
                value:
                  session_id: "session_12345"
                  contact_method: "sms"
                  contact_value: "+1234567890"
              email_request:
                summary: Request OTP via email
                value:
                  session_id: "session_67890"
                  contact_method: "email"
                  contact_value: "user@example.com"
      responses:
        '200':
          description: OTP sent successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
              example:
                success: true
                message: "Authentication code sent to your email address"
                expires_in: 300
                retry_after: null
        '400':
          description: Invalid request (malformed contact info, etc.)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '429':
          description: Rate limit exceeded for OTP requests
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: "rate_limit_exceeded"
                message: "Too many OTP requests. Please wait before requesting another code."
                details:
                  retry_after: 900
                request_id: "req_abc123"
                timestamp: "2024-01-15T10:30:00Z"

  /v1/auth/verify-token:
    post:
      tags:
        - Authentication
      summary: Verify OTP and get session token
      description: |
        Verify the OTP received via SMS or email and obtain a session token for authenticated requests.
        
        **Verification Process:**
        1. Submit session ID and received OTP
        2. System validates OTP against stored hash
        3. If valid, generates JWT session token
        4. Session token includes user context and permissions
        5. OTP is immediately invalidated after use
        
        **Session Token Features:**
        - JWT format with configurable expiration (default 1 hour)
        - Includes user ID, session context, and permissions
        - Can be refreshed before expiration
        - Automatically revoked on suspicious activity
        
        Failed verification attempts are rate limited and logged for security monitoring.
      operationId: verifyAuthToken
      security: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AuthVerification'
            example:
              session_id: "session_12345"
              token: "ABC123"
      responses:
        '200':
          description: Token verified successfully, session token issued
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TokenResponse'
              example:
                success: true
                message: "Authentication successful"
                session_token: "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
                expires_at: "2024-01-15T11:30:00Z"
                user_id: "123e4567-e89b-12d3-a456-426614174000"
        '400':
          description: Invalid or expired token
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '429':
          description: Too many verification attempts
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/auth/status:
    get:
      tags:
        - Authentication
      summary: Check authentication status
      description: |
        Check the current authentication status for a session including:
        
        - Authentication state (authenticated/unauthenticated)
        - Session token expiration time
        - Time remaining before token expires
        - Whether token renewal is recommended
        - User profile information if authenticated
        
        This endpoint is useful for:
        - Frontend authentication state management
        - Proactive token renewal before expiration
        - Session validation in multi-tab scenarios
        - Security monitoring and audit trails
      operationId: checkAuthStatus
      parameters:
        - name: session_id
          in: query
          required: true
          description: Session identifier to check
          schema:
            type: string
          example: "session_12345"
      responses:
        '200':
          description: Authentication status retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthStatus'
        '400':
          description: Invalid session ID format
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/auth/refresh:
    post:
      tags:
        - Authentication
      summary: Refresh session token
      description: |
        Refresh an existing session token to extend the authentication period.
        
        **Refresh Requirements:**
        - Current token must be valid (not expired)
        - Token must be within refresh window (default: can refresh if >15 minutes remaining)
        - User activity within refresh period (default: 24 hours)
        - No suspicious activity detected on session
        
        **New Token Properties:**
        - Same permissions as original token
        - Extended expiration time
        - New token ID for security tracking
        - Original token immediately invalidated
        
        This enables seamless user experience without requiring re-authentication.
      operationId: refreshSessionToken
      responses:
        '200':
          description: Token refreshed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TokenResponse'
        '401':
          description: Current token invalid or expired
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '403':
          description: Token not eligible for refresh
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  # ==========================================
  # KNOWLEDGE BASE ENDPOINTS
  # ==========================================

  /v1/knowledge/upload:
    post:
      tags:
        - Knowledge Base
      summary: Upload document to knowledge base
      description: |
        Upload a document to the knowledge base for RAG processing. The system will:
        
        **Document Processing Pipeline:**
        1. **File Validation**: Check file type, size, and content
        2. **Content Extraction**: Extract text from various formats (PDF, DOCX, TXT, MD, etc.)
        3. **Text Chunking**: Split content into optimal chunks for embedding
        4. **Metadata Extraction**: Extract title, headings, structure, etc.
        5. **Embedding Generation**: Create vector embeddings for semantic search
        6. **Vector Storage**: Store embeddings in configured vector database
        7. **Indexing**: Create searchable index with metadata
        
        **Supported Formats:**
        - PDF documents (.pdf)
        - Microsoft Word (.docx)
        - Plain text (.txt)
        - Markdown (.md)
        - Rich Text Format (.rtf)
        - HTML documents (.html)
        
        **Processing Features:**
        - Automatic language detection
        - OCR for scanned documents (optional)
        - Duplicate detection and handling
        - Configurable chunk size and overlap
        - Custom metadata tagging
        - Expiration date support
      operationId: uploadDocument
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                  description: Document file to upload
                category:
                  type: string
                  enum: [general, technical, policy, faq, legal]
                  description: Document category for organization
                  default: general
                tags:
                  type: array
                  items:
                    type: string
                  description: Comma-separated tags for document classification
                  example: ["troubleshooting", "network", "guide"]
                title:
                  type: string
                  description: Custom title (if not extracted from document)
                  maxLength: 200
                expires_at:
                  type: string
                  format: date-time
                  description: Optional expiration date for document
                chunk_size:
                  type: integer
                  minimum: 100
                  maximum: 2000
                  default: 500
                  description: Maximum characters per chunk
                chunk_overlap:
                  type: integer
                  minimum: 0
                  maximum: 200
                  default: 50
                  description: Character overlap between chunks
              required:
                - file
            encoding:
              file:
                contentType: application/pdf, application/vnd.openxmlformats-officedocument.wordprocessingml.document, text/plain, text/markdown
      responses:
        '201':
          description: Document uploaded and processed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentUploadResponse'
              example:
                document_id: "123e4567-e89b-12d3-a456-426614174000"
                filename: "network_troubleshooting_guide.pdf"
                title: "Network Troubleshooting Guide"
                category: "technical"
                tags: ["network", "troubleshooting", "guide"]
                processing_status: "completed"
                chunks_created: 15
                processing_time_ms: 5420
                file_size_bytes: 245760
                expires_at: "2025-01-15T00:00:00Z"
                upload_timestamp: "2024-01-15T10:30:00Z"
                extracted_metadata:
                  author: "IT Department"
                  created_date: "2024-01-10T00:00:00Z"
                  language: "en"
                  page_count: 12
        '400':
          description: Invalid file format or processing error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: "invalid_file_format"
                message: "Unsupported file type. Please upload PDF, DOCX, TXT, or MD files."
                details:
                  supported_formats: ["pdf", "docx", "txt", "md", "rtf", "html"]
                  file_type_detected: "xlsx"
                request_id: "req_upload_123"
                timestamp: "2024-01-15T10:30:00Z"
        '413':
          description: File too large
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '422':
          description: Document processing failed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/knowledge/documents:
    get:
      tags:
        - Knowledge Base
      summary: List documents in knowledge base
      description: |
        Retrieve a paginated list of documents in the knowledge base with filtering and sorting options.
        
        **Query Capabilities:**
        - Filter by category, tags, upload date range
        - Search by title or content keywords
        - Sort by relevance, date, title, or size
        - Pagination with configurable page sizes
        - Include/exclude expired documents
        
        **Response Information:**
        - Document metadata and statistics
        - Processing status and health
        - Usage analytics (search frequency)
        - Vector embedding status
        - Expiration information
        
        This endpoint is useful for knowledge base management, content auditing, and analytics.
      operationId: listDocuments
      parameters:
        - name: category
          in: query
          description: Filter by document category
          schema:
            type: string
            enum: [general, technical, policy, faq, legal]
        - name: tags
          in: query
          description: Filter by tags (comma-separated)
          schema:
            type: string
          example: "network,troubleshooting"
        - name: search
          in: query
          description: Search query for title and content
          schema:
            type: string
          example: "password reset"
        - name: limit
          in: query
          description: Maximum number of documents to return
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
        - name: offset
          in: query
          description: Number of documents to skip for pagination
          schema:
            type: integer
            minimum: 0
            default: 0
        - name: sort_by
          in: query
          description: Sort field
          schema:
            type: string
            enum: [upload_date, title, file_size, category, relevance]
            default: upload_date
        - name: sort_order
          in: query
          description: Sort order
          schema:
            type: string
            enum: [asc, desc]
            default: desc
        - name: include_expired
          in: query
          description: Include expired documents in results
          schema:
            type: boolean
            default: false
      responses:
        '200':
          description: Documents retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentListResponse'
        '400':
          description: Invalid query parameters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/knowledge/documents/{document_id}:
    get:
      tags:
        - Knowledge Base
      summary: Get document details
      description: |
        Retrieve detailed information about a specific document including:
        
        - Complete metadata and properties
        - Processing status and chunk information
        - Usage statistics and analytics
        - Vector embedding status
        - Content preview and structure
        - Related documents and suggestions
        
        This endpoint provides comprehensive document information for management and debugging.
      operationId: getDocumentDetails
      parameters:
        - name: document_id
          in: path
          required: true
          description: Unique identifier for the document
          schema:
            type: string
            format: uuid
        - name: include_chunks
          in: query
          description: Include chunk details in response
          schema:
            type: boolean
            default: false
        - name: include_analytics
          in: query
          description: Include usage analytics
          schema:
            type: boolean
            default: true
      responses:
        '200':
          description: Document details retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentDetails'
        '404':
          description: Document not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

    delete:
      tags:
        - Knowledge Base
      summary: Delete document from knowledge base
      description: |
        Permanently delete a document from the knowledge base. This operation:
        
        1. **Removes Vector Embeddings**: Deletes all embeddings from vector database
        2. **Cleans Metadata**: Removes document metadata from primary database
        3. **Updates Cache**: Invalidates related semantic cache entries
        4. **Audit Logging**: Records deletion for compliance and debugging
        5. **File Cleanup**: Removes original file from storage (if configured)
        
        **Important Notes:**
        - This operation is irreversible
        - Related conversations may lose context
        - Cached responses referencing this document will be invalidated
        - Deletion is atomic - either fully succeeds or fully fails
        
        Consider document deactivation instead of deletion for content that may be referenced again.
      operationId: deleteDocument
      parameters:
        - name: document_id
          in: path
          required: true
          description: Unique identifier for the document to delete
          schema:
            type: string
            format: uuid
        - name: force
          in: query
          description: Force deletion even if document is recently accessed
          schema:
            type: boolean
            default: false
      responses:
        '200':
          description: Document deleted successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                    example: true
                  message:
                    type: string
                    example: "Document deleted successfully"
                  document_id:
                    type: string
                    format: uuid
                  deleted_at:
                    type: string
                    format: date-time
                  cleanup_status:
                    type: object
                    properties:
                      vector_embeddings:
                        type: string
                        example: "removed"
                      metadata:
                        type: string
                        example: "removed"
                      cache_entries:
                        type: string
                        example: "invalidated"
                      file_storage:
                        type: string
                        example: "removed"
        '404':
          description: Document not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '409':
          description: Document cannot be deleted (recently accessed, etc.)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/knowledge/search:
    post:
      tags:
        - Knowledge Base
      summary: Search knowledge base
      description: |
        Perform semantic search across the knowledge base using vector similarity.
        
        **Search Capabilities:**
        - **Semantic Search**: Vector similarity using embeddings
        - **Keyword Search**: Traditional text-based search
        - **Hybrid Search**: Combines semantic and keyword approaches
        - **Faceted Search**: Filter by categories, tags, date ranges
        - **Relevance Ranking**: ML-based relevance scoring
        - **Context-Aware**: Considers conversation context for better results
        
        **Search Features:**
        - Configurable similarity thresholds
        - Multi-language support
        - Typo tolerance and fuzzy matching
        - Real-time results with caching
        - Search analytics and optimization
        
        Results include relevance scores, matched text chunks, and suggested follow-up queries.
      operationId: searchKnowledgeBase
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/KnowledgeSearchRequest'
            example:
              query: "How to reset password when locked out"
              search_type: "hybrid"
              max_results: 10
              min_relevance_score: 0.7
              filters:
                categories: ["technical", "faq"]
                tags: ["password", "security"]
              include_chunks: true
      responses:
        '200':
          description: Search completed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentSearchResponse'
        '400':
          description: Invalid search parameters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  # ==========================================
  # CONFIGURATION ENDPOINTS
  # ==========================================

  /v1/config:
    get:
      tags:
        - Configuration
      summary: Get system configuration
      description: |
        Retrieve current system configuration including:
        
        **Configuration Categories:**
        - **Rate Limiting**: Request limits per user type and time window
        - **Model Settings**: LLM provider settings and routing rules
        - **Authentication**: OTP settings, session timeouts, security policies
        - **Knowledge Base**: Document processing, chunking, and retention settings
        - **Caching**: Semantic cache configuration and expiration policies
        - **Analytics**: Metrics collection and retention settings
        - **Security**: Encryption, audit logging, and compliance settings
        
        **Access Control:**
        - Requires admin-level authentication
        - Sensitive values are masked in response
        - Configuration versioning for change tracking
        - Audit logging for configuration access
        
        This endpoint is essential for system administration and troubleshooting.
      operationId: getConfiguration
      responses:
        '200':
          description: Configuration retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SystemConfiguration'
              example:
                rate_limits:
                  guest_per_minute: 10
                  authenticated_per_minute: 60
                  premium_per_minute: 300
                  global_per_minute: 10000
                models:
                  default_provider: "openai"
                  simple_query_model: "gpt-3.5-turbo"
                  complex_query_model: "gpt-4"
                  embedding_model: "text-embedding-3-small"
                  routing_threshold: 0.7
                authentication:
                  otp_expiry_seconds: 300
                  session_timeout_hours: 1
                  max_verification_attempts: 3
                  rate_limit_window_minutes: 15
                knowledge_base:
                  default_chunk_size: 500
                  chunk_overlap: 50
                  supported_formats: ["pdf", "docx", "txt", "md"]
                  max_file_size_mb: 50
                  document_retention_days: 365
                caching:
                  semantic_similarity_threshold: 0.85
                  cache_expiry_hours: 24
                  max_cache_entries: 10000
                analytics:
                  retention_days: 90
                  anonymize_pii: true
                  export_formats: ["json", "csv"]
                version: "1.2.0"
                last_updated: "2024-01-15T09:00:00Z"
        '403':
          description: Insufficient permissions to access configuration
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

    put:
      tags:
        - Configuration
      summary: Update system configuration
      description: |
        Update system configuration with validation and rollback capabilities.
        
        **Update Process:**
        1. **Validation**: Verify all configuration values are valid
        2. **Backup**: Create backup of current configuration
        3. **Apply**: Update configuration atomically
        4. **Verification**: Test critical system functions
        5. **Rollback**: Automatic rollback if verification fails
        
        **Configuration Validation:**
        - Value range and type checking
        - Dependency validation (e.g., rate limits vs. system capacity)
        - Security policy compliance
        - Performance impact assessment
        
        **Change Management:**
        - Configuration versioning
        - Change audit logging
        - Notification to system administrators
        - Gradual rollout for critical changes
        
        Only partial updates are supported - include only fields to change.
      operationId: updateConfiguration
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SystemConfigurationUpdate'
            example:
              rate_limits:
                authenticated_per_minute: 80
                premium_per_minute: 400
              models:
                complex_query_model: "gpt-4-turbo"
                routing_threshold: 0.75
              knowledge_base:
                max_file_size_mb: 100
      responses:
        '200':
          description: Configuration updated successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                    example: true
                  message:
                    type: string
                    example: "Configuration updated successfully"
                  updated_fields:
                    type: array
                    items:
                      type: string
                    example: ["rate_limits.authenticated_per_minute", "models.complex_query_model"]
                  version:
                    type: string
                    example: "1.2.1"
                  updated_at:
                    type: string
                    format: date-time
                  backup_id:
                    type: string
                    description: ID for configuration backup (for rollback)
        '400':
          description: Invalid configuration values
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: "validation_error"
                message: "Invalid configuration values provided"
                details:
                  invalid_fields:
                    - field: "rate_limits.authenticated_per_minute"
                      error: "Value exceeds system capacity (max: 100)"
                    - field: "models.routing_threshold"
                      error: "Must be between 0.0 and 1.0"
        '403':
          description: Insufficient permissions to modify configuration
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  # ==========================================
  # ANALYTICS ENDPOINTS
  # ==========================================

  /v1/analytics/conversations:
    get:
      tags:
        - Analytics
      summary: Get conversation analytics
      description: |
        Retrieve comprehensive conversation analytics including:
        
        **Metrics Provided:**
        - **Volume Metrics**: Total conversations, messages per conversation, daily/hourly patterns
        - **Performance Metrics**: Average response time, pipeline step performance, error rates
        - **User Engagement**: Session duration, return users, conversation completion rates
        - **Quality Metrics**: User satisfaction scores, feedback sentiment, resolution rates
        - **Cache Efficiency**: Hit rates, response time improvements, cost savings
        - **Authentication Patterns**: Authentication frequency, method preferences, success rates
        
        **Aggregation Options:**
        - Time-based aggregation (hour, day, week, month)
        - User segment analysis (guest vs. authenticated)
        - Geographic distribution (if location data available)
        - Device/platform breakdown
        
        **Data Export:**
        - JSON format for API integration
        - CSV format for spreadsheet analysis
        - Real-time streaming for dashboards
        
        Analytics data is anonymized and complies with privacy regulations.
      operationId: getConversationAnalytics
      parameters:
        - name: start_date
          in: query
          description: Start date for analytics period (ISO 8601 format)
          schema:
            type: string
            format: date-time
          example: "2024-01-01T00:00:00Z"
        - name: end_date
          in: query
          description: End date for analytics period (ISO 8601 format)
          schema:
            type: string
            format: date-time
          example: "2024-01-31T23:59:59Z"
        - name: granularity
          in: query
          description: Data aggregation granularity
          schema:
            type: string
            enum: [hour, day, week, month]
            default: day
        - name: metrics
          in: query
          description: Specific metrics to include (comma-separated)
          schema:
            type: string
          example: "volume,performance,satisfaction"
        - name: segment_by
          in: query
          description: Segmentation dimension
          schema:
            type: string
            enum: [user_type, device_type, geographic, none]
            default: none
      responses:
        '200':
          description: Analytics data retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ConversationAnalytics'
              example:
                period:
                  start_date: "2024-01-01T00:00:00Z"
                  end_date: "2024-01-31T23:59:59Z"
                  granularity: "day"
                summary:
                  total_conversations: 1250
                  total_messages: 8420
                  average_messages_per_conversation: 6.7
                  average_session_duration_minutes: 12.3
                  user_satisfaction_score: 4.2
                  cache_hit_rate: 0.34
                metrics:
                  daily_data:
                    - date: "2024-01-01"
                      conversations: 45
                      messages: 310
                      avg_response_time_ms: 1250
                      satisfaction_score: 4.1
                    - date: "2024-01-02"
                      conversations: 52
                      messages: 380
                      avg_response_time_ms: 1180
                      satisfaction_score: 4.3
        '400':
          description: Invalid analytics parameters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/analytics/performance:
    get:
      tags:
        - Analytics
      summary: Get system performance analytics
      description: |
        Retrieve detailed system performance analytics including:
        
        **Performance Categories:**
        - **Response Times**: End-to-end latency, pipeline step breakdown, percentile distributions
        - **Throughput**: Requests per second, concurrent user capacity, peak load handling
        - **Resource Utilization**: CPU, memory, database, and vector store performance
        - **Error Analysis**: Error rates by type, failure patterns, recovery times
        - **Cache Performance**: Hit rates, miss patterns, invalidation frequency
        - **Model Performance**: Response quality metrics, token usage, cost efficiency
        
        **Monitoring Features:**
        - Real-time performance dashboards
        - Alerting thresholds and notifications
        - Anomaly detection and root cause analysis
        - Capacity planning recommendations
        - Performance trend analysis
        
        **Optimization Insights:**
        - Bottleneck identification
        - Caching optimization opportunities
        - Model routing efficiency
        - Resource allocation recommendations
        
        This data is crucial for system optimization and capacity planning.
      operationId: getPerformanceAnalytics
      parameters:
        - name: start_date
          in: query
          description: Start date for performance analysis
          schema:
            type: string
            format: date-time
        - name: end_date
          in: query
          description: End date for performance analysis
          schema:
            type: string
            format: date-time
        - name: metric_type
          in: query
          description: Type of performance metrics to retrieve
          schema:
            type: string
            enum: [response_time, throughput, errors, cache, resources, all]
            default: all
        - name: granularity
          in: query
          description: Data granularity for time-series analysis
          schema:
            type: string
            enum: [minute, hour, day]
            default: hour
      responses:
        '200':
          description: Performance analytics retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PerformanceAnalytics'
        '400':
          description: Invalid performance analytics parameters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/analytics/costs:
    get:
      tags:
        - Analytics
      summary: Get cost analysis
      description: |
        Retrieve detailed cost analysis for LLM usage and system operations.
        
        **Cost Categories:**
        - **LLM Provider Costs**: Token usage costs by provider and model
        - **Infrastructure Costs**: Database, caching, storage, and compute costs
        - **Vector Database Costs**: Embedding storage and query costs
        - **Authentication Costs**: SMS and email delivery costs
        - **Total Cost of Ownership**: Complete cost breakdown and trends
        
        **Cost Optimization:**
        - Model routing efficiency analysis
        - Cache ROI and savings calculation
        - Usage pattern optimization opportunities
        - Cost forecasting and budgeting
        
        **Analytics Features:**
        - Cost per conversation analysis
        - User segment cost analysis
        - Geographic cost distribution
        - Cost trend prediction
        
        Helps organizations optimize their AI spending and plan budgets effectively.
      operationId: getCostAnalysis
      parameters:
        - name: start_date
          in: query
          description: Start date for cost analysis
          schema:
            type: string
            format: date-time
        - name: end_date
          in: query
          description: End date for cost analysis
          schema:
            type: string
            format: date-time
        - name: group_by
          in: query
          description: Grouping dimension for cost analysis
          schema:
            type: string
            enum: [provider, model, user_type, day, week, month]
            default: day
        - name: currency
          in: query
          description: Currency for cost reporting
          schema:
            type: string
            enum: [USD, EUR, GBP]
            default: USD
      responses:
        '200':
          description: Cost analysis retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CostAnalytics'
        '400':
          description: Invalid cost analysis parameters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  # ==========================================
  # HEALTH AND STATUS ENDPOINTS
  # ==========================================

  /v1/health:
    get:
      tags:
        - Health
      summary: System health check
      description: |
        Comprehensive system health check including all critical components:
        
        **Health Checks:**
        - **API Server**: Response time and availability
        - **Database**: Connection, query performance, disk space
        - **Redis Cache**: Connection, memory usage, performance
        - **Vector Database**: Connection, index status, query performance
        - **LLM Providers**: API availability, rate limit status, response times
        - **External Services**: SMS/email providers, file storage, monitoring
        
        **Health Status Levels:**
        - **Healthy**: All systems operational within normal parameters
        - **Degraded**: Some components showing reduced performance
        - **Unhealthy**: Critical components failing or severely degraded
        - **Maintenance**: Planned maintenance mode
        
        **Monitoring Integration:**
        - Prometheus metrics export
        - Custom alerting webhook support
        - Health check history and trends
        - Automated remediation triggers
        
        This endpoint is typically used by load balancers and monitoring systems.
      operationId: healthCheck
      security: []
      responses:
        '200':
          description: System is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'
              example:
                status: "healthy"
                version: "1.2.0"
                timestamp: "2024-01-15T10:30:00Z"
                uptime_seconds: 3600
                components:
                  api_server:
                    status: "healthy"
                    response_time_ms: 5
                    last_check: "2024-01-15T10:30:00Z"
                  database:
                    status: "healthy"
                    response_time_ms: 15
                    connections_active: 8
                    connections_max: 100
                    last_check: "2024-01-15T10:30:00Z"
                  redis:
                    status: "healthy"
                    response_time_ms: 2
                    memory_usage_percent: 45
                    last_check: "2024-01-15T10:30:00Z"
                  vector_db:
                    status: "healthy"
                    response_time_ms: 120
                    index_status: "ready"
                    last_check: "2024-01-15T10:30:00Z"
                  llm_providers:
                    openai:
                      status: "healthy"
                      response_time_ms: 850
                      rate_limit_remaining: 450
                      last_check: "2024-01-15T10:29:45Z"
                    anthropic:
                      status: "healthy"
                      response_time_ms: 920
                      rate_limit_remaining: 980
                      last_check: "2024-01-15T10:29:50Z"
        '503':
          description: System is unhealthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'
              example:
                status: "unhealthy"
                version: "1.2.0"
                timestamp: "2024-01-15T10:30:00Z"
                uptime_seconds: 3600
                components:
                  api_server:
                    status: "healthy"
                    response_time_ms: 5
                  database:
                    status: "unhealthy"
                    error: "Connection timeout"
                    last_check: "2024-01-15T10:29:30Z"
                  redis:
                    status: "degraded"
                    response_time_ms: 150
                    memory_usage_percent: 95
                    warning: "High memory usage"

  /v1/metrics:
    get:
      tags:
        - Health
      summary: Get system metrics
      description: |
        Retrieve real-time system metrics in Prometheus format or JSON.
        
        **Metrics Categories:**
        - **Request Metrics**: Request rate, response times, error rates
        - **System Metrics**: CPU, memory, disk usage, network I/O
        - **Application Metrics**: Cache hit rates, queue sizes, active sessions
        - **Business Metrics**: Conversations, users, document uploads
        - **Cost Metrics**: Token usage, provider costs, efficiency ratios
        
        **Export Formats:**
        - Prometheus format for monitoring integration
        - JSON format for custom dashboards
        - CSV format for data analysis
        
        Metrics are updated in real-time and include historical context.
      operationId: getMetrics
      security: []
      parameters:
        - name: format
          in: query
          description: Metrics export format
          schema:
            type: string
            enum: [prometheus, json]
            default: json
      responses:
        '200':
          description: Metrics retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MetricsData'
            text/plain:
              schema:
                type: string
                description: Prometheus metrics format

# ==========================================
# COMPONENT SCHEMAS
# ==========================================

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |
        JWT session token obtained through OTP verification.
        Include the token in the Authorization header as: `Bearer <token>`

  schemas:
    # ==========================================
    # CHAT SCHEMAS
    # ==========================================
    
    ChatMessage:
      type: object
      required:
        - session_id
        - message
      properties:
        session_id:
          type: string
          description: Unique session identifier for conversation context
          example: "session_12345"
        message:
          type: string
          minLength: 1
          maxLength: 4000
          description: User message content to process
          example: "What are your business hours?"
        conversation_id:
          type: string
          format: uuid
          nullable: true
          description: Optional conversation ID for continuing existing conversation
          example: "123e4567-e89b-12d3-a456-426614174000"
        user_context:
          type: object
          description: Optional user context for personalization
          properties:
            timezone:
              type: string
              description: User's timezone (IANA format)
              example: "America/New_York"
            user_agent:
              type: string
              description: Browser user agent string
              example: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            device_type:
              type: string
              enum: [desktop, mobile, tablet]
              description: Device type for response optimization
            referrer:
              type: string
              description: Page referrer URL
              example: "https://example.com/support"
            language:
              type: string
              description: Preferred language (ISO 639-1)
              example: "en"

    ChatResponse:
      type: object
      required:
        - message_id
        - response
        - conversation_id
        - session_id
        - cached
        - processing_time_ms
        - timestamp
      properties:
        message_id:
          type: string
          format: uuid
          description: Unique identifier for this message
          example: "msg_123e4567-e89b-12d3-a456-426614174000"
        response:
          type: string
          description: Generated response to user message
          example: "Our business hours are Monday-Friday 9 AM to 6 PM EST."
        conversation_id:
          type: string
          format: uuid
          description: Conversation identifier for message threading
          example: "conv_123e4567-e89b-12d3-a456-426614174000"
        session_id:
          type: string
          description: Session identifier from request
          example: "session_12345"
        cached:
          type: boolean
          description: Whether response was served from semantic cache
          example: true
        processing_time_ms:
          type: integer
          minimum: 0
          description: Total processing time in milliseconds
          example: 1250
        pipeline_steps:
          type: array
          description: Detailed breakdown of pipeline processing steps
          items:
            type: object
            properties:
              step:
                type: string
                description: Pipeline step name
                example: "semantic_cache"
              duration_ms:
                type: integer
                description: Step processing time
                example: 35
              status:
                type: string
                enum: [passed, failed, skipped, hit, miss, required]
                description: Step execution status
                example: "hit"
              metadata:
                type: object
                description: Step-specific metadata
                additionalProperties: true
        suggested_actions:
          type: array
          description: Suggested follow-up actions for user
          items:
            type: object
            properties:
              action:
                type: string
                description: Action type
                example: "authenticate"
              description:
                type: string
                description: Human-readable action description
                example: "Complete verification to access account features"
              endpoint:
                type: string
                description: API endpoint for action
                example: "/api/v1/auth/request-token"
              data:
                type: object
                description: Additional action data
                additionalProperties: true
        requires_authentication:
          type: boolean
          description: Whether the query requires user authentication
          example: false
        auth_methods:
          type: array
          items:
            type: string
            enum: [sms, email]
          description: Available authentication methods if auth required
        knowledge_sources:
          type: array
          description: Knowledge base sources used in response
          items:
            type: object
            properties:
              document_id:
                type: string
                format: uuid
              title:
                type: string
              relevance_score:
                type: number
                minimum: 0
                maximum: 1
              chunk_text:
                type: string
        model_used:
          type: string
          description: LLM model used for response generation
          example: "gpt-3.5-turbo"
        cost_info:
          type: object
          description: Cost information for the request
          properties:
            tokens_used:
              type: integer
              description: Total tokens consumed
            estimated_cost_usd:
              type: number
              description: Estimated cost in USD
            provider:
              type: string
              description: LLM provider used
        timestamp:
          type: string
          format: date-time
          description: Response generation timestamp
          example: "2024-01-15T10:30:00Z"

    ConversationHistory:
      type: object
      required:
        - conversation_id
        - messages
        - total_messages
        - session_info
      properties:
        conversation_id:
          type: string
          format: uuid
          description: Conversation unique identifier
        messages:
          type: array
          description: Chronologically ordered message history
          items:
            type: object
            properties:
              message_id:
                type: string
                format: uuid
              role:
                type: string
                enum: [user, assistant, system]
              content:
                type: string
              timestamp:
                type: string
                format: date-time
              processing_info:
                type: object
                description: Processing metadata for debugging
                properties:
                  cached:
                    type: boolean
                  processing_time_ms:
                    type: integer
                  model_used:
                    type: string
                  pipeline_steps:
                    type: array
                    items:
                      type: object
        total_messages:
          type: integer
          minimum: 0
          description: Total messages in conversation
        session_info:
          type: object
          properties:
            session_id:
              type: string
            created_at:
              type: string
              format: date-time
            last_activity:
              type: string
              format: date-time
            user_authenticated:
              type: boolean
            message_count:
              type: integer
        pagination:
          type: object
          properties:
            limit:
              type: integer
            offset:
              type: integer
            has_more:
              type: boolean

    FeedbackRequest:
      type: object
      required:
        - message_id
        - feedback_type
      properties:
        message_id:
          type: string
          format: uuid
          description: ID of message being rated
        feedback_type:
          type: string
          enum: [thumbs_up, thumbs_down, rating, detailed]
          description: Type of feedback provided
        rating:
          type: integer
          minimum: 1
          maximum: 5
          description: Numeric rating (1-5, required for rating type)
        comment:
          type: string
          maxLength: 1000
          description: Optional detailed feedback comment
        categories:
          type: array
          items:
            type: string
            enum: [accuracy, helpfulness, clarity, speed, relevance]
          description: Specific feedback categories
        anonymous:
          type: boolean
          default: true
          description: Whether feedback should be anonymous

    FeedbackResponse:
      type: object
      properties:
        success:
          type: boolean
          example: true
        message:
          type: string
          example: "Feedback submitted successfully"
        feedback_id:
          type: string
          format: uuid
          description: Unique identifier for submitted feedback

    # ==========================================
    # AUTHENTICATION SCHEMAS
    # ==========================================

    AuthRequest:
      type: object
      required:
        - session_id
        - contact_method
        - contact_value
      properties:
        session_id:
          type: string
          description: Session identifier requiring authentication
          example: "session_12345"
        contact_method:
          type: string
          enum: [sms, email]
          description: Method for OTP delivery
        contact_value:
          type: string
          description: Phone number (+1234567890) or email address
          example: "user@example.com"

    AuthVerification:
      type: object
      required:
        - session_id
        - token
      properties:
        session_id:
          type: string
          description: Session identifier from auth request
          example: "session_12345"
        token:
          type: string
          minLength: 4
          maxLength: 10
          description: OTP received via SMS or email
          example: "ABC123"

    AuthResponse:
      type: object
      required:
        - success
        - message
      properties:
        success:
          type: boolean
          description: Whether OTP was sent successfully
        message:
          type: string
          description: Human-readable response message
          example: "Authentication code sent to your email"
        expires_in:
          type: integer
          minimum: 1
          description: OTP expiration time in seconds
          example: 300
        retry_after:
          type: integer
          nullable: true
          description: Seconds before another OTP can be requested
          example: null

    TokenResponse:
      type: object
      required:
        - success
        - message
      properties:
        success:
          type: boolean
          description: Whether token verification was successful
        message:
          type: string
          description: Response message
          example: "Authentication successful"
        session_token:
          type: string
          nullable: true
          description: JWT session token for authenticated requests
          example: "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
        expires_at:
          type: string
          format: date-time
          nullable: true
          description: Session token expiration timestamp
        user_id:
          type: string
          format: uuid
          nullable: true
          description: User identifier if authentication successful

    AuthStatus:
      type: object
      required:
        - session_id
        - authenticated
      properties:
        session_id:
          type: string
          description: Session identifier
        authenticated:
          type: boolean
          description: Whether session is currently authenticated
        expires_at:
          type: string
          format: date-time
          nullable: true
          description: Authentication expiration timestamp
        time_remaining:
          type: integer
          nullable: true
          description: Seconds until authentication expires
        requires_renewal:
          type: boolean
          default: false
          description: Whether token should be renewed soon
        user_profile:
          $ref: '#/components/schemas/UserProfile'

    UserProfile:
      type: object
      required:
        - id
        - created_at
        - authentication_count
      properties:
        id:
          type: string
          format: uuid
          description: User unique identifier
        mobile_number:
          type: string
          nullable: true
          description: User's mobile number
          example: "+1234567890"
        email:
          type: string
          format: email
          nullable: true
          description: User's email address
          example: "user@example.com"
        created_at:
          type: string
          format: date-time
          description: Account creation timestamp
        last_authenticated:
          type: string
          format: date-time
          nullable: true
          description: Last successful authentication timestamp
        authentication_count:
          type: integer
          minimum: 0
          description: Total number of successful authentications
        preferences:
          type: object
          description: User preferences and settings
          properties:
            language:
              type: string
              example: "en"
            timezone:
              type: string
              example: "America/New_York"
            notification_methods:
              type: array
              items:
                type: string
                enum: [sms, email]

    # ==========================================
    # KNOWLEDGE BASE SCHEMAS
    # ==========================================

    DocumentUploadResponse:
      type: object
      required:
        - document_id
        - filename
        - processing_status
        - upload_timestamp
      properties:
        document_id:
          type: string
          format: uuid
          description: Unique identifier for uploaded document
        filename:
          type: string
          description: Original filename
          example: "network_troubleshooting_guide.pdf"
        title:
          type: string
          nullable: true
          description: Extracted or provided document title
          example: "Network Troubleshooting Guide"
        category:
          type: string
          enum: [general, technical, policy, faq, legal]
          description: Document category
        tags:
          type: array
          items:
            type: string
          description: Applied tags for document classification
        processing_status:
          type: string
          enum: [processing, completed, failed]
          description: Current processing status
        chunks_created:
          type: integer
          minimum: 0
          description: Number of text chunks created
        processing_time_ms:
          type: integer
          minimum: 0
          description: Processing time in milliseconds
        file_size_bytes:
          type: integer
          minimum: 0
          description: Original file size in bytes
        expires_at:
          type: string
          format: date-time
          nullable: true
          description: Document expiration timestamp
        upload_timestamp:
          type: string
          format: date-time
          description: Upload completion timestamp
        extracted_metadata:
          type: object
          description: Metadata extracted from document
          properties:
            author:
              type: string
            created_date:
              type: string
              format: date-time
            language:
              type: string
            page_count:
              type: integer
            word_count:
              type: integer
        processing_errors:
          type: array
          items:
            type: string
          description: Any errors encountered during processing

    DocumentListResponse:
      type: object
      required:
        - documents
        - total_count
        - pagination
      properties:
        documents:
          type: array
          items:
            $ref: '#/components/schemas/DocumentSummary'
        total_count:
          type: integer
          minimum: 0
          description: Total documents matching query
        pagination:
          type: object
          properties:
            limit:
              type: integer
            offset:
              type: integer
            has_more:
              type: boolean
        filters_applied:
          type: object
          description: Summary of applied filters
          properties:
            category:
              type: string
            tags:
              type: array
              items:
                type: string
            search_query:
              type: string
            date_range:
              type: object
              properties:
                start:
                  type: string
                  format: date-time
                end:
                  type: string
                  format: date-time

    DocumentSummary:
      type: object
      required:
        - document_id
        - filename
        - category
        - upload_date
        - processing_status
      properties:
        document_id:
          type: string
          format: uuid
        filename:
          type: string
        title:
          type: string
          nullable: true
        category:
          type: string
          enum: [general, technical, policy, faq, legal]
        tags:
          type: array
          items:
            type: string
        file_size_bytes:
          type: integer
        upload_date:
          type: string
          format: date-time
        last_modified:
          type: string
          format: date-time
        processing_status:
          type: string
          enum: [processing, completed, failed]
        expires_at:
          type: string
          format: date-time
          nullable: true
        usage_stats:
          type: object
          properties:
            total_queries:
              type: integer
            last_accessed:
              type: string
              format: date-time
            avg_relevance_score:
              type: number
              minimum: 0
              maximum: 1

    DocumentDetails:
      type: object
      required:
        - document_id
        - filename
        - category
        - processing_status
        - upload_date
      properties:
        document_id:
          type: string
          format: uuid
        filename:
          type: string
        title:
          type: string
          nullable: true
        category:
          type: string
          enum: [general, technical, policy, faq, legal]
        tags:
          type: array
          items:
            type: string
        file_size_bytes:
          type: integer
        upload_date:
          type: string
          format: date-time
        last_modified:
          type: string
          format: date-time
        processing_status:
          type: string
          enum: [processing, completed, failed]
        expires_at:
          type: string
          format: date-time
          nullable: true
        content_preview:
          type: string
          description: First 500 characters of extracted content
        extracted_metadata:
          type: object
          additionalProperties: true
        chunk_info:
          type: object
          properties:
            total_chunks:
              type: integer
            avg_chunk_size:
              type: integer
            chunk_overlap:
              type: integer
            embedding_model:
              type: string
        usage_analytics:
          type: object
          properties:
            total_queries:
              type: integer
            unique_sessions:
              type: integer
            avg_relevance_score:
              type: number
            last_accessed:
              type: string
              format: date-time
            top_queries:
              type: array
              items:
                type: object
                properties:
                  query:
                    type: string
                  count:
                    type: integer
                  avg_relevance:
                    type: number
        related_documents:
          type: array
          items:
            type: object
            properties:
              document_id:
                type: string
                format: uuid
              title:
                type: string
              similarity_score:
                type: number
        chunks:
          type: array
          items:
            type: object
            properties:
              chunk_id:
                type: string
              text:
                type: string
              start_position:
                type: integer
              end_position:
                type: integer
              metadata:
                type: object

    KnowledgeSearchRequest:
      type: object
      required:
        - query
      properties:
        query:
          type: string
          minLength: 1
          maxLength: 500
          description: Search query string
          example: "How to reset password when locked out"
        search_type:
          type: string
          enum: [semantic, keyword, hybrid]
          default: hybrid
          description: Type of search to perform
        max_results:
          type: integer
          minimum: 1
          maximum: 50
          default: 10
          description: Maximum number of results to return
        min_relevance_score:
          type: number
          minimum: 0.0
          maximum: 1.0
          default: 0.7
          description: Minimum relevance score threshold
        filters:
          type: object
          description: Search result filters
          properties:
            categories:
              type: array
              items:
                type: string
                enum: [general, technical, policy, faq, legal]
            tags:
              type: array
              items:
                type: string
            date_range:
              type: object
              properties:
                start:
                  type: string
                  format: date-time
                end:
                  type: string
                  format: date-time
            file_types:
              type: array
              items:
                type: string
                enum: [pdf, docx, txt, md, html]
        include_chunks:
          type: boolean
          default: true
          description: Include relevant text chunks in results
        conversation_context:
          type: object
          description: Optional conversation context for better results
          properties:
            conversation_id:
              type: string
              format: uuid
            recent_messages:
              type: array
              items:
                type: string
            user_intent:
              type: string

    DocumentSearchResponse:
      type: object
      required:
        - results
        - total_results
        - query
        - search_time_ms
      properties:
        results:
          type: array
          items:
            $ref: '#/components/schemas/DocumentSearchResult'
        total_results:
          type: integer
          minimum: 0
          description: Total matching documents
        query:
          type: string
          description: Original search query
        search_time_ms:
          type: integer
          minimum: 0
          description: Search execution time
        search_metadata:
          type: object
          properties:
            search_type:
              type: string
            filters_applied:
              type: object
            suggestions:
              type: array
              items:
                type: string
            corrected_query:
              type: string
              description: Spell-corrected version of query if applicable

    DocumentSearchResult:
      type: object
      required:
        - document_id
        - filename
        - category
        - relevance_score
        - upload_date
      properties:
        document_id:
          type: string
          format: uuid
        filename:
          type: string
        title:
          type: string
          nullable: true
        category:
          type: string
          enum: [general, technical, policy, faq, legal]
        tags:
          type: array
          items:
            type: string
        relevance_score:
          type: number
          minimum: 0.0
          maximum: 1.0
          description: Relevance score for the search query
        matched_chunks:
          type: array
          items:
            type: object
            properties:
              chunk_id:
                type: string
              text:
                type: string
              relevance_score:
                type: number
              start_position:
                type: integer
              highlighted_text:
                type: string
                description: Text with search terms highlighted
        upload_date:
          type: string
          format: date-time
        content_preview:
          type: string
          description: Preview of document content
        match_summary:
          type: object
          properties:
            total_matches:
              type: integer
            best_match_score:
              type: number
            match_distribution:
              type: array
              items:
                type: object
                properties:
                  section:
                    type: string
                  matches:
                    type: integer

    # ==========================================
    # CONFIGURATION SCHEMAS
    # ==========================================

    SystemConfiguration:
      type: object
      properties:
        rate_limits:
          type: object
          properties:
            guest_per_minute:
              type: integer
              minimum: 1
              maximum: 1000
              description: Requests per minute for unauthenticated users
            authenticated_per_minute:
              type: integer
              minimum: 1
              maximum: 1000
              description: Requests per minute for authenticated users
            premium_per_minute:
              type: integer
              minimum: 1
              maximum: 1000
              description: Requests per minute for premium users
            global_per_minute:
              type: integer
              minimum: 100
              maximum: 100000
              description: Global system request limit per minute
        models:
          type: object
          properties:
            default_provider:
              type: string
              enum: [openai, anthropic, huggingface, local]
              description: Default LLM provider
            simple_query_model:
              type: string
              description: Model for simple queries
              example: "gpt-3.5-turbo"
            complex_query_model:
              type: string
              description: Model for complex queries
              example: "gpt-4"
            embedding_model:
              type: string
              description: Model for generating embeddings
              example: "text-embedding-3-small"
            routing_threshold:
              type: number
              minimum: 0.0
              maximum: 1.0
              description: Complexity threshold for model routing
            fallback_models:
              type: array
              items:
                type: string
              description: Fallback models if primary fails
        authentication:
          type: object
          properties:
            otp_expiry_seconds:
              type: integer
              minimum: 60
              maximum: 3600
              description: OTP expiration time in seconds
            session_timeout_hours:
              type: integer
              minimum: 1
              maximum: 24
              description: Session token timeout in hours
            max_verification_attempts:
              type: integer
              minimum: 1
              maximum: 10
              description: Maximum OTP verification attempts
            rate_limit_window_minutes:
              type: integer
              minimum: 5
              maximum: 60
              description: Rate limit window for OTP requests
        knowledge_base:
          type: object
          properties:
            default_chunk_size:
              type: integer
              minimum: 100
              maximum: 2000
              description: Default text chunk size for documents
            chunk_overlap:
              type: integer
              minimum: 0
              maximum: 200
              description: Overlap between text chunks
            supported_formats:
              type: array
              items:
                type: string
              description: Supported document formats
            max_file_size_mb:
              type: integer
              minimum: 1
              maximum: 500
              description: Maximum file size for uploads
            document_retention_days:
              type: integer
              minimum: 30
              maximum: 3650
              description: Default document retention period
        caching:
          type: object
          properties:
            semantic_similarity_threshold:
              type: number
              minimum: 0.5
              maximum: 1.0
              description: Similarity threshold for semantic cache hits
            cache_expiry_hours:
              type: integer
              minimum: 1
              maximum: 168
              description: Cache entry expiration time
            max_cache_entries:
              type: integer
              minimum: 1000
              maximum: 1000000
              description: Maximum number of cache entries
        analytics:
          type: object
          properties:
            retention_days:
              type: integer
              minimum: 30
              maximum: 365
              description: Analytics data retention period
            anonymize_pii:
              type: boolean
              description: Whether to anonymize personally identifiable information
            export_formats:
              type: array
              items:
                type: string
                enum: [json, csv]
              description: Supported export formats
        version:
          type: string
          description: Configuration version
        last_updated:
          type: string
          format: date-time
          description: Last configuration update timestamp

    SystemConfigurationUpdate:
      type: object
      description: Partial configuration update (only include fields to change)
      properties:
        rate_limits:
          type: object
          properties:
            guest_per_minute:
              type: integer
              minimum: 1
              maximum: 1000
            authenticated_per_minute:
              type: integer
              minimum: 1
              maximum: 1000
            premium_per_minute:
              type: integer
              minimum: 1
              maximum: 1000
            global_per_minute:
              type: integer
              minimum: 100
              maximum: 100000
        models:
          type: object
          properties:
            default_provider:
              type: string
              enum: [openai, anthropic, huggingface, local]
            simple_query_model:
              type: string
            complex_query_model:
              type: string
            embedding_model:
              type: string
            routing_threshold:
              type: number
              minimum: 0.0
              maximum: 1.0
        authentication:
          type: object
          properties:
            otp_expiry_seconds:
              type: integer
              minimum: 60
              maximum: 3600
            session_timeout_hours:
              type: integer
              minimum: 1
              maximum: 24
            max_verification_attempts:
              type: integer
              minimum: 1
              maximum: 10
        knowledge_base:
          type: object
          properties:
            default_chunk_size:
              type: integer
              minimum: 100
              maximum: 2000
            chunk_overlap:
              type: integer
              minimum: 0
              maximum: 200
            max_file_size_mb:
              type: integer
              minimum: 1
              maximum: 500
            document_retention_days:
              type: integer
              minimum: 30
              maximum: 3650
        caching:
          type: object
          properties:
            semantic_similarity_threshold:
              type: number
              minimum: 0.5
              maximum: 1.0
            cache_expiry_hours:
              type: integer
              minimum: 1
              maximum: 168
            max_cache_entries:
              type: integer
              minimum: 1000
              maximum: 1000000

    # ==========================================
    # ANALYTICS SCHEMAS
    # ==========================================

    ConversationAnalytics:
      type: object
      required:
        - period
        - summary
        - metrics
      properties:
        period:
          type: object
          properties:
            start_date:
              type: string
              format: date-time
            end_date:
              type: string
              format: date-time
            granularity:
              type: string
              enum: [hour, day, week, month]
        summary:
          type: object
          properties:
            total_conversations:
              type: integer
              minimum: 0
            total_messages:
              type: integer
              minimum: 0
            unique_users:
              type: integer
              minimum: 0
            average_messages_per_conversation:
              type: number
              minimum: 0
            average_session_duration_minutes:
              type: number
              minimum: 0
            user_satisfaction_score:
              type: number
              minimum: 1.0
              maximum: 5.0
            cache_hit_rate:
              type: number
              minimum: 0.0
              maximum: 1.0
            authentication_rate:
              type: number
              minimum: 0.0
              maximum: 1.0
              description: Percentage of conversations requiring authentication
        metrics:
          type: object
          properties:
            daily_data:
              type: array
              items:
                type: object
                properties:
                  date:
                    type: string
                    format: date
                  conversations:
                    type: integer
                  messages:
                    type: integer
                  unique_users:
                    type: integer
                  avg_response_time_ms:
                    type: number
                  satisfaction_score:
                    type: number
                  cache_hit_rate:
                    type: number
                  error_rate:
                    type: number
            user_segments:
              type: object
              properties:
                guest_users:
                  $ref: '#/components/schemas/UserSegmentMetrics'
                authenticated_users:
                  $ref: '#/components/schemas/UserSegmentMetrics'
                premium_users:
                  $ref: '#/components/schemas/UserSegmentMetrics'
            popular_topics:
              type: array
              items:
                type: object
                properties:
                  topic:
                    type: string
                  message_count:
                    type: integer
                  avg_satisfaction:
                    type: number
                  resolution_rate:
                    type: number

    UserSegmentMetrics:
      type: object
      properties:
        total_conversations:
          type: integer
        avg_messages_per_conversation:
          type: number
        avg_session_duration_minutes:
          type: number
        satisfaction_score:
          type: number
        retention_rate:
          type: number
        conversion_rate:
          type: number
          description: Rate of guest users becoming authenticated

    PerformanceAnalytics:
      type: object
      required:
        - period
        - summary
        - metrics
      properties:
        period:
          type: object
          properties:
            start_date:
              type: string
              format: date-time
            end_date:
              type: string
              format: date-time
            granularity:
              type: string
              enum: [minute, hour, day]
        summary:
          type: object
          properties:
            avg_response_time_ms:
              type: number
            p95_response_time_ms:
              type: number
            p99_response_time_ms:
              type: number
            total_requests:
              type: integer
            error_rate:
              type: number
            cache_hit_rate:
              type: number
            throughput_rpm:
              type: number
              description: Requests per minute
        metrics:
          type: object
          properties:
            response_times:
              type: array
              items:
                type: object
                properties:
                  timestamp:
                    type: string
                    format: date-time
                  avg_ms:
                    type: number
                  p95_ms:
                    type: number
                  p99_ms:
                    type: number
            pipeline_performance:
              type: object
              properties:
                rate_limiter:
                  type: object
                  properties:
                    avg_duration_ms:
                      type: number
                    success_rate:
                      type: number
                relevance_checker:
                  type: object
                  properties:
                    avg_duration_ms:
                      type: number
                    accuracy_rate:
                      type: number
                semantic_cache:
                  type: object
                  properties:
                    avg_duration_ms:
                      type: number
                    hit_rate:
                      type: number
                model_router:
                  type: object
                  properties:
                    avg_duration_ms:
                      type: number
                    routing_accuracy:
                      type: number
                rag_engine:
                  type: object
                  properties:
                    avg_duration_ms:
                      type: number
                    retrieval_accuracy:
                      type: number
            error_breakdown:
              type: array
              items:
                type: object
                properties:
                  error_type:
                    type: string
                  count:
                    type: integer
                  percentage:
                    type: number
                  avg_recovery_time_ms:
                    type: number
            system_resources:
              type: object
              properties:
                cpu_usage_percent:
                  type: number
                memory_usage_percent:
                  type: number
                database_connections:
                  type: integer
                redis_memory_mb:
                  type: number

    CostAnalytics:
      type: object
      required:
        - period
        - summary
        - breakdown
      properties:
        period:
          type: object
          properties:
            start_date:
              type: string
              format: date-time
            end_date:
              type: string
              format: date-time
            currency:
              type: string
              enum: [USD, EUR, GBP]
        summary:
          type: object
          properties:
            total_cost:
              type: number
              description: Total cost for the period
            cost_per_conversation:
              type: number
              description: Average cost per conversation
            cost_per_message:
              type: number
              description: Average cost per message
            token_usage:
              type: integer
              description: Total tokens consumed
            projected_monthly_cost:
              type: number
              description: Projected monthly cost based on current usage
        breakdown:
          type: object
          properties:
            by_provider:
              type: array
              items:
                type: object
                properties:
                  provider:
                    type: string
                  cost:
                    type: number
                  percentage:
                    type: number
                  tokens_used:
                    type: integer
                  requests:
                    type: integer
            by_model:
              type: array
              items:
                type: object
                properties:
                  model:
                    type: string
                  cost:
                    type: number
                  percentage:
                    type: number
                  tokens_used:
                    type: integer
                  requests:
                    type: integer
                  avg_cost_per_request:
                    type: number
            by_operation:
              type: object
              properties:
                chat_generation:
                  type: number
                embeddings:
                  type: number
                relevance_checking:
                  type: number
                knowledge_search:
                  type: number
            daily_costs:
              type: array
              items:
                type: object
                properties:
                  date:
                    type: string
                    format: date
                  cost:
                    type: number
                  conversations:
                    type: integer
                  cost_per_conversation:
                    type: number
        optimization_suggestions:
          type: array
          items:
            type: object
            properties:
              category:
                type: string
                enum: [model_selection, caching, routing, usage_patterns]
              suggestion:
                type: string
              potential_savings:
                type: number
              implementation_effort:
                type: string
                enum: [low, medium, high]

    # ==========================================
    # HEALTH AND METRICS SCHEMAS
    # ==========================================

    HealthStatus:
      type: object
      required:
        - status
        - version
        - timestamp
        - components
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy, maintenance]
          description: Overall system health status
        version:
          type: string
          description: Application version
          example: "1.2.0"
        timestamp:
          type: string
          format: date-time
          description: Health check timestamp
        uptime_seconds:
          type: integer
          minimum: 0
          description: System uptime in seconds
        components:
          type: object
          properties:
            api_server:
              $ref: '#/components/schemas/ComponentHealth'
            database:
              $ref: '#/components/schemas/ComponentHealth'
            redis:
              $ref: '#/components/schemas/ComponentHealth'
            vector_db:
              $ref: '#/components/schemas/ComponentHealth'
            llm_providers:
              type: object
              additionalProperties:
                $ref: '#/components/schemas/ComponentHealth'
            external_services:
              type: object
              properties:
                sms_provider:
                  $ref: '#/components/schemas/ComponentHealth'
                email_provider:
                  $ref: '#/components/schemas/ComponentHealth'
                file_storage:
                  $ref: '#/components/schemas/ComponentHealth'

    ComponentHealth:
      type: object
      required:
        - status
        - last_check
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy, unknown]
        response_time_ms:
          type: number
          minimum: 0
          description: Component response time
        last_check:
          type: string
          format: date-time
          description: Last health check timestamp
        error:
          type: string
          nullable: true
          description: Error message if unhealthy
        warning:
          type: string
          nullable: true
          description: Warning message if degraded
        metadata:
          type: object
          description: Component-specific health metadata
          additionalProperties: true

    MetricsData:
      type: object
      required:
        - requests_per_minute
        - average_response_time_ms
        - cache_hit_rate
        - active_conversations
        - error_rate
      properties:
        requests_per_minute:
          type: integer
          minimum: 0
          description: Current requests per minute
        average_response_time_ms:
          type: number
          minimum: 0
          description: Average response time in milliseconds
        cache_hit_rate:
          type: number
          minimum: 0.0
          maximum: 1.0
          description: Cache hit rate (0.0 to 1.0)
        active_conversations:
          type: integer
          minimum: 0
          description: Number of currently active conversations
        total_documents:
          type: integer
          minimum: 0
          description: Total documents in knowledge base
        error_rate:
          type: number
          minimum: 0.0
          maximum: 1.0
          description: Error rate (0.0 to 1.0)
        model_usage:
          type: object
          additionalProperties:
            type: integer
          description: Request count by model name
          example:
            "gpt-3.5-turbo": 1250
            "gpt-4": 340
            "claude-3-sonnet": 180
        system_resources:
          type: object
          properties:
            cpu_usage_percent:
              type: number
              minimum: 0
              maximum: 100
            memory_usage_percent:
              type: number
              minimum: 0
              maximum: 100
            disk_usage_percent:
              type: number
              minimum: 0
              maximum: 100
            database_connections_active:
              type: integer
              minimum: 0
            redis_memory_usage_mb:
              type: number
              minimum: 0

    # ==========================================
    # ERROR SCHEMAS
    # ==========================================

    ErrorResponse:
      type: object
      required:
        - error
        - message
        - request_id
        - timestamp
      properties:
        error:
          type: string
          description: Error code identifier
          example: "validation_error"
        message:
          type: string
          description: Human-readable error message
          example: "Invalid request format"
        details:
          type: object
          description: Additional error details
          additionalProperties: true
        request_id:
          type: string
          description: Unique request identifier for debugging
          example: "req_123abc456def"
        timestamp:
          type: string
          format: date-time
          description: Error occurrence timestamp
        suggested_action:
          type: string
          nullable: true
          description: Suggested action to resolve the error
          example: "Please check your request format and try again"
        documentation_url:
          type: string
          format: uri
          nullable: true
          description: Link to relevant documentation
          example: "https://docs.chatbot-platform.com/errors/validation"

# ==========================================
# TAGS FOR ORGANIZATION
# ==========================================

tags:
  - name: Chat
    description: |
      Core chat functionality including message processing, conversation management, 
      and the intelligent processing pipeline. These endpoints handle the primary 
      user interaction with the chatbot system.
  - name: Authentication
    description: |
      OTP-based authentication system with SMS and email delivery options. 
      Provides secure session management with JWT tokens and configurable 
      expiration policies.
  - name: Knowledge Base
    description: |
      Document management and RAG (Retrieval-Augmented Generation) functionality. 
      Handles document upload, processing, chunking, vector storage, and semantic search.
  - name: Configuration
    description: |
      System configuration management for administrators. Controls rate limiting, 
      model settings, authentication policies, caching, and other system parameters.
  - name: Analytics
    description: |
      Comprehensive analytics and reporting including conversation metrics, 
      performance analysis, cost tracking, and usage patterns.
  - name: Health
    description: |
      System health monitoring, metrics collection, and status reporting. 
      Provides real-time system status and performance metrics for monitoring 
      and alerting systems.

# ==========================================
# EXAMPLES AND USE CASES
# ==========================================

x-examples:
  basic_chat_flow:
    summary: Basic chat conversation flow
    description: |
      This example demonstrates a typical chat interaction without authentication:
      
      1. User sends a simple question
      2. System processes through pipeline (rate limit → relevance check → cache check → model routing → response generation)
      3. Response served from semantic cache for improved performance
    steps:
      - step: 1
        description: Send chat message
        request:
          method: POST
          url: /v1/chat/message
          body:
            session_id: "session_12345"
            message: "What are your business hours?"
        response:
          status: 200
          body:
            message_id: "msg_123e4567"
            response: "Our business hours are Monday-Friday 9 AM to 6 PM EST."
            cached: true
            processing_time_ms: 45
            requires_authentication: false

  authenticated_chat_flow:
    summary: Chat flow requiring authentication
    description: |
      This example shows how the system handles queries requiring authentication:
      
      1. User asks sensitive question
      2. System detects authentication requirement
      3. User completes OTP authentication
      4. User repeats query with session token
      5. System provides authenticated response
    steps:
      - step: 1
        description: Send sensitive query
        request:
          method: POST
          url: /v1/chat/message
          body:
            session_id: "session_67890"
            message: "Can you help me reset my account password?"
        response:
          status: 200
          body:
            response: "I'd be happy to help with your account. Please verify your identity first."
            requires_authentication: true
            auth_methods: ["sms", "email"]
      - step: 2
        description: Request OTP
        request:
          method: POST
          url: /v1/auth/request-token
          body:
            session_id: "session_67890"
            contact_method: "email"
            contact_value: "user@example.com"
        response:
          status: 200
          body:
            success: true
            message: "Authentication code sent to your email"
            expires_in: 300
      - step: 3
        description: Verify OTP
        request:
          method: POST
          url: /v1/auth/verify-token
          body:
            session_id: "session_67890"
            token: "ABC123"
        response:
          status: 200
          body:
            success: true
            session_token: "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
            expires_at: "2024-01-15T11:30:00Z"
      - step: 4
        description: Repeat query with authentication
        request:
          method: POST
          url: /v1/chat/message
          headers:
            Authorization: "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
          body:
            session_id: "session_67890"
            message: "Can you help me reset my account password?"
        response:
          status: 200
          body:
            response: "I can help you reset your password. Here are the steps..."
            requires_authentication: false

  knowledge_base_management:
    summary: Knowledge base document lifecycle
    description: |
      This example demonstrates the complete lifecycle of a knowledge base document:
      
      1. Upload document with metadata
      2. Monitor processing status
      3. Search document content
      4. View document analytics
      5. Update document expiration
      6. Delete document when no longer needed
    steps:
      - step: 1
        description: Upload document
        request:
          method: POST
          url: /v1/knowledge/upload
          headers:
            Authorization: "Bearer <admin_token>"
            Content-Type: "multipart/form-data"
          body:
            file: "network_troubleshooting.pdf"
            category: "technical"
            tags: ["network", "troubleshooting", "guide"]
            expires_at: "2025-01-15T00:00:00Z"
        response:
          status: 201
          body:
            document_id: "123e4567-e89b-12d3-a456-426614174000"
            processing_status: "completed"
            chunks_created: 15
      - step: 2
        description: Search knowledge base
        request:
          method: POST
          url: /v1/knowledge/search
          body:
            query: "network connectivity issues"
            search_type: "hybrid"
            max_results: 5
        response:
          status: 200
          body:
            results:
              - document_id: "123e4567-e89b-12d3-a456-426614174000"
                relevance_score: 0.89
                matched_chunks: ["To troubleshoot network connectivity..."]
      - step: 3
        description: Delete document
        request:
          method: DELETE
          url: /v1/knowledge/documents/123e4567-e89b-12d3-a456-426614174000
          headers:
            Authorization: "Bearer <admin_token>"
        response:
          status: 200
          body:
            success: true
            message: "Document deleted successfully"

# ==========================================
# API TESTING GUIDE
# ==========================================

x-testing-guide:
  overview: |
    This section provides guidance for incrementally testing the chatbot platform API backend functionality.
    Tests are organized by complexity and dependency requirements.

  testing_phases:
    phase_1_basic_connectivity:
      description: Test basic API connectivity and health endpoints
      prerequisites: []
      endpoints_to_test:
        - GET /v1/health
        - GET /v1/metrics
      test_scenarios:
        - name: "Health Check Success"
          endpoint: "GET /v1/health"
          expected_status: 200
          expected_response:
            status: "healthy"
            components:
              api_server:
                status: "healthy"
        - name: "Metrics Collection"
          endpoint: "GET /v1/metrics"
          expected_status: 200
          expected_fields:
            - "requests_per_minute"
            - "average_response_time_ms"
            - "cache_hit_rate"

    phase_2_authentication:
      description: Test OTP authentication flow
      prerequisites:
        - SMS/Email provider configured
        - Redis available for session storage
      endpoints_to_test:
        - POST /v1/auth/request-token
        - POST /v1/auth/verify-token
        - GET /v1/auth/status
        - POST /v1/auth/refresh
      test_scenarios:
        - name: "OTP Request Success"
          endpoint: "POST /v1/auth/request-token"
          request_body:
            session_id: "test_session_001"
            contact_method: "email"
            contact_value: "test@example.com"
          expected_status: 200
          expected_response:
            success: true
            expires_in: 300
        - name: "OTP Verification Success"
          endpoint: "POST /v1/auth/verify-token"
          request_body:
            session_id: "test_session_001"
            token: "123456"
          expected_status: 200
          expected_fields:
            - "session_token"
            - "expires_at"
        - name: "Authentication Status Check"
          endpoint: "GET /v1/auth/status?session_id=test_session_001"
          expected_status: 200
          expected_response:
            authenticated: true

    phase_3_basic_chat:
      description: Test basic chat functionality without authentication
      prerequisites:
        - At least one LLM provider configured
        - Rate limiting system functional
        - Redis cache available
      endpoints_to_test:
        - POST /v1/chat/message
        - GET /v1/chat/conversations/{id}/history
        - POST /v1/chat/feedback
      test_scenarios:
        - name: "Simple Chat Message"
          endpoint: "POST /v1/chat/message"
          request_body:
            session_id: "test_session_002"
            message: "Hello, what can you help me with?"
          expected_status: 200
          expected_fields:
            - "message_id"
            - "response"
            - "conversation_id"
            - "processing_time_ms"
        - name: "Rate Limiting Test"
          description: "Send multiple requests to test rate limiting"
          endpoint: "POST /v1/chat/message"
          repeat_count: 15
          expected_status: 429
          expected_response:
            error: "rate_limit_exceeded"

    phase_4_knowledge_base:
      description: Test knowledge base functionality
      prerequisites:
        - Vector database configured
        - File storage system available
        - Document processing pipeline functional
      endpoints_to_test:
        - POST /v1/knowledge/upload
        - GET /v1/knowledge/documents
        - POST /v1/knowledge/search
        - DELETE /v1/knowledge/documents/{id}
      test_scenarios:
        - name: "Document Upload"
          endpoint: "POST /v1/knowledge/upload"
          request_type: "multipart/form-data"
          request_body:
            file: "test_document.txt"
            category: "general"
            tags: ["test", "sample"]
          expected_status: 201
          expected_fields:
            - "document_id"
            - "processing_status"
        - name: "Knowledge Search"
          endpoint: "POST /v1/knowledge/search"
          request_body:
            query: "test information"
            search_type: "hybrid"
            max_results: 5
          expected_status: 200
          expected_fields:
            - "results"
            - "total_results"
            - "search_time_ms"

    phase_5_advanced_features:
      description: Test advanced features and integration scenarios
      prerequisites:
        - All previous phases passing
        - Analytics database configured
        - All external integrations functional
      endpoints_to_test:
        - GET /v1/config
        - PUT /v1/config
        - GET /v1/analytics/conversations
        - GET /v1/analytics/performance
        - GET /v1/analytics/costs
      test_scenarios:
        - name: "Configuration Retrieval"
          endpoint: "GET /v1/config"
          headers:
            Authorization: "Bearer <admin_token>"
          expected_status: 200
          expected_fields:
            - "rate_limits"
            - "models"
            - "authentication"
        - name: "Analytics Data"
          endpoint: "GET /v1/analytics/conversations?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z"
          headers:
            Authorization: "Bearer <admin_token>"
          expected_status: 200
          expected_fields:
            - "summary"
            - "metrics"

  test_data_setup:
    description: |
      Before running tests, ensure the following test data and configurations are in place:

    environment_variables:
      - name: "DATABASE_URL"
        description: "PostgreSQL connection string"
        example: "postgresql://user:pass@localhost:5432/chatbot_test"
      - name: "REDIS_URL"
        description: "Redis connection string"
        example: "redis://localhost:6379/0"
      - name: "OPENAI_API_KEY"
        description: "OpenAI API key for LLM integration"
        required: true
      - name: "TWILIO_ACCOUNT_SID"
        description: "Twilio account SID for SMS"
        required_for: "SMS authentication"
      - name: "SENDGRID_API_KEY"
        description: "SendGrid API key for email"
        required_for: "Email authentication"

    test_documents:
      - filename: "test_document.txt"
        content: "This is a test document for knowledge base testing. It contains sample information about testing procedures."
        category: "general"
        tags: ["test", "sample", "documentation"]
      - filename: "network_guide.pdf"
        description: "Sample PDF document for document processing tests"
        category: "technical"
        tags: ["network", "troubleshooting"]

    test_users:
      - email: "test@example.com"
        description: "Primary test user for authentication flows"
      - phone: "+1234567890"
        description: "Test phone number for SMS authentication"

  automated_test_scripts:
    description: |
      Example test scripts for automated testing of the API endpoints.

    bash_script_example: |
      #!/bin/bash
      # Basic API connectivity test
      
      BASE_URL="http://localhost:8000/api"
      
      echo "Testing health endpoint..."
      curl -s "$BASE_URL/v1/health" | jq .
      
      echo "Testing authentication flow..."
      AUTH_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/auth/request-token" \
        -H "Content-Type: application/json" \
        -d '{"session_id":"test_001","contact_method":"email","contact_value":"test@example.com"}')
      
      echo "Auth response: $AUTH_RESPONSE"
      
      echo "Testing basic chat..."
      CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/chat/message" \
        -H "Content-Type: application/json" \
        -d '{"session_id":"test_002","message":"Hello world"}')
      
      echo "Chat response: $CHAT_RESPONSE"

    python_script_example: |
      import requests
      import json
      import time
      
      BASE_URL = "http://localhost:8000/api"
      
      def test_health():
          """Test health endpoint"""
          response = requests.get(f"{BASE_URL}/v1/health")
          assert response.status_code == 200
          data = response.json()
          assert data["status"] in ["healthy", "degraded"]
          print("✅ Health check passed")
      
      def test_authentication():
          """Test authentication flow"""
          # Request OTP
          auth_request = {
              "session_id": "test_python_001",
              "contact_method": "email", 
              "contact_value": "test@example.com"
          }
          
          response = requests.post(
              f"{BASE_URL}/v1/auth/request-token",
              json=auth_request
          )
          assert response.status_code == 200
          data = response.json()
          assert data["success"] == True
          print("✅ OTP request passed")
      
      def test_chat():
          """Test basic chat functionality"""
          chat_request = {
              "session_id": "test_python_002",
              "message": "What is 2+2?"
          }
          
          response = requests.post(
              f"{BASE_URL}/v1/chat/message",
              json=chat_request
          )
          assert response.status_code == 200
          data = response.json()
          assert "response" in data
          assert "message_id" in data
          print("✅ Basic chat passed")
      
      if __name__ == "__main__":
          test_health()
          test_authentication()
          test_chat()
          print("🎉 All tests passed!")

  troubleshooting:
    common_issues:
      - issue: "Health check returns 503"
        cause: "Database or Redis connection failure"
        solution: "Check database connectivity and Redis service status"
      - issue: "Authentication endpoints timeout"
        cause: "SMS/Email provider not configured"
        solution: "Verify Twilio/SendGrid credentials and network connectivity"
      - issue: "Chat responses are slow"
        cause: "LLM provider API issues or rate limiting"
        solution: "Check LLM provider status and API key validity"
      - issue: "Knowledge base upload fails"
        cause: "Vector database not configured or file storage issues"
        solution: "Verify vector database connection and file storage permissions"

# ==========================================
# POSTMAN COLLECTION EXPORT
# ==========================================

x-postman-collection:
  info:
    name: "Chatbot Platform API"
    description: "Complete API collection for testing the chatbot platform"
    version: "1.0.0"
  
  variables:
    - key: "base_url"
      value: "http://localhost:8000/api"
      description: "API base URL"
    - key: "session_id"
      value: "{{$randomUUID}}"
      description: "Unique session identifier"
    - key: "admin_token"
      value: ""
      description: "Admin authentication token"
  
  folders:
    - name: "Health & Status"
      requests:
        - name: "Health Check"
          method: "GET"
          url: "{{base_url}}/v1/health"
        - name: "System Metrics"
          method: "GET"
          url: "{{base_url}}/v1/metrics"
    
    - name: "Authentication"
      requests:
        - name: "Request OTP (Email)"
          method: "POST"
          url: "{{base_url}}/v1/auth/request-token"
          body:
            session_id: "{{session_id}}"
            contact_method: "email"
            contact_value: "test@example.com"
        - name: "Verify OTP"
          method: "POST"
          url: "{{base_url}}/v1/auth/verify-token"
          body:
            session_id: "{{session_id}}"
            token: "123456"
        - name: "Check Auth Status"
          method: "GET"
          url: "{{base_url}}/v1/auth/status?session_id={{session_id}}"
    
    - name: "Chat"
      requests:
        - name: "Send Basic Message"
          method: "POST"
          url: "{{base_url}}/v1/chat/message"
          body:
            session_id: "{{session_id}}"
            message: "Hello, how can you help me today?"
        - name: "Send Authenticated Query"
          method: "POST"
          url: "{{base_url}}/v1/chat/message"
          headers:
            Authorization: "Bearer {{auth_token}}"
          body:
            session_id: "{{session_id}}"
            message: "Can you help me with my account settings?"
    
    - name: "Knowledge Base"
      requests:
        - name: "Upload Document"
          method: "POST"
          url: "{{base_url}}/v1/knowledge/upload"
          headers:
            Authorization: "Bearer {{admin_token}}"
          body:
            # Note: In Postman, this would be a form-data request
            # with file upload capability
        - name: "Search Knowledge Base"
          method: "POST"
          url: "{{base_url}}/v1/knowledge/search"
          body:
            query: "password reset"
            search_type: "hybrid"
            max_results: 10
        - name: "List Documents"
          method: "GET"
          url: "{{base_url}}/v1/knowledge/documents?limit=20"
          headers:
            Authorization: "Bearer {{admin_token}}"

---

## Testing Implementation Plan

### Phase 1: Basic Infrastructure (Week 1)
1. **Health Endpoints**: Verify API server startup and basic connectivity
2. **Database Connectivity**: Test PostgreSQL connection and basic queries
3. **Redis Connectivity**: Verify cache and session storage functionality
4. **Environment Configuration**: Validate all required environment variables

### Phase 2: Authentication System (Week 2)
1. **OTP Generation**: Test SMS and email OTP delivery
2. **OTP Verification**: Validate token verification logic
3. **Session Management**: Test JWT creation, validation, and expiration
4. **Rate Limiting**: Verify authentication rate limiting works correctly

### Phase 3: Core Chat Pipeline (Week 3)
1. **Basic Message Processing**: Test simple queries without authentication
2. **Rate Limiting**: Verify per-user and global rate limits
3. **Model Integration**: Test LLM provider connectivity and responses
4. **Error Handling**: Test various error scenarios and responses

### Phase 4: Advanced Features (Week 4)
1. **Knowledge Base**: Test document upload, processing, and search
2. **Semantic Caching**: Verify cache hit/miss behavior
3. **Authentication Requirements**: Test conditional authentication flow
4. **Analytics**: Verify metrics collection and reporting

### Phase 5: Performance & Production (Week 5)
1. **Load Testing**: Test system under various load conditions
2. **Integration Testing**: End-to-end workflow testing
3. **Security Testing**: Penetration testing and vulnerability assessment
4. **Documentation**: Finalize API documentation and deployment guides

This comprehensive OpenAPI specification provides detailed documentation for all API endpoints, complete with examples, testing guidance, and implementation details. The specification is designed to support incremental backend testing and serves as both documentation and a testing blueprint for the chatbot platform.
```

## Summary

This OpenAPI specification provides:

1. **Complete API Documentation**: All endpoints with detailed descriptions, parameters, and responses
2. **Authentication Flow**: Comprehensive OTP-based authentication system
3. **Chat Pipeline**: Intelligent message processing with caching and routing
4. **Knowledge Base**: RAG functionality with document management
5. **Analytics**: Comprehensive metrics and reporting capabilities
6. **Testing Guide**: Incremental testing approach with specific test scenarios
7. **Error Handling**: Detailed error responses and troubleshooting guidance

The specification is designed to facilitate incremental backend testing, starting with basic connectivity and progressing through authentication, chat functionality, knowledge base features, and advanced analytics. Each phase includes specific test scenarios and expected responses to validate backend functionality systematically.