# Backend Testing Implementation Guide

This guide provides step-by-step instructions for systematically testing the chatbot platform backend functionality across 5 progressive phases.

## Prerequisites

Before starting any testing phase, ensure you have:

- **Docker & Docker Compose** installed and running
- **Access to the running containers** (API, database, Redis, etc.)
- **API testing tool** (curl, Postman, or similar)
- **Admin credentials** for protected endpoints
- **Test data** (sample documents, test email/phone numbers)

## Phase 1: Basic Infrastructure (Week 1)

### 1.1 Health Endpoints - Verify API Server Startup

**Objective**: Confirm the API server is running and responding to requests.

**Steps**:
```bash
# Test basic health endpoint
curl -X GET http://localhost:3000/api/v1/health

# Expected Response (200 OK):
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api_server": {
      "status": "healthy",
      "response_time_ms": 5
    }
  }
}
```

**Validation**:
- ✅ Response status is 200
- ✅ JSON response contains "status" field
- ✅ Response time is reasonable (<1000ms)

### 1.2 Database Connectivity - Test PostgreSQL Connection

**Objective**: Verify the API can connect to and query the PostgreSQL database.

**Steps**:
```bash
# Check database component in health endpoint
curl -X GET http://localhost:3000/api/v1/health | jq '.components.database'

# Expected database status:
{
  "status": "healthy",
  "response_time_ms": 15,
  "connections_active": 8,
  "connections_max": 100
}
```

**Direct Database Testing**:
```bash
# Connect directly to database (if accessible)
docker exec -it chatbot-platform-postgres psql -U chatbot_user -d chatbot_platform

# Run basic query
SELECT COUNT(*) FROM conversations;

# list databases
\list

# list tables in current database
\dt
\dt+

# describe a table
\d <table_name>
```

**Validation**:
- ✅ Database status shows "healthy"
- ✅ Connection count is reasonable
- ✅ Basic queries execute without errors

### 1.3 Redis Connectivity - Verify Cache and Session Storage

**Objective**: Ensure Redis is accessible for caching and session management.

**Steps**:
```bash
# Check Redis component health
curl -X GET http://localhost:3000/api/v1/health | jq '.components.redis'

# Expected Redis status:
{
  "status": "healthy",
  "response_time_ms": 2,
  "memory_usage_percent": 45
}
```

**Direct Redis Testing**:
```bash
# Connect to Redis container
docker exec -it chatbot-platform-redis redis-cli

# Simple ping test
PING
# expect: PONG

# Test basic operations
SET test_key "test_value"
GET test_key
DEL test_key
```

**Validation**:
- ✅ Redis status shows "healthy"
- ✅ Memory usage is within acceptable limits (<80%)
- ✅ Basic Redis operations work

### 1.4 Environment Configuration - Validate Required Variables

**Objective**: Ensure all necessary environment variables are properly configured.

**Steps**:
```bash
# Check configuration endpoint (requires admin auth)
curl -X GET http://localhost:3000/api/v1/config \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Check metrics for system information
curl -X GET http://localhost:3000/api/v1/metrics
```

**Environment Variables Checklist**:
```bash
# Check these variables are set in your containers
echo $DATABASE_URL
echo $REDIS_URL
echo $OPENAI_API_KEY
echo $TWILIO_ACCOUNT_SID  # If using SMS
echo $SENDGRID_API_KEY    # If using email
```

**Validation**:
- ✅ All required environment variables are set
- ✅ Configuration endpoint returns expected structure
- ✅ No missing dependency errors in logs

## Phase 2: Authentication System (Week 2)

### 2.1 OTP Generation - Test SMS and Email Delivery

**Objective**: Verify OTP tokens can be generated and delivered via SMS/email.

**Steps**:
```bash
# Test email OTP request
curl -X POST http://localhost:3000/api/v1/auth/request-token \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_001",
    "contact_method": "email",
    "contact_value": "test@yourdomain.com"
  }'

# Expected Response (200 OK):
{
  "success": true,
  "message": "Authentication code sent to your email",
  "expires_in": 300,
  "retry_after": null
}
```

```bash
# Test SMS OTP request
curl -X POST http://localhost:3000/api/v1/auth/request-token \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_002",
    "contact_method": "sms",
    "contact_value": "+1234567890"
  }'
```

**Validation**:
- ✅ Response indicates success
- ✅ OTP email/SMS is actually received
- ✅ OTP contains valid token (4-10 characters)

### 2.2 OTP Verification - Validate Token Verification Logic

**Objective**: Test that valid OTP tokens are accepted and invalid ones rejected.

**Steps**:
```bash
# Test valid OTP verification
curl -X POST http://localhost:3000/api/v1/auth/verify-token \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_001",
    "token": "ACTUAL_OTP_FROM_EMAIL"
  }'

# Expected Response (200 OK):
{
  "success": true,
  "message": "Authentication successful",
  "session_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_at": "2024-01-15T11:30:00Z",
  "user_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

```bash
# Test invalid OTP
curl -X POST http://localhost:3000/api/v1/auth/verify-token \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_001",
    "token": "WRONG123"
  }'

# Expected Response (400 Bad Request):
{
  "success": false,
  "message": "Invalid or expired token"
}
```

**Validation**:
- ✅ Valid OTP returns session token
- ✅ Invalid OTP returns error
- ✅ Expired OTP is rejected
- ✅ OTP can only be used once

### 2.3 Session Management - Test JWT Creation and Validation

**Objective**: Verify JWT tokens work for authenticated endpoints.

**Steps**:
```bash
# Test authentication status with valid token
curl -X GET "http://localhost:3000/api/v1/auth/status?session_id=test_session_001" \
  -H "Authorization: Bearer YOUR_SESSION_TOKEN"

# Expected Response (200 OK):
{
  "session_id": "test_session_001",
  "authenticated": true,
  "expires_at": "2024-01-15T11:30:00Z",
  "time_remaining": 3540,
  "requires_renewal": false
}
```

```bash
# Test token refresh
curl -X POST http://localhost:3000/api/v1/auth/refresh \
  -H "Authorization: Bearer YOUR_SESSION_TOKEN"
```

**Validation**:
- ✅ Valid tokens allow access to protected endpoints
- ✅ Invalid/expired tokens are rejected
- ✅ Token refresh works when eligible
- ✅ Session expiration is enforced

### 2.4 Rate Limiting - Verify Authentication Rate Limiting

**Objective**: Ensure authentication endpoints have appropriate rate limiting.

**Steps**:
```bash
# Script to test rate limiting
for i in {1..10}; do
  echo "Request $i:"
  curl -X POST http://localhost:3000/api/v1/auth/request-token \
    -H "Content-Type: application/json" \
    -d '{
      "session_id": "rate_test_'$i'",
      "contact_method": "email",
      "contact_value": "test@yourdomain.com"
    }' \
    -w "Status: %{http_code}\n"
  sleep 1
done
```

**Expected Behavior**:
- First few requests: 200 OK
- After limit: 429 Too Many Requests
- Response includes retry-after header

**Validation**:
- ✅ Rate limiting activates after configured threshold
- ✅ 429 response includes retry information
- ✅ Rate limit resets after time window

## Phase 3: Core Chat Pipeline (Week 3)

### 3.1 Basic Message Processing - Test Simple Queries

**Objective**: Verify the chat pipeline processes basic messages without authentication.

**Steps**:
```bash
# Test simple chat message
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "chat_test_001",
    "message": "Hello, what can you help me with?"
  }'

# Expected Response (200 OK):
{
  "message_id": "msg_123e4567",
  "response": "Hello! I can help you with...",
  "conversation_id": "conv_123e4567",
  "session_id": "chat_test_001",
  "cached": false,
  "processing_time_ms": 1250,
  "requires_authentication": false,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Validation**:
- ✅ Response contains valid message structure
- ✅ Processing time is reasonable (<5000ms)
- ✅ Response is coherent and relevant
- ✅ Conversation ID is generated

### 3.2 Rate Limiting - Verify Chat Rate Limits

**Objective**: Test per-user and global rate limiting for chat endpoints.

**Steps**:
```bash
# Test user rate limiting
for i in {1..20}; do
  curl -X POST http://localhost:3000/api/v1/chat/message \
    -H "Content-Type: application/json" \
    -d '{
      "session_id": "rate_test_chat",
      "message": "Test message '$i'"
    }' \
    -w "Status: %{http_code}\n"
done
```

**Validation**:
- ✅ Rate limiting kicks in after configured limit
- ✅ Different users have separate rate limits
- ✅ Rate limits reset properly after time window

### 3.3 Model Integration - Test LLM Provider Connectivity

**Objective**: Verify LLM providers are working and responses are generated.

**Steps**:
```bash
# Test different types of queries
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "model_test_001",
    "message": "What is 2 + 2?"
  }'

# Test complex query (should route to more powerful model)
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "model_test_002",
    "message": "Explain the theory of relativity and its implications for modern physics."
  }'
```

**Check Model Usage in Response**:
```bash
# Look for model_used field in response
jq '.model_used' response.json
```

**Validation**:
- ✅ Simple queries get routed to basic model
- ✅ Complex queries get routed to advanced model
- ✅ Model responses are appropriate for query complexity
- ✅ Model failover works if primary fails

### 3.4 Error Handling - Test Various Error Scenarios

**Objective**: Ensure the system handles errors gracefully.

**Steps**:
```bash
# Test empty message
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "error_test_001",
    "message": ""
  }'

# Test missing required fields
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Test without session_id"
  }'

# Test malformed JSON
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"invalid": json}'
```

**Validation**:
- ✅ Appropriate error codes returned (400, 422, etc.)
- ✅ Error messages are descriptive
- ✅ System doesn't crash on invalid input
- ✅ Request IDs provided for debugging

## Phase 4: Advanced Features (Week 4)

### 4.1 Knowledge Base - Test Document Management

**Objective**: Verify document upload, processing, and search functionality.

**Steps**:
```bash
# Test document upload
curl -X POST http://localhost:3000/api/v1/knowledge/upload \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -F "file=@test_document.txt" \
  -F "category=general" \
  -F "tags=test,sample"

# Expected Response (201 Created):
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "test_document.txt",
  "processing_status": "completed",
  "chunks_created": 5
}
```

```bash
# Test knowledge base search
curl -X POST http://localhost:3000/api/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test information",
    "search_type": "hybrid",
    "max_results": 5
  }'
```

**Validation**:
- ✅ Documents upload successfully
- ✅ Processing completes without errors
- ✅ Search returns relevant results
- ✅ Relevance scores are reasonable

### 4.2 Semantic Caching - Verify Cache Behavior

**Objective**: Test that similar queries are served from cache.

**Steps**:
```bash
# Send initial query
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "cache_test_001",
    "message": "What are your business hours?"
  }' > first_response.json

# Send similar query
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "cache_test_002",
    "message": "What time are you open?"
  }' > second_response.json

# Check if second response was cached
jq '.cached' second_response.json
jq '.processing_time_ms' first_response.json second_response.json
```

**Validation**:
- ✅ Similar queries return cached responses
- ✅ Cache hits have much faster response times
- ✅ Cache field indicates hit/miss status
- ✅ Cache similarity scores are reasonable

### 4.3 Authentication Requirements - Test Conditional Auth

**Objective**: Verify that sensitive queries require authentication.

**Steps**:
```bash
# Test query that should require authentication
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "auth_test_001",
    "message": "Can you help me reset my password?"
  }'

# Expected response indicates auth required:
{
  "response": "I can help with that. Please verify your identity first.",
  "requires_authentication": true,
  "auth_methods": ["sms", "email"]
}
```

```bash
# Test same query with authentication
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SESSION_TOKEN" \
  -d '{
    "session_id": "auth_test_001",
    "message": "Can you help me reset my password?"
  }'
```

**Validation**:
- ✅ Sensitive queries trigger authentication requirement
- ✅ Authenticated queries provide full responses
- ✅ Authentication methods are clearly indicated
- ✅ System maintains conversation context through auth

### 4.4 Analytics - Verify Metrics Collection

**Objective**: Ensure analytics data is being collected and can be retrieved.

**Steps**:
```bash
# Test conversation analytics
curl -X GET "http://localhost:3000/api/v1/analytics/conversations?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Test performance analytics
curl -X GET http://localhost:3000/api/v1/analytics/performance \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Test cost analytics
curl -X GET http://localhost:3000/api/v1/analytics/costs \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

**Validation**:
- ✅ Analytics endpoints return data
- ✅ Metrics include expected fields
- ✅ Data appears accurate based on test activity
- ✅ Date filtering works correctly

## Phase 5: Performance & Production (Week 5)

### 5.1 Load Testing - Test Under Various Load Conditions

**Objective**: Verify system performance under concurrent load.

**Setup Load Testing Tool**:
```bash
# Install Apache Bench (ab) or use existing tool
sudo apt-get install apache2-utils

# Or use curl in parallel
```

**Steps**:
```bash
# Light load test - 10 concurrent requests
ab -n 100 -c 10 -T "application/json" \
  -p chat_payload.json \
  http://localhost:3000/api/v1/chat/message

# Medium load test - 50 concurrent requests
ab -n 500 -c 50 -T "application/json" \
  -p chat_payload.json \
  http://localhost:3000/api/v1/chat/message

# Monitor system resources during tests
docker stats
```

**Create chat_payload.json**:
```json
{
  "session_id": "load_test_session",
  "message": "What services do you offer?"
}
```

**Validation**:
- ✅ System handles concurrent requests without errors
- ✅ Response times remain reasonable under load
- ✅ No memory leaks or resource exhaustion
- ✅ Error rates stay within acceptable limits

### 5.2 Integration Testing - End-to-End Workflow Testing

**Objective**: Test complete user workflows from start to finish.

**Steps**:
```bash
# Complete authentication + chat workflow
echo "=== Starting Integration Test ==="

# Step 1: Request OTP
echo "Step 1: Requesting OTP..."
curl -X POST http://localhost:3000/api/v1/auth/request-token \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "integration_test",
    "contact_method": "email",
    "contact_value": "test@yourdomain.com"
  }'

# Step 2: Get OTP from email (manual step)
read -p "Enter OTP from email: " otp

# Step 3: Verify OTP
echo "Step 3: Verifying OTP..."
auth_response=$(curl -s -X POST http://localhost:3000/api/v1/auth/verify-token \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "integration_test",
    "token": "'$otp'"
  }')

# Extract token
token=$(echo $auth_response | jq -r '.session_token')

# Step 4: Send authenticated message
echo "Step 4: Sending authenticated message..."
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $token" \
  -d '{
    "session_id": "integration_test",
    "message": "Can you help me with my account?"
  }'
```

**Validation**:
- ✅ Complete workflow executes without errors
- ✅ State is maintained throughout the process
- ✅ All components work together correctly
- ✅ User experience is smooth and logical

### 5.3 Security Testing - Basic Penetration Testing

**Objective**: Identify common security vulnerabilities.

**Steps**:
```bash
# Test SQL injection attempts
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "security_test",
    "message": "Hello\"; DROP TABLE conversations; --"
  }'

# Test XSS attempts
curl -X POST http://localhost:3000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "security_test",
    "message": "<script>alert(\"xss\")</script>"
  }'

# Test authentication bypass attempts
curl -X GET http://localhost:3000/api/v1/config \
  -H "Authorization: Bearer invalid_token"

# Test rate limit bypass attempts
for i in {1..100}; do
  curl -X POST http://localhost:3000/api/v1/auth/request-token \
    -H "Content-Type: application/json" \
    -H "X-Forwarded-For: 192.168.1.$i" \
    -d '{
      "session_id": "bypass_test_'$i'",
      "contact_method": "email",
      "contact_value": "test'$i'@example.com"
    }' &
done
```

**Validation**:
- ✅ SQL injection attempts are blocked
- ✅ XSS content is properly sanitized
- ✅ Authentication is properly enforced
- ✅ Rate limiting cannot be easily bypassed
- ✅ Error messages don't leak sensitive information

### 5.4 Documentation - Finalize and Validate Documentation

**Objective**: Ensure all documentation is accurate and complete.

**Steps**:
1. **API Documentation Review**:
   ```bash
   # Verify OpenAPI spec matches actual API
   curl http://localhost:3000/docs
   curl http://localhost:3000/openapi.json
   ```

2. **Test All Examples**:
   - Go through OpenAPI spec examples
   - Verify each curl command works
   - Check response formats match documentation

3. **Deployment Guide Validation**:
   - Follow deployment steps on fresh environment
   - Document any missing steps or dependencies
   - Test environment variable configurations

4. **User Guide Testing**:
   - Have someone else follow the guides
   - Note any unclear instructions
   - Update based on feedback

**Validation**:
- ✅ API documentation is accurate and up-to-date
- ✅ All examples work as documented
- ✅ Deployment guides are complete and tested
- ✅ User guides are clear and comprehensive

## Success Criteria

Each phase is considered complete when:

- **All test steps pass** without errors
- **Performance metrics** meet expectations
- **Error handling** works correctly
- **Documentation** is updated with findings
- **Issues are logged** and prioritized for fixing

## Troubleshooting Quick Reference

**Common Issues and Solutions**:

| Issue | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| Health check fails | Service not running | Check Docker containers: `docker ps` |
| Database errors | Connection issues | Verify DATABASE_URL and network |
| Authentication fails | Missing credentials | Check SMS/email provider settings |
| Chat responses slow | LLM provider issues | Verify API keys and rate limits |
| Cache not working | Redis issues | Check Redis connection and memory |
| File upload fails | Storage/permissions | Check file storage configuration |

**Logging and Debugging**:
```bash
# View container logs
docker logs chatbot-api
docker logs chatbot-postgres
docker logs chatbot-redis

# Follow logs in real-time
docker logs -f chatbot-api

# Check system resources
docker stats
```

This testing guide provides a systematic approach to validating your chatbot platform backend functionality. Each phase builds upon the previous one, ensuring that foundational components are working before testing more complex features.