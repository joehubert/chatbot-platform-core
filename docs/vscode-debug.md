# VS Code Debugging Setup for Chatbot Platform

This guide will help you set up VS Code to debug API calls in your Python FastAPI chatbot platform project.

## Prerequisites

- VS Code installed
- Python extension for VS Code installed
- Your project environment set up with dependencies installed
- Docker and Docker Compose running (for database services)

## 1. VS Code Configuration Files

### `.vscode/launch.json`

Create a `.vscode/launch.json` file in your project root with the following configurations:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug FastAPI App",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "app.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--log-level",
                "debug"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "ENVIRONMENT": "development",
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG"
            },
            "envFile": "${workspaceFolder}/.env",
            "console": "integratedTerminal",
            "justMyCode": false,
            "subProcess": true
        },
        {
            "name": "Debug FastAPI - No Reload",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--log-level",
                "debug"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "ENVIRONMENT": "development",
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG"
            },
            "envFile": "${workspaceFolder}/.env",
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Debug Single Test",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "${file}",
                "-v",
                "-s"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "ENVIRONMENT": "test"
            },
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Debug Test Suite",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v",
                "-s",
                "--tb=short"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "ENVIRONMENT": "test"
            },
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

### `.vscode/settings.json`

Create a `.vscode/settings.json` file for project-specific settings:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        "*.egg-info": true
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.envFile": "${workspaceFolder}/.env"
}
```

### `.vscode/tasks.json`

Create tasks for common development operations:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Start Services",
            "type": "shell",
            "command": "docker-compose",
            "args": ["up", "-d", "postgres", "redis", "chroma"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": []
        },
        {
            "label": "Stop Services",
            "type": "shell",
            "command": "docker-compose",
            "args": ["down"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": []
        },
        {
            "label": "Run Database Migrations",
            "type": "shell",
            "command": "alembic",
            "args": ["upgrade", "head"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "dependsOn": "Start Services",
            "problemMatcher": []
        },
        {
            "label": "Install Dependencies",
            "type": "shell",
            "command": "pip",
            "args": ["install", "-r", "requirements.txt"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": []
        }
    ]
}
```

## 2. Environment Setup

### `.env` File

Ensure your `.env` file is configured for debugging:

```bash
# Development Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database (when running locally)
DATABASE_URL=postgresql://chatbot:password@localhost:5432/chatbot_db
REDIS_URL=redis://localhost:6379/0

# Add your actual API keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Security (use development keys)
SECRET_KEY=development-secret-key-here
ENCRYPTION_KEY=development-encryption-key-here

# Enable detailed logging
ENABLE_METRICS=true
```

## 3. Debugging Workflow

### Step 1: Start Supporting Services

Before debugging, start the required services:

1. **Using VS Code Tasks**: Press `Ctrl+Shift+P` → "Tasks: Run Task" → "Start Services"
2. **Or using terminal**:
   ```bash
   docker-compose up -d postgres redis chroma
   ```

### Step 2: Set Breakpoints

1. Open the Python file where you want to debug (e.g., `app/api/v1/endpoints/chat.py`)
2. Click in the left margin next to line numbers to set breakpoints
3. Common places to set breakpoints:
   - API endpoint entry points
   - Pipeline processing steps
   - Database operations
   - Error handling blocks

### Step 3: Start Debugging

1. Go to the Debug view (`Ctrl+Shift+D`)
2. Select "Debug FastAPI App" from the dropdown
3. Press `F5` or click the green play button
4. The debugger will start and show output in the integrated terminal

### Step 4: Make API Calls with Postman

With the debugger running, use Postman to make calls to your API:

**Example Chat Request**:
- **Method**: POST
- **URL**: `http://localhost:8000/api/v1/chat/message`
- **Headers**: `Content-Type: application/json`
- **Body**:
  ```json
  {
    "message": "Hello, can you help me?",
    "session_id": "debug-session-123"
  }
  ```

**Example Knowledge Upload**:
- **Method**: POST
- **URL**: `http://localhost:8000/api/v1/knowledge/upload`
- **Headers**: `Content-Type: multipart/form-data`
- **Body**: Form data with file upload

## 4. Key Debugging Locations

### API Endpoints (`app/api/v1/endpoints/`)

Set breakpoints in these files to debug specific functionality:

- **`chat.py`**: Chat message processing
- **`auth.py`**: Authentication flows
- **`knowledge.py`**: Document upload and management
- **`config.py`**: Configuration management

### Pipeline Components (`app/core/pipeline/`)

Debug the LangGraph processing pipeline:

- **`rate_limiter.py`**: Rate limiting logic
- **`relevance_checker.py`**: Query relevance analysis
- **`semantic_cache.py`**: Cache hit/miss logic
- **`model_router.py`**: Model selection logic
- **`rag_engine.py`**: RAG processing

### Database Operations (`app/models/`)

Debug database interactions:

- **`conversation.py`**: Chat session storage
- **`document.py`**: Knowledge base operations
- **`user.py`**: User management

## 5. Debugging Tips

### Inspect Variables

When stopped at a breakpoint:
- Hover over variables to see their values
- Use the Variables panel to explore object properties
- Use the Debug Console to execute Python expressions

### Step Through Code

- **F10**: Step over (execute current line)
- **F11**: Step into (enter function calls)
- **Shift+F11**: Step out (exit current function)
- **F5**: Continue execution

### Debug Console Commands

Use the Debug Console to inspect state:

```python
# Check request data
print(request.json())

# Inspect pipeline state
print(pipeline_state)

# Check database connection
await db.execute("SELECT 1")

# Examine environment variables
import os; print(os.environ.get("OPENAI_API_KEY", "Not set"))
```

### Conditional Breakpoints

Right-click on a breakpoint to set conditions:

```python
# Break only for specific sessions
session_id == "debug-session-123"

# Break only on errors
"error" in response_data

# Break for specific message content
"help" in message.lower()
```

## 6. Common Debugging Scenarios

### Debug Chat Pipeline

1. Set breakpoint in `app/api/v1/endpoints/chat.py` at the message processing function
2. Set breakpoints in pipeline components you want to inspect
3. Send chat request from Postman
4. Step through the entire processing flow

### Debug Authentication

1. Set breakpoints in `app/api/v1/endpoints/auth.py`
2. Set breakpoints in `app/core/pipeline/auth_handler.py`
3. Test login/logout flows with Postman

### Debug Document Processing

1. Set breakpoints in `app/api/v1/endpoints/knowledge.py`
2. Set breakpoints in document processing pipeline
3. Upload a test document via Postman

### Debug Database Issues

1. Set breakpoints in model files (`app/models/`)
2. Use the Debug Console to execute database queries
3. Inspect SQLAlchemy session state

## 7. Performance Debugging

### Memory Usage

```python
# In Debug Console
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

### Request Timing

The app includes `RequestLoggingMiddleware` that logs request timing. Check the integrated terminal for performance data.

### Database Query Performance

Set breakpoints before/after database operations and check execution time in logs.

## 8. Troubleshooting

### Common Issues

1. **"Module not found" errors**: Check PYTHONPATH in launch.json
2. **Database connection errors**: Ensure PostgreSQL is running via Docker
3. **Redis connection errors**: Ensure Redis is running via Docker
4. **API key errors**: Check .env file configuration
5. **Import errors**: Ensure virtual environment is activated

### Debug Configuration Issues

If debugging doesn't work:
1. Check that the Python interpreter is correctly set
2. Verify .env file is loaded
3. Ensure dependencies are installed in the correct virtual environment
4. Check that ports 8000, 5432, and 6379 are available

## 9. Additional Tools

### VS Code Extensions

Install these helpful extensions:
- **Python** (Microsoft)
- **Python Docstring Generator** 
- **REST Client** (alternative to Postman)
- **Docker** (for container management)
- **SQLTools** (database management)

### REST Client Alternative

Create a `requests.http` file for testing without Postman:

```http
### Health Check
GET http://localhost:8000/health

### Chat Message
POST http://localhost:8000/api/v1/chat/message
Content-Type: application/json

{
  "message": "Hello, world!",
  "session_id": "debug-session"
}

### Upload Document
POST http://localhost:8000/api/v1/knowledge/upload
Content-Type: multipart/form-data; boundary=boundary

--boundary
Content-Disposition: form-data; name="file"; filename="test.txt"
Content-Type: text/plain

This is a test document for debugging.
--boundary--
```

This setup will give you comprehensive debugging capabilities for your FastAPI chatbot platform, allowing you to step through code, inspect variables, and trace the entire request lifecycle from API call to response.