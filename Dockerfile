# ===============================================
# Chatbot Platform Core - Dockerfile
# ===============================================
# Multi-stage build for optimized production images

# ===============================================
# Base Python Image
# ===============================================
FROM python:3.11-slim-bookworm as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    gcc \
    g++ \
    make \
    # For PDF processing
    poppler-utils \
    # For image processing and OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # For document processing
    libreoffice \
    # Network tools
    curl \
    wget \
    # System utilities
    procps \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Create application directories
RUN mkdir -p /app /app/logs /app/uploads /app/temp \
    && chown -R appuser:appuser /app

# ===============================================
# Development Stage
# ===============================================
FROM base as development

# Copy dependency files
COPY pyproject.toml /app/
WORKDIR /app

# Copy source code (needed for -e install)
COPY --chown=appuser:appuser . /app/

# Install development dependencies with editable install
RUN pip install --no-cache-dir -e ".[dev]"

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/temp /app/static \
    && chown -R appuser:appuser /app

# Switch to application user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# ===============================================
# Production Dependencies Stage
# ===============================================
FROM base as production-deps

# Copy dependency files
COPY pyproject.toml /app/
WORKDIR /app

# Copy source code (needed for package install)
COPY app/ /app/app/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/

# Install only production dependencies
RUN pip install --no-cache-dir ".[production]"

# ===============================================
# Production Stage
# ===============================================
FROM python:3.11-slim-bookworm as production

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production \
    WORKERS=4 \
    WORKER_CLASS=uvicorn.workers.UvicornWorker \
    WORKER_CONNECTIONS=1000 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100 \
    TIMEOUT=30 \
    KEEP_ALIVE=2

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime libraries
    libpq5 \
    # For PDF processing
    poppler-utils \
    # For image processing and OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # For document processing
    libreoffice \
    # Network tools for health checks
    curl \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Create application directories
RUN mkdir -p /app /app/logs /app/uploads /app/temp /app/static \
    && chown -R appuser:appuser /app

# Copy Python dependencies from production-deps stage
COPY --from=production-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=production-deps /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser alembic/ /app/alembic/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser alembic.ini /app/
COPY --chown=appuser:appuser pyproject.toml /app/

# Create gunicorn configuration
RUN cat > /app/gunicorn.conf.py << 'EOF'
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')
worker_connections = int(os.getenv('WORKER_CONNECTIONS', '1000'))
max_requests = int(os.getenv('MAX_REQUESTS', '1000'))
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', '100'))
timeout = int(os.getenv('TIMEOUT', '30'))
keepalive = int(os.getenv('KEEP_ALIVE', '2'))

# Restart workers
preload_app = True
max_worker_memory = 200 * 1024 * 1024  # 200MB

# Logging
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'chatbot-platform-core'

# Server mechanics
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = 1000
group = 1000
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None
EOF

# Fix ownership
RUN chown appuser:appuser /app/gunicorn.conf.py

# Switch to application user
USER appuser

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app.main:app"]

# ===============================================
# Testing Stage
# ===============================================
FROM development as testing

# Copy test files
COPY --chown=appuser:appuser tests/ /app/tests/

# Switch to application user
USER appuser

# Default command for testing
CMD ["pytest", "-v", "--cov=app", "--cov-report=html", "--cov-report=term-missing"]

# ===============================================
# Migration Stage
# ===============================================
FROM production-deps as migration

# Create application user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && chown -R appuser:appuser /app

# Switch to application user
USER appuser

# Set working directory
WORKDIR /app

# Default command for migrations
CMD ["alembic", "upgrade", "head"]

# ===============================================
# Build Arguments and Labels
# ===============================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add labels
LABEL maintainer="Chatbot Platform Team <team@chatbot-platform.com>" \
    org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.name="chatbot-platform-core" \
    org.label-schema.description="Enterprise-grade chatbot platform for SMEs" \
    org.label-schema.url="https://github.com/your-org/chatbot-platform-core" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/your-org/chatbot-platform-core" \
    org.label-schema.version=$VERSION \
    org.label-schema.schema-version="1.0"