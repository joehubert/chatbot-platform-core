"""
Monitoring utilities for the chatbot platform.
Provides performance metrics, system monitoring, and health tracking.
"""

import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextvars import ContextVar
from app.utils.logging import get_logger
from app.core.redis import get_redis_client

logger = get_logger(__name__)

# Context variables for request tracking
request_start_time: ContextVar[Optional[float]] = ContextVar(
    "request_start_time", default=None
)
request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


@dataclass
class Metric:
    """Individual metric data point."""

    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class PerformanceStats:
    """Performance statistics for operations."""

    operation: str
    count: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    errors: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def avg_duration(self) -> float:
        """Calculate average duration."""
        return self.total_duration / self.count if self.count > 0 else 0.0

    def add_measurement(self, duration: float, success: bool = True):
        """Add a new measurement to the stats."""
        self.count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        if not success:
            self.errors += 1
        self.last_updated = datetime.now(timezone.utc)


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self, max_history: int = 1000):
        self.metrics: List[Metric] = []
        self.max_history = max_history
        self.performance_stats: Dict[str, PerformanceStats] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def record_metric(
        self, name: str, value: float, labels: Dict[str, str] = None, unit: str = ""
    ):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            unit=unit,
        )

        self.metrics.append(metric)

        # Keep only recent metrics
        if len(self.metrics) > self.max_history:
            self.metrics = self.metrics[-self.max_history :]

        logger.debug(
            f"Metric recorded: {name}={value}{unit}",
            extra={"metric_name": name, "metric_value": value, "metric_unit": unit},
        )

    def increment_counter(
        self, name: str, value: int = 1, labels: Dict[str, str] = None
    ):
        """Increment a counter metric."""
        self.counters[name] += value
        self.record_metric(name, self.counters[name], labels, "count")

    def set_gauge(
        self, name: str, value: float, labels: Dict[str, str] = None, unit: str = ""
    ):
        """Set a gauge metric."""
        self.gauges[name] = value
        self.record_metric(name, value, labels, unit)

    def record_histogram(
        self, name: str, value: float, labels: Dict[str, str] = None, unit: str = ""
    ):
        """Record a value in a histogram."""
        self.histograms[name].append(value)
        self.record_metric(name, value, labels, unit)

    def record_performance(self, operation: str, duration: float, success: bool = True):
        """Record performance metrics for an operation."""
        if operation not in self.performance_stats:
            self.performance_stats[operation] = PerformanceStats(operation)

        self.performance_stats[operation].add_measurement(duration, success)

        # Also record as histogram
        self.record_histogram(f"{operation}_duration", duration, unit="ms")

        if not success:
            self.increment_counter(f"{operation}_errors")

    def get_performance_stats(self, operation: str) -> Optional[PerformanceStats]:
        """Get performance statistics for an operation."""
        return self.performance_stats.get(operation)

    def get_all_performance_stats(self) -> Dict[str, PerformanceStats]:
        """Get all performance statistics."""
        return self.performance_stats.copy()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "performance_stats": {
                name: {
                    "count": stats.count,
                    "avg_duration_ms": round(stats.avg_duration, 2),
                    "min_duration_ms": round(stats.min_duration, 2)
                    if stats.min_duration != float("inf")
                    else 0,
                    "max_duration_ms": round(stats.max_duration, 2),
                    "error_rate": round(stats.errors / stats.count * 100, 2)
                    if stats.count > 0
                    else 0,
                    "last_updated": stats.last_updated.isoformat(),
                }
                for name, stats in self.performance_stats.items()
            },
            "histogram_summaries": {
                name: {
                    "count": len(values),
                    "avg": round(sum(values) / len(values), 2) if values else 0,
                    "min": round(min(values), 2) if values else 0,
                    "max": round(max(values), 2) if values else 0,
                }
                for name, values in self.histograms.items()
            },
        }

    def clear_old_metrics(self, max_age_hours: int = 24):
        """Clear metrics older than specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]


# Global metrics collector instance
metrics_collector = MetricsCollector()


class SystemMonitor:
    """Monitors system resources and health."""

    def __init__(self):
        self.start_time = time.time()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Network statistics
            network = psutil.net_io_counters()

            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                "cpu": {"percent": cpu_percent, "count": psutil.cpu_count()},
                "memory": {
                    "total_mb": round(memory.total / 1024 / 1024, 2),
                    "used_mb": round(memory.used / 1024 / 1024, 2),
                    "available_mb": round(memory.available / 1024 / 1024, 2),
                    "percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                    "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                    "percent": round(disk.used / disk.total * 100, 2),
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "process": {
                    "memory_rss_mb": round(process_memory.rss / 1024 / 1024, 2),
                    "memory_vms_mb": round(process_memory.vms / 1024 / 1024, 2),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                    "uptime_seconds": round(time.time() - self.start_time, 2),
                },
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return {}

    async def collect_and_store_metrics(self):
        """Collect system metrics and store them."""
        system_metrics = self.get_system_metrics()

        if system_metrics:
            # Store key metrics
            metrics_collector.set_gauge(
                "cpu_percent", system_metrics["cpu"]["percent"], unit="%"
            )
            metrics_collector.set_gauge(
                "memory_percent", system_metrics["memory"]["percent"], unit="%"
            )
            metrics_collector.set_gauge(
                "disk_percent", system_metrics["disk"]["percent"], unit="%"
            )
            metrics_collector.set_gauge(
                "process_memory_mb",
                system_metrics["process"]["memory_rss_mb"],
                unit="MB",
            )
            metrics_collector.set_gauge(
                "uptime_seconds", system_metrics["process"]["uptime_seconds"], unit="s"
            )


# Global system monitor instance
system_monitor = SystemMonitor()


class PerformanceMonitor:
    """Context manager for monitoring operation performance."""

    def __init__(self, operation: str, auto_record: bool = True):
        self.operation = operation
        self.auto_record = auto_record
        self.start_time = None
        self.end_time = None
        self.success = True
        self.error = None

    def __enter__(self):
        self.start_time = time.time()
        request_start_time.set(self.start_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.success = exc_type is None
        self.error = str(exc_val) if exc_val else None

        if self.auto_record:
            duration_ms = (self.end_time - self.start_time) * 1000
            metrics_collector.record_performance(
                self.operation, duration_ms, self.success
            )

            if not self.success:
                logger.error(
                    f"Operation failed: {self.operation}",
                    extra={
                        "operation": self.operation,
                        "duration_ms": round(duration_ms, 2),
                        "error": self.error,
                    },
                )

    @property
    def duration_ms(self) -> float:
        """Get operation duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


def monitor_performance(operation: str):
    """Decorator for monitoring function performance."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with PerformanceMonitor(operation):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with PerformanceMonitor(operation):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


class AlertManager:
    """Manages system alerts and notifications."""

    def __init__(self):
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_ms": 5000.0,
            "error_rate_percent": 5.0,
        }
        self.alert_cooldowns = {}  # Track when alerts were last sent
        self.cooldown_period = 300  # 5 minutes in seconds

    def check_thresholds(self) -> List[Dict[str, Any]]:
        """Check all thresholds and return alerts."""
        alerts = []
        current_time = time.time()

        # Check system metrics
        system_metrics = system_monitor.get_system_metrics()

        if system_metrics:
            # CPU alert
            cpu_percent = system_metrics["cpu"]["percent"]
            if cpu_percent > self.thresholds["cpu_percent"]:
                if self._should_send_alert("cpu_high", current_time):
                    alerts.append(
                        {
                            "type": "cpu_high",
                            "severity": "warning",
                            "message": f"High CPU usage: {cpu_percent}%",
                            "value": cpu_percent,
                            "threshold": self.thresholds["cpu_percent"],
                        }
                    )

            # Memory alert
            memory_percent = system_metrics["memory"]["percent"]
            if memory_percent > self.thresholds["memory_percent"]:
                if self._should_send_alert("memory_high", current_time):
                    alerts.append(
                        {
                            "type": "memory_high",
                            "severity": "warning",
                            "message": f"High memory usage: {memory_percent}%",
                            "value": memory_percent,
                            "threshold": self.thresholds["memory_percent"],
                        }
                    )

            # Disk alert
            disk_percent = system_metrics["disk"]["percent"]
            if disk_percent > self.thresholds["disk_percent"]:
                if self._should_send_alert("disk_high", current_time):
                    alerts.append(
                        {
                            "type": "disk_high",
                            "severity": "critical",
                            "message": f"High disk usage: {disk_percent}%",
                            "value": disk_percent,
                            "threshold": self.thresholds["disk_percent"],
                        }
                    )

        # Check performance metrics
        performance_stats = metrics_collector.get_all_performance_stats()

        for operation, stats in performance_stats.items():
            # Response time alert
            if stats.avg_duration > self.thresholds["response_time_ms"]:
                if self._should_send_alert(
                    f"response_time_high_{operation}", current_time
                ):
                    alerts.append(
                        {
                            "type": "response_time_high",
                            "severity": "warning",
                            "message": f"High response time for {operation}: {stats.avg_duration:.2f}ms",
                            "operation": operation,
                            "value": stats.avg_duration,
                            "threshold": self.thresholds["response_time_ms"],
                        }
                    )

            # Error rate alert
            if stats.count > 0:
                error_rate = (stats.errors / stats.count) * 100
                if error_rate > self.thresholds["error_rate_percent"]:
                    if self._should_send_alert(
                        f"error_rate_high_{operation}", current_time
                    ):
                        alerts.append(
                            {
                                "type": "error_rate_high",
                                "severity": "critical",
                                "message": f"High error rate for {operation}: {error_rate:.2f}%",
                                "operation": operation,
                                "value": error_rate,
                                "threshold": self.thresholds["error_rate_percent"],
                            }
                        )

        return alerts

    def _should_send_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_alert_time = self.alert_cooldowns.get(alert_key, 0)
        if current_time - last_alert_time > self.cooldown_period:
            self.alert_cooldowns[alert_key] = current_time
            return True
        return False

    async def process_alerts(self):
        """Process and handle any active alerts."""
        alerts = self.check_thresholds()

        for alert in alerts:
            logger.warning(
                f"Alert triggered: {alert['message']}",
                extra={
                    "alert_type": alert["type"],
                    "alert_severity": alert["severity"],
                    "alert_value": alert["value"],
                    "alert_threshold": alert["threshold"],
                },
            )

            # Store alert in Redis for dashboard display
            await self._store_alert(alert)

    async def _store_alert(self, alert: Dict[str, Any]):
        """Store alert in Redis for dashboard access."""
        try:
            redis_client = await get_redis()
            alert_key = f"alert:{alert['type']}:{int(time.time())}"

            await redis_client.setex(
                alert_key,
                3600,  # Expire after 1 hour
                str(alert),
            )

            # Also add to recent alerts list
            await redis_client.lpush("recent_alerts", str(alert))
            await redis_client.ltrim("recent_alerts", 0, 99)  # Keep last 100 alerts

        except Exception as e:
            logger.error(f"Failed to store alert in Redis: {str(e)}")


# Global alert manager instance
alert_manager = AlertManager()


class HealthChecker:
    """Performs periodic health checks and monitoring."""

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.is_running = False
        self.last_check = None

    async def start_monitoring(self):
        """Start the monitoring loop."""
        self.is_running = True
        logger.info("Health monitoring started")

        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.is_running = False
        logger.info("Health monitoring stopped")

    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        check_start = time.time()

        # Collect system metrics
        await system_monitor.collect_and_store_metrics()

        # Process alerts
        await alert_manager.process_alerts()

        # Clean up old metrics
        metrics_collector.clear_old_metrics()

        # Record health check performance
        check_duration = (time.time() - check_start) * 1000
        metrics_collector.record_performance("health_check", check_duration)

        self.last_check = datetime.now(timezone.utc)

        logger.debug(
            f"Health check completed in {check_duration:.2f}ms",
            extra={"duration_ms": check_duration},
        )


# Global health checker instance
health_checker = HealthChecker()


# Utility functions for easy access
def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of all collected metrics."""
    return {
        "metrics": metrics_collector.get_metrics_summary(),
        "system": system_monitor.get_system_metrics(),
        "last_health_check": health_checker.last_check.isoformat()
        if health_checker.last_check
        else None,
        "monitoring_active": health_checker.is_running,
    }


def record_api_metrics(
    endpoint: str, method: str, status_code: int, duration_ms: float
):
    """Record API-specific metrics."""
    # Increment request counter
    metrics_collector.increment_counter(
        f"api_requests_total",
        labels={"endpoint": endpoint, "method": method, "status": str(status_code)},
    )

    # Record response time
    metrics_collector.record_histogram(
        f"api_response_time",
        duration_ms,
        labels={"endpoint": endpoint, "method": method},
        unit="ms",
    )

    # Record success/error metrics
    if 200 <= status_code < 400:
        metrics_collector.increment_counter(
            f"api_requests_success", labels={"endpoint": endpoint}
        )
    else:
        metrics_collector.increment_counter(
            f"api_requests_error",
            labels={"endpoint": endpoint, "status": str(status_code)},
        )


def record_llm_metrics(
    provider: str, model: str, tokens_used: int, duration_ms: float, success: bool
):
    """Record LLM-specific metrics."""
    # Token usage
    metrics_collector.record_histogram(
        "llm_tokens_used",
        tokens_used,
        labels={"provider": provider, "model": model},
        unit="tokens",
    )

    # Response time
    metrics_collector.record_histogram(
        "llm_response_time",
        duration_ms,
        labels={"provider": provider, "model": model},
        unit="ms",
    )

    # Success/error counts
    if success:
        metrics_collector.increment_counter(
            "llm_requests_success", labels={"provider": provider, "model": model}
        )
    else:
        metrics_collector.increment_counter(
            "llm_requests_error", labels={"provider": provider, "model": model}
        )


def record_database_metrics(
    operation: str, table: str, duration_ms: float, success: bool
):
    """Record database operation metrics."""
    metrics_collector.record_histogram(
        "db_operation_time",
        duration_ms,
        labels={"operation": operation, "table": table},
        unit="ms",
    )

    if success:
        metrics_collector.increment_counter(
            "db_operations_success", labels={"operation": operation, "table": table}
        )
    else:
        metrics_collector.increment_counter(
            "db_operations_error", labels={"operation": operation, "table": table}
        )


# Export commonly used instances
__all__ = [
    "metrics_collector",
    "system_monitor",
    "alert_manager",
    "health_checker",
    "PerformanceMonitor",
    "monitor_performance",
    "get_metrics_summary",
    "record_api_metrics",
    "record_llm_metrics",
    "record_database_metrics",
]
