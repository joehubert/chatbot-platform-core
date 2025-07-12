"""
MCP (Model Context Protocol) Server Registry

This module manages the registration, discovery, and lifecycle of MCP servers.
It provides centralized management for all MCP server connections, health monitoring,
and load balancing capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set
from uuid import uuid4
import json
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.mcp_client import MCPClient, MCPServer, MCPAuthContext

logger = logging.getLogger(__name__)
settings = get_settings()


class ServerLoadInfo(BaseModel):
    """Server load and performance information"""
    server_id: str
    active_connections: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    last_updated: float = Field(default_factory=time.time)


class MCPServerConfig(BaseModel):
    """Extended MCP server configuration with registry metadata"""
    server: MCPServer
    enabled: bool = True
    priority: int = 1  # 1 = highest priority
    max_connections: int = 10
    load_info: ServerLoadInfo = Field(default_factory=lambda: ServerLoadInfo(server_id=""))
    tags: Set[str] = Field(default_factory=set)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class MCPRegistry:
    """
    MCP Server Registry for managing multiple MCP server connections.
    
    Provides centralized registration, discovery, health monitoring,
    and load balancing for MCP servers.
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.servers: Dict[str, MCPServerConfig] = {}
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_interval = settings.MCP_HEALTH_CHECK_INTERVAL or 60
        self.config_file_path = Path(settings.MCP_CONFIG_FILE or "mcp_servers.json")
        self._running = False

    async def start(self):
        """Start the MCP registry and health monitoring"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting MCP Registry")
        
        # Load server configurations
        await self.load_server_configurations()
        
        # Start health monitoring
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        
        # Auto-connect to all enabled servers
        await self.connect_all_servers()

    async def stop(self):
        """Stop the MCP registry and cleanup"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping MCP Registry")
        
        # Cancel health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
                
        # Disconnect from all servers
        await self.disconnect_all_servers()

    async def register_server(
        self,
        server_config: MCPServerConfig,
        auto_connect: bool = True
    ) -> bool:
        """
        Register a new MCP server.
        
        Args:
            server_config: Server configuration
            auto_connect: Whether to automatically connect to the server
            
        Returns:
            bool: True if registration successful
        """
        try:
            server_id = server_config.server.id
            
            # Update timestamps
            server_config.updated_at = time.time()
            if server_id not in self.servers:
                server_config.created_at = time.time()
                
            # Initialize load info
            if not server_config.load_info.server_id:
                server_config.load_info.server_id = server_id
                
            # Store configuration
            self.servers[server_id] = server_config
            
            # Persist to database if available
            await self._persist_server_config(server_config)
            
            # Auto-connect if requested and enabled
            if auto_connect and server_config.enabled:
                await self.connect_to_server(server_id)
                
            logger.info(f"Registered MCP server: {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register MCP server {server_config.server.id}: {e}")
            return False

    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server.
        
        Args:
            server_id: ID of the server to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        try:
            if server_id not in self.servers:
                return True
                
            # Disconnect if connected
            await self.mcp_client.disconnect_from_server(server_id)
            
            # Remove from registry
            del self.servers[server_id]
            
            # Remove from database
            await self._remove_server_config(server_id)
            
            logger.info(f"Unregistered MCP server: {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister MCP server {server_id}: {e}")
            return False

    async def connect_to_server(self, server_id: str) -> bool:
        """
        Connect to a specific MCP server.
        
        Args:
            server_id: ID of the server to connect to
            
        Returns:
            bool: True if connection successful
        """
        if server_id not in self.servers:
            logger.error(f"Server {server_id} not registered")
            return False
            
        server_config = self.servers[server_id]
        if not server_config.enabled:
            logger.warning(f"Server {server_id} is disabled")
            return False
            
        try:
            success = await self.mcp_client.connect_to_server(server_config.server)
            if success:
                # Initialize server capabilities
                await self._initialize_server_capabilities(server_id)
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to connect to server {server_id}: {e}")
            return False

    async def disconnect_from_server(self, server_id: str) -> bool:
        """
        Disconnect from a specific MCP server.
        
        Args:
            server_id: ID of the server to disconnect from
            
        Returns:
            bool: True if disconnection successful
        """
        return await self.mcp_client.disconnect_from_server(server_id)

    async def connect_all_servers(self):
        """Connect to all enabled servers"""
        for server_id, config in self.servers.items():
            if config.enabled:
                await self.connect_to_server(server_id)

    async def disconnect_all_servers(self):
        """Disconnect from all servers"""
        for server_id in self.servers.keys():
            await self.disconnect_from_server(server_id)

    async def get_available_servers(
        self,
        capability: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        healthy_only: bool = True
    ) -> List[MCPServerConfig]:
        """
        Get list of available servers matching criteria.
        
        Args:
            capability: Required capability
            tags: Required tags
            healthy_only: Only return healthy servers
            
        Returns:
            list: Matching server configurations
        """
        available_servers = []
        
        for server_config in self.servers.values():
            # Check if enabled
            if not server_config.enabled:
                continue
                
            # Check health status
            if healthy_only and not server_config.server.is_healthy:
                continue
                
            # Check capability
            if capability and capability not in server_config.server.capabilities:
                continue
                
            # Check tags
            if tags and not tags.issubset(server_config.tags):
                continue
                
            available_servers.append(server_config)
            
        # Sort by priority and load
        available_servers.sort(
            key=lambda x: (
                x.priority,  # Higher priority first
                x.load_info.active_connections,  # Lower load first
                x.load_info.error_rate  # Lower error rate first
            )
        )
        
        return available_servers

    async def select_best_server(
        self,
        capability: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Select the best available server for a request.
        
        Args:
            capability: Required capability
            tags: Required tags
            
        Returns:
            str: Server ID of the best server, or None if none available
        """
        available_servers = await self.get_available_servers(
            capability=capability,
            tags=tags,
            healthy_only=True
        )
        
        if not available_servers:
            return None
            
        # Return the first server (best by our sorting criteria)
        return available_servers[0].server.id

    async def execute_on_best_server(
        self,
        method: str,
        params: Optional[Dict] = None,
        capability: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        auth_context: Optional[MCPAuthContext] = None
    ) -> Dict:
        """
        Execute a method on the best available server.
        
        Args:
            method: MCP method to execute
            params: Method parameters
            capability: Required capability
            tags: Required tags
            auth_context: Authentication context
            
        Returns:
            dict: Response from the server
            
        Raises:
            ConnectionError: If no servers are available
        """
        server_id = await self.select_best_server(capability=capability, tags=tags)
        
        if not server_id:
            raise ConnectionError("No available MCP servers for the request")
            
        try:
            # Record request start
            start_time = time.time()
            self._record_request_start(server_id)
            
            # Execute request
            result = await self.mcp_client.send_request(
                server_id=server_id,
                method=method,
                params=params,
                auth_context=auth_context
            )
            
            # Record successful completion
            response_time = time.time() - start_time
            self._record_request_success(server_id, response_time)
            
            return result
            
        except Exception as e:
            # Record error
            response_time = time.time() - start_time
            self._record_request_error(server_id, response_time)
            raise

    async def call_tool_on_best_server(
        self,
        tool_name: str,
        arguments: Dict,
        capability: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        auth_context: Optional[MCPAuthContext] = None
    ) -> Dict:
        """
        Call a tool on the best available server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            capability: Required capability
            tags: Required tags
            auth_context: Authentication context
            
        Returns:
            dict: Tool execution result
        """
        return await self.execute_on_best_server(
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
            capability=capability,
            tags=tags,
            auth_context=auth_context
        )

    async def list_all_available_tools(
        self,
        auth_context: Optional[MCPAuthContext] = None
    ) -> Dict[str, List[Dict]]:
        """
        List all available tools across all servers.
        
        Args:
            auth_context: Authentication context
            
        Returns:
            dict: Tools organized by server ID
        """
        all_tools = {}
        
        for server_id, config in self.servers.items():
            if not config.enabled or not config.server.is_healthy:
                continue
                
            try:
                tools = await self.mcp_client.list_tools(
                    server_id=server_id,
                    auth_context=auth_context
                )
                all_tools[server_id] = tools
                
            except Exception as e:
                logger.warning(f"Failed to list tools for server {server_id}: {e}")
                all_tools[server_id] = []
                
        return all_tools

    async def get_server_status(self) -> Dict[str, Dict]:
        """
        Get status of all registered servers.
        
        Returns:
            dict: Server status information
        """
        status = {}
        
        for server_id, config in self.servers.items():
            status[server_id] = {
                "enabled": config.enabled,
                "healthy": config.server.is_healthy,
                "connected": server_id in self.mcp_client.connections,
                "priority": config.priority,
                "capabilities": config.server.capabilities,
                "tags": list(config.tags),
                "load": config.load_info.model_dump(),
                "last_health_check": config.server.last_health_check
            }
            
        return status

    async def load_server_configurations(self):
        """Load server configurations from database and config file"""
        try:
            # Load from database first
            await self._load_from_database()
            
            # Load from config file if it exists
            if self.config_file_path.exists():
                await self._load_from_config_file()
                
        except Exception as e:
            logger.error(f"Failed to load server configurations: {e}")

    async def _load_from_config_file(self):
        """Load servers from JSON config file"""
        try:
            with open(self.config_file_path, 'r') as f:
                config_data = json.load(f)
                
            for server_data in config_data.get("servers", []):
                server = MCPServer(**server_data["server"])
                config = MCPServerConfig(
                    server=server,
                    enabled=server_data.get("enabled", True),
                    priority=server_data.get("priority", 1),
                    max_connections=server_data.get("max_connections", 10),
                    tags=set(server_data.get("tags", []))
                )
                
                await self.register_server(config, auto_connect=False)
                
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_file_path}: {e}")

    async def _load_from_database(self):
        """Load servers from database"""
        # This would load from the database if available
        # Implementation depends on the specific database models
        pass

    async def _persist_server_config(self, server_config: MCPServerConfig):
        """Persist server configuration to database"""
        # This would save to the database if available
        # Implementation depends on the specific database models
        pass

    async def _remove_server_config(self, server_id: str):
        """Remove server configuration from database"""
        # This would remove from the database if available
        # Implementation depends on the specific database models
        pass

    async def _initialize_server_capabilities(self, server_id: str):
        """Initialize server capabilities after connection"""
        try:
            capabilities = await self.mcp_client.get_server_capabilities(server_id)
            
            if server_id in self.servers:
                server_config = self.servers[server_id]
                # Update capabilities based on server response
                if "capabilities" in capabilities:
                    server_config.server.capabilities = capabilities["capabilities"].get("experimental", [])
                    
        except Exception as e:
            logger.warning(f"Failed to initialize capabilities for server {server_id}: {e}")

    async def _health_monitor_loop(self):
        """Background task for monitoring server health"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)  # Short delay before retry

    async def _perform_health_checks(self):
        """Perform health checks on all registered servers"""
        for server_id, config in self.servers.items():
            if not config.enabled:
                continue
                
            try:
                # Check if health check is due
                current_time = time.time()
                last_check = config.server.last_health_check or 0
                
                if current_time - last_check >= config.server.health_check_interval:
                    await self.mcp_client.health_check(server_id)
                    
            except Exception as e:
                logger.warning(f"Health check failed for server {server_id}: {e}")

    def _record_request_start(self, server_id: str):
        """Record the start of a request for load tracking"""
        if server_id in self.servers:
            load_info = self.servers[server_id].load_info
            load_info.active_connections += 1
            load_info.last_updated = time.time()

    def _record_request_success(self, server_id: str, response_time: float):
        """Record successful request completion"""
        if server_id in self.servers:
            load_info = self.servers[server_id].load_info
            load_info.active_connections = max(0, load_info.active_connections - 1)
            load_info.total_requests += 1
            
            # Update average response time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            if load_info.average_response_time == 0:
                load_info.average_response_time = response_time
            else:
                load_info.average_response_time = (
                    alpha * response_time + 
                    (1 - alpha) * load_info.average_response_time
                )
                
            load_info.last_updated = time.time()

    def _record_request_error(self, server_id: str, response_time: float):
        """Record failed request"""
        if server_id in self.servers:
            load_info = self.servers[server_id].load_info
            load_info.active_connections = max(0, load_info.active_connections - 1)
            load_info.total_requests += 1
            
            # Update error rate (exponential moving average)
            alpha = 0.1
            current_error_rate = 1.0  # This request failed
            if load_info.error_rate == 0:
                load_info.error_rate = current_error_rate
            else:
                load_info.error_rate = (
                    alpha * current_error_rate + 
                    (1 - alpha) * load_info.error_rate
                )
                
            load_info.last_updated = time.time()


# Global MCP registry instance
mcp_registry: Optional[MCPRegistry] = None


async def get_mcp_registry() -> MCPRegistry:
    """Dependency injection for MCP registry"""
    global mcp_registry
    if mcp_registry is None:
        from app.services.mcp_client import mcp_client
        mcp_registry = MCPRegistry(mcp_client)
        await mcp_registry.start()
    return mcp_registry


async def shutdown_mcp_registry():
    """Shutdown the global MCP registry"""
    global mcp_registry
    if mcp_registry:
        await mcp_registry.stop()
        mcp_registry = None
