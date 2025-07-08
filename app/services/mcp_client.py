"""
MCP (Model Context Protocol) Client Implementation

This module provides the core MCP client functionality for connecting to and
communicating with MCP servers. It handles protocol communication, authentication,
and error handling for external system integrations.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import aiohttp
import websockets
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MCPMessage(BaseModel):
    """Base MCP message structure"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class MCPServer(BaseModel):
    """MCP Server configuration"""
    id: str
    name: str
    url: str
    protocol: str = "websocket"  # websocket, http, stdio
    auth_required: bool = False
    auth_type: Optional[str] = None  # "bearer", "basic", "api_key"
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    health_check_interval: int = 60  # seconds
    last_health_check: Optional[float] = None
    is_healthy: bool = True


class MCPAuthContext(BaseModel):
    """Authentication context for MCP operations"""
    user_id: Optional[str] = None
    session_id: str
    auth_token: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)


class MCPClient:
    """
    MCP Client for communicating with Model Context Protocol servers.
    
    Handles connection management, authentication, and protocol communication
    with external MCP servers for system integrations.
    """

    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.request_timeout = settings.MCP_REQUEST_TIMEOUT or 30
        self.max_retries = settings.MCP_MAX_RETRIES or 3
        self.retry_delay = settings.MCP_RETRY_DELAY or 1.0
        
    async def connect_to_server(self, server: MCPServer) -> bool:
        """
        Establish connection to an MCP server.
        
        Args:
            server: MCP server configuration
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if server.protocol == "websocket":
                return await self._connect_websocket(server)
            elif server.protocol == "http":
                return await self._connect_http(server)
            else:
                logger.error(f"Unsupported protocol: {server.protocol}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server.id}: {e}")
            return False

    async def _connect_websocket(self, server: MCPServer) -> bool:
        """Connect to WebSocket-based MCP server"""
        try:
            # Add authentication headers if required
            extra_headers = {}
            if server.auth_required and server.auth_type:
                auth_header = await self._get_auth_header(server)
                if auth_header:
                    extra_headers.update(auth_header)

            websocket = await websockets.connect(
                server.url,
                extra_headers=extra_headers,
                timeout=self.request_timeout
            )
            
            self.connections[server.id] = {
                "type": "websocket",
                "connection": websocket,
                "server": server,
                "connected_at": time.time()
            }
            
            logger.info(f"Connected to MCP server {server.id} via WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed for server {server.id}: {e}")
            return False

    async def _connect_http(self, server: MCPServer) -> bool:
        """Connect to HTTP-based MCP server"""
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            )
            
            self.connections[server.id] = {
                "type": "http",
                "connection": session,
                "server": server,
                "connected_at": time.time()
            }
            
            logger.info(f"Connected to MCP server {server.id} via HTTP")
            return True
            
        except Exception as e:
            logger.error(f"HTTP connection failed for server {server.id}: {e}")
            return False

    async def disconnect_from_server(self, server_id: str) -> bool:
        """
        Disconnect from an MCP server.
        
        Args:
            server_id: ID of the server to disconnect from
            
        Returns:
            bool: True if disconnection successful
        """
        try:
            if server_id not in self.connections:
                return True
                
            connection_info = self.connections[server_id]
            connection = connection_info["connection"]
            
            if connection_info["type"] == "websocket":
                await connection.close()
            elif connection_info["type"] == "http":
                await connection.close()
                
            del self.connections[server_id]
            logger.info(f"Disconnected from MCP server {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from MCP server {server_id}: {e}")
            return False

    async def send_request(
        self,
        server_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        auth_context: Optional[MCPAuthContext] = None
    ) -> Dict[str, Any]:
        """
        Send a request to an MCP server.
        
        Args:
            server_id: ID of the target server
            method: MCP method to call
            params: Parameters for the method
            auth_context: Authentication context for the request
            
        Returns:
            dict: Response from the MCP server
            
        Raises:
            ConnectionError: If server is not connected
            TimeoutError: If request times out
            ValueError: If response is invalid
        """
        if server_id not in self.connections:
            raise ConnectionError(f"Not connected to MCP server {server_id}")
            
        connection_info = self.connections[server_id]
        server = connection_info["server"]
        
        # Check if authentication is required
        if server.auth_required and not auth_context:
            raise ValueError(f"Authentication required for server {server_id}")
            
        message = MCPMessage(method=method, params=params or {})
        
        try:
            if connection_info["type"] == "websocket":
                return await self._send_websocket_request(connection_info, message, auth_context)
            elif connection_info["type"] == "http":
                return await self._send_http_request(connection_info, message, auth_context)
            else:
                raise ValueError(f"Unsupported connection type: {connection_info['type']}")
                
        except Exception as e:
            logger.error(f"Request failed for server {server_id}, method {method}: {e}")
            raise

    async def _send_websocket_request(
        self,
        connection_info: Dict[str, Any],
        message: MCPMessage,
        auth_context: Optional[MCPAuthContext]
    ) -> Dict[str, Any]:
        """Send request via WebSocket connection"""
        websocket = connection_info["connection"]
        
        # Add authentication if provided
        if auth_context and auth_context.auth_token:
            if message.params is None:
                message.params = {}
            message.params["auth_token"] = auth_context.auth_token
            
        # Send message
        await websocket.send(message.model_dump_json())
        
        # Wait for response with timeout
        try:
            response_data = await asyncio.wait_for(
                websocket.recv(),
                timeout=self.request_timeout
            )
            response = json.loads(response_data)
            
            if "error" in response and response["error"]:
                raise ValueError(f"MCP server error: {response['error']}")
                
            return response.get("result", response)
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timeout for message {message.id}")

    async def _send_http_request(
        self,
        connection_info: Dict[str, Any],
        message: MCPMessage,
        auth_context: Optional[MCPAuthContext]
    ) -> Dict[str, Any]:
        """Send request via HTTP connection"""
        session = connection_info["connection"]
        server = connection_info["server"]
        
        headers = {"Content-Type": "application/json"}
        
        # Add authentication headers if provided
        if auth_context and auth_context.auth_token:
            if server.auth_type == "bearer":
                headers["Authorization"] = f"Bearer {auth_context.auth_token}"
            elif server.auth_type == "api_key":
                headers["X-API-Key"] = auth_context.auth_token
                
        try:
            async with session.post(
                server.url,
                json=message.model_dump(),
                headers=headers
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                
                if "error" in response_data and response_data["error"]:
                    raise ValueError(f"MCP server error: {response_data['error']}")
                    
                return response_data.get("result", response_data)
                
        except aiohttp.ClientError as e:
            raise ConnectionError(f"HTTP request failed: {e}")

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        auth_context: Optional[MCPAuthContext] = None
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.
        
        Args:
            server_id: ID of the target server
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            auth_context: Authentication context
            
        Returns:
            dict: Tool execution result
        """
        return await self.send_request(
            server_id=server_id,
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            },
            auth_context=auth_context
        )

    async def list_tools(
        self,
        server_id: str,
        auth_context: Optional[MCPAuthContext] = None
    ) -> List[Dict[str, Any]]:
        """
        List available tools on an MCP server.
        
        Args:
            server_id: ID of the target server
            auth_context: Authentication context
            
        Returns:
            list: Available tools
        """
        result = await self.send_request(
            server_id=server_id,
            method="tools/list",
            auth_context=auth_context
        )
        return result.get("tools", [])

    async def get_server_capabilities(
        self,
        server_id: str,
        auth_context: Optional[MCPAuthContext] = None
    ) -> Dict[str, Any]:
        """
        Get capabilities of an MCP server.
        
        Args:
            server_id: ID of the target server
            auth_context: Authentication context
            
        Returns:
            dict: Server capabilities
        """
        return await self.send_request(
            server_id=server_id,
            method="initialize",
            params={"clientInfo": {"name": "chatbot-platform", "version": "1.0.0"}},
            auth_context=auth_context
        )

    async def health_check(self, server_id: str) -> bool:
        """
        Perform health check on an MCP server.
        
        Args:
            server_id: ID of the server to check
            
        Returns:
            bool: True if server is healthy
        """
        try:
            if server_id not in self.connections:
                return False
                
            # Try to ping the server
            await self.send_request(
                server_id=server_id,
                method="ping",
                params={}
            )
            
            # Update health status
            connection_info = self.connections[server_id]
            connection_info["server"].last_health_check = time.time()
            connection_info["server"].is_healthy = True
            
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for server {server_id}: {e}")
            
            # Update health status
            if server_id in self.connections:
                connection_info = self.connections[server_id]
                connection_info["server"].last_health_check = time.time()
                connection_info["server"].is_healthy = False
                
            return False

    async def _get_auth_header(self, server: MCPServer) -> Optional[Dict[str, str]]:
        """Get authentication header for server connection"""
        if not server.auth_required:
            return None
            
        # This would integrate with the main authentication system
        # For now, return basic structure
        if server.auth_type == "bearer":
            # Would get token from auth service
            return {"Authorization": "Bearer <token>"}
        elif server.auth_type == "api_key":
            # Would get API key from configuration
            return {"X-API-Key": "<api_key>"}
            
        return None

    async def shutdown(self):
        """Shutdown all MCP connections"""
        for server_id in list(self.connections.keys()):
            await self.disconnect_from_server(server_id)
        logger.info("MCP client shutdown complete")


# Global MCP client instance
mcp_client = MCPClient()


async def get_mcp_client() -> MCPClient:
    """Dependency injection for MCP client"""
    return mcp_client
