"""
SMS Service for sending authentication tokens

Handles SMS delivery for one-time passwords (OTP) using configurable providers.
Supports Twilio as the primary provider with fallback options.
"""

import logging
from typing import Dict, Optional
from abc import ABC, abstractmethod

from app.core.config import settings

logger = logging.getLogger(__name__)


class SMSProvider(ABC):
    """Abstract base class for SMS providers."""
    
    @abstractmethod
    async def send_message(self, to_number: str, message: str) -> bool:
        """Send SMS message to the specified number."""
        pass


class TwilioProvider(SMSProvider):
    """Twilio SMS provider implementation."""
    
    def __init__(self):
        self.account_sid = settings.TWILIO_ACCOUNT_SID
        self.auth_token = settings.TWILIO_AUTH_TOKEN
        self.from_number = settings.TWILIO_FROM_NUMBER
        self._client = None
    
    def _get_client(self):
        """Lazy load Twilio client."""
        if self._client is None:
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
            except ImportError:
                logger.error("Twilio library not installed. Run: pip install twilio")
                raise
        return self._client
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """
        Send SMS using Twilio.
        
        Args:
            to_number: Recipient phone number
            message: SMS message content
            
        Returns:
            True if message was sent successfully
        """
        try:
            client = self._get_client()
            
            message_obj = client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            logger.info(f"SMS sent successfully. SID: {message_obj.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS via Twilio: {str(e)}")
            return False


class MockSMSProvider(SMSProvider):
    """Mock SMS provider for development and testing."""
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """
        Mock SMS sending - logs message instead of sending.
        
        Args:
            to_number: Recipient phone number
            message: SMS message content
            
        Returns:
            Always True (mock success)
        """
        logger.info(f"MOCK SMS to {to_number}: {message}")
        print(f"🔔 MOCK SMS SENT")
        print(f"📱 To: {to_number}")
        print(f"💬 Message: {message}")
        print("=" * 50)
        return True


class SMSService:
    """
    SMS service for sending authentication tokens.
    
    Handles OTP delivery via SMS using configurable providers.
    Automatically selects provider based on configuration.
    """
    
    def __init__(self):
        self.provider = self._initialize_provider()
    
    def _initialize_provider(self) -> SMSProvider:
        """Initialize SMS provider based on configuration."""
        provider_name = getattr(settings, 'SMS_PROVIDER', 'mock').lower()
        
        if provider_name == 'twilio':
            # Check if required Twilio settings are present
            if (hasattr(settings, 'TWILIO_ACCOUNT_SID') and 
                hasattr(settings, 'TWILIO_AUTH_TOKEN') and 
                hasattr(settings, 'TWILIO_FROM_NUMBER')):
                
                if (settings.TWILIO_ACCOUNT_SID and 
                    settings.TWILIO_AUTH_TOKEN and 
                    settings.TWILIO_FROM_NUMBER):
                    
                    logger.info("Initializing Twilio SMS provider")
                    return TwilioProvider()
                else:
                    logger.warning("Twilio credentials not configured, falling back to mock provider")
                    return MockSMSProvider()
            else:
                logger.warning("Twilio settings not found, falling back to mock provider")
                return MockSMSProvider()
        
        elif provider_name == 'mock':
            logger.info("Using mock SMS provider")
            return MockSMSProvider()
        
        else:
            logger.warning(f"Unknown SMS provider '{provider_name}', using mock provider")
            return MockSMSProvider()
    
    async def send_otp(self, phone_number: str, token: str) -> bool:
        """
        Send OTP token via SMS.
        
        Args:
            phone_number: Recipient phone number
            token: Authentication token to send
            
        Returns:
            True if SMS was sent successfully
        """
        try:
            # Format the message
            message = self._format_otp_message(token)
            
            # Send via configured provider
            success = await self.provider.send_message(phone_number, message)
            
            if success:
                logger.info(f"OTP sent successfully to {self._mask_phone(phone_number)}")
            else:
                logger.error(f"Failed to send OTP to {self._mask_phone(phone_number)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending OTP: {str(e)}")
            return False
    
    def _format_otp_message(self, token: str) -> str:
        """
        Format the OTP message text.
        
        Args:
            token: Authentication token
            
        Returns:
            Formatted SMS message
        """
        company_name = getattr(settings, 'COMPANY_NAME', 'Chatbot Platform')
        expiry_minutes = getattr(settings, 'OTP_EXPIRY_MINUTES', 5)
        
        message = (
            f"Your {company_name} authentication code is: {token}\n\n"
            f"This code expires in {expiry_minutes} minutes. "
            f"Do not share this code with anyone."
        )
        
        return message
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for logging."""
        if len(phone) >= 4:
            return f"***-***-{phone[-4:]}"
        return "***-***-****"
    
    async def test_connection(self) -> Dict[str, any]:
        """
        Test SMS provider connection and configuration.
        
        Returns:
            Dict with test results
        """
        try:
            provider_type = type(self.provider).__name__
            
            if isinstance(self.provider, TwilioProvider):
                # Test Twilio connection
                try:
                    client = self.provider._get_client()
                    # Try to fetch account info to test connection
                    account = client.api.accounts(self.provider.account_sid).fetch()
                    
                    return {
                        "success": True,
                        "provider": "Twilio",
                        "status": "Connected",
                        "account_sid": account.sid[:8] + "...",  # Masked
                        "from_number": self.provider.from_number
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "provider": "Twilio",
                        "status": "Connection failed",
                        "error": str(e)
                    }
            
            elif isinstance(self.provider, MockSMSProvider):
                return {
                    "success": True,
                    "provider": "Mock",
                    "status": "Ready",
                    "note": "SMS messages will be logged only"
                }
            
            else:
                return {
                    "success": False,
                    "provider": provider_type,
                    "status": "Unknown provider"
                }
                
        except Exception as e:
            logger.error(f"Error testing SMS connection: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Utility function for direct access
async def send_sms_otp(phone_number: str, token: str) -> bool:
    """
    Convenience function to send OTP via SMS.
    
    Args:
        phone_number: Recipient phone number
        token: Authentication token
        
    Returns:
        True if successful
    """
    service = SMSService()
    return await service.send_otp(phone_number, token)
