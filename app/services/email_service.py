"""
Email Service for sending authentication tokens

Handles email delivery for one-time passwords (OTP) using configurable providers.
Supports SendGrid as the primary provider with SMTP fallback.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from abc import ABC, abstractmethod

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailProvider(ABC):
    """Abstract base class for email providers."""
    
    @abstractmethod
    async def send_email(self, to_email: str, subject: str, message: str, html_message: Optional[str] = None) -> bool:
        """Send email message to the specified address."""
        pass


class SendGridProvider(EmailProvider):
    """SendGrid email provider implementation."""
    
    def __init__(self):
        self.api_key = settings.SENDGRID_API_KEY
        self.from_email = settings.SENDGRID_FROM_EMAIL
        self.from_name = getattr(settings, 'SENDGRID_FROM_NAME', 'Chatbot Platform')
    
    async def send_email(self, to_email: str, subject: str, message: str, html_message: Optional[str] = None) -> bool:
        """
        Send email using SendGrid.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            message: Plain text message
            html_message: Optional HTML message
            
        Returns:
            True if email was sent successfully
        """
        try:
            import sendgrid
            from sendgrid.helpers.mail import Mail
            
            sg = sendgrid.SendGridAPIClient(api_key=self.api_key)
            
            mail = Mail(
                from_email=(self.from_email, self.from_name),
                to_emails=to_email,
                subject=subject,
                plain_text_content=message
            )
            
            if html_message:
                mail.html_content = html_message
            
            response = sg.send(mail)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Email sent successfully via SendGrid. Status: {response.status_code}")
                return True
            else:
                logger.error(f"SendGrid returned status {response.status_code}: {response.body}")
                return False
                
        except ImportError:
            logger.error("SendGrid library not installed. Run: pip install sendgrid")
            return False
        except Exception as e:
            logger.error(f"Failed to send email via SendGrid: {str(e)}")
            return False


class SMTPProvider(EmailProvider):
    """SMTP email provider implementation."""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.smtp_use_tls = getattr(settings, 'SMTP_USE_TLS', True)
        self.from_email = settings.SMTP_FROM_EMAIL
        self.from_name = getattr(settings, 'SMTP_FROM_NAME', 'Chatbot Platform')
    
    async def send_email(self, to_email: str, subject: str, message: str, html_message: Optional[str] = None) -> bool:
        """
        Send email using SMTP.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            message: Plain text message
            html_message: Optional HTML message
            
        Returns:
            True if email was sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add plain text part
            text_part = MIMEText(message, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_message:
                html_part = MIMEText(html_message, 'html')
                msg.attach(html_part)
            
            # Connect to SMTP server and send
            if self.smtp_use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.from_email, to_email, text)
            server.quit()
            
            logger.info(f"Email sent successfully via SMTP")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {str(e)}")
            return False


class MockEmailProvider(EmailProvider):
    """Mock email provider for development and testing."""
    
    async def send_email(self, to_email: str, subject: str, message: str, html_message: Optional[str] = None) -> bool:
        """
        Mock email sending - logs message instead of sending.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            message: Email message content
            html_message: Optional HTML message
            
        Returns:
            Always True (mock success)
        """
        logger.info(f"MOCK EMAIL to {to_email}: {subject}")
        print(f"📧 MOCK EMAIL SENT")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Message:\n{message}")
        if html_message:
            print(f"HTML Message:\n{html_message}")
        print("=" * 50)
        return True


class EmailService:
    """
    Email service for sending authentication tokens.
    
    Handles OTP delivery via email using configurable providers.
    Automatically selects provider based on configuration.
    """
    
    def __init__(self):
        self.provider = self._initialize_provider()
    
    def _initialize_provider(self) -> EmailProvider:
        """Initialize email provider based on configuration."""
        provider_name = getattr(settings, 'EMAIL_PROVIDER', 'mock').lower()
        
        if provider_name == 'sendgrid':
            # Check if required SendGrid settings are present
            if (hasattr(settings, 'SENDGRID_API_KEY') and 
                hasattr(settings, 'SENDGRID_FROM_EMAIL') and
                settings.SENDGRID_API_KEY and 
                settings.SENDGRID_FROM_EMAIL):
                
                logger.info("Initializing SendGrid email provider")
                return SendGridProvider()
            else:
                logger.warning("SendGrid credentials not configured, falling back to mock provider")
                return MockEmailProvider()
        
        elif provider_name == 'smtp':
            # Check if required SMTP settings are present
            required_smtp_settings = ['SMTP_HOST', 'SMTP_PORT', 'SMTP_FROM_EMAIL']
            if all(hasattr(settings, attr) and getattr(settings, attr) for attr in required_smtp_settings):
                logger.info("Initializing SMTP email provider")
                return SMTPProvider()
            else:
                logger.warning("SMTP settings not configured, falling back to mock provider")
                return MockEmailProvider()
        
        elif provider_name == 'mock':
            logger.info("Using mock email provider")
            return MockEmailProvider()
        
        else:
            logger.warning(f"Unknown email provider '{provider_name}', using mock provider")
            return MockEmailProvider()
    
    async def send_otp(self, email_address: str, token: str) -> bool:
        """
        Send OTP token via email.
        
        Args:
            email_address: Recipient email address
            token: Authentication token to send
            
        Returns:
            True if email was sent successfully
        """
        try:
            # Format the message
            subject, message, html_message = self._format_otp_email(token)
            
            # Send via configured provider
            success = await self.provider.send_email(
                email_address, 
                subject, 
                message, 
                html_message
            )
            
            if success:
                logger.info(f"OTP sent successfully to {self._mask_email(email_address)}")
            else:
                logger.error(f"Failed to send OTP to {self._mask_email(email_address)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending OTP email: {str(e)}")
            return False
    
    def _format_otp_email(self, token: str) -> tuple[str, str, str]:
        """
        Format the OTP email content.
        
        Args:
            token: Authentication token
            
        Returns:
            Tuple of (subject, plain_text_message, html_message)
        """
        company_name = getattr(settings, 'COMPANY_NAME', 'Chatbot Platform')
        expiry_minutes = getattr(settings, 'OTP_EXPIRY_MINUTES', 5)
        
        subject = f"Your {company_name} Authentication Code"
        
        # Plain text message
        plain_message = f"""
Your {company_name} authentication code is: {token}

This code expires in {expiry_minutes} minutes.
Do not share this code with anyone.

If you did not request this code, please ignore this email.

Best regards,
{company_name} Team
        """.strip()
        
        # HTML message
        html_message = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            border-radius: 8px 8px 0 0;
        }}
        .content {{
            background-color: #ffffff;
            padding: 30px;
            border: 1px solid #e9ecef;
        }}
        .code-box {{
            background-color: #f8f9fa;
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }}
        .code {{
            font-size: 32px;
            font-weight: bold;
            color: #007bff;
            letter-spacing: 4px;
            font-family: monospace;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            margin: 20px 0;
            color: #856404;
        }}
        .footer {{
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            border-radius: 0 0 8px 8px;
            font-size: 14px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{company_name}</h1>
        <p>Authentication Code</p>
    </div>
    
    <div class="content">
        <p>Hello,</p>
        
        <p>You requested an authentication code for your {company_name} account. Please use the code below to continue:</p>
        
        <div class="code-box">
            <div class="code">{token}</div>
        </div>
        
        <div class="warning">
            <strong>Important:</strong>
            <ul>
                <li>This code expires in {expiry_minutes} minutes</li>
                <li>Do not share this code with anyone</li>
                <li>If you did not request this code, please ignore this email</li>
            </ul>
        </div>
        
        <p>If you're having trouble, please contact our support team.</p>
        
        <p>Best regards,<br>
        {company_name} Team</p>
    </div>
    
    <div class="footer">
        <p>This is an automated message. Please do not reply to this email.</p>
    </div>
</body>
</html>
        """.strip()
        
        return subject, plain_message, html_message
    
    def _mask_email(self, email: str) -> str:
        """Mask email address for logging."""
        if '@' in email:
            local, domain = email.split('@', 1)
            if len(local) > 2:
                masked_local = local[:2] + '*' * (len(local) - 2)
            else:
                masked_local = '*' * len(local)
            return f"{masked_local}@{domain}"
        return email
    
    async def test_connection(self) -> Dict[str, any]:
        """
        Test email provider connection and configuration.
        
        Returns:
            Dict with test results
        """
        try:
            provider_type = type(self.provider).__name__
            
            if isinstance(self.provider, SendGridProvider):
                # Test SendGrid connection
                try:
                    import sendgrid
                    sg = sendgrid.SendGridAPIClient(api_key=self.provider.api_key)
                    
                    return {
                        "success": True,
                        "provider": "SendGrid",
                        "status": "Connected",
                        "from_email": self.provider.from_email,
                        "from_name": self.provider.from_name
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "provider": "SendGrid",
                        "status": "Connection failed",
                        "error": str(e)
                    }
            
            elif isinstance(self.provider, SMTPProvider):
                # Test SMTP connection
                try:
                    if self.provider.smtp_use_tls:
                        server = smtplib.SMTP(self.provider.smtp_host, self.provider.smtp_port)
                        server.starttls()
                    else:
                        server = smtplib.SMTP_SSL(self.provider.smtp_host, self.provider.smtp_port)
                    
                    if self.provider.smtp_username and self.provider.smtp_password:
                        server.login(self.provider.smtp_username, self.provider.smtp_password)
                    
                    server.quit()
                    
                    return {
                        "success": True,
                        "provider": "SMTP",
                        "status": "Connected",
                        "host": self.provider.smtp_host,
                        "port": self.provider.smtp_port,
                        "from_email": self.provider.from_email
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "provider": "SMTP",
                        "status": "Connection failed",
                        "error": str(e)
                    }
            
            elif isinstance(self.provider, MockEmailProvider):
                return {
                    "success": True,
                    "provider": "Mock",
                    "status": "Ready",
                    "note": "Emails will be logged only"
                }
            
            else:
                return {
                    "success": False,
                    "provider": provider_type,
                    "status": "Unknown provider"
                }
                
        except Exception as e:
            logger.error(f"Error testing email connection: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Utility function for direct access
async def send_email_otp(email_address: str, token: str) -> bool:
    """
    Convenience function to send OTP via email.
    
    Args:
        email_address: Recipient email address
        token: Authentication token
        
    Returns:
        True if successful
    """
    service = EmailService()
    return await service.send_otp(email_address, token)
