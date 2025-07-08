"""
Authentication Service for Chatbot Platform

Handles user authentication through one-time tokens (OTP) sent via SMS or email.
Manages session creation, validation, and cleanup for authenticated users.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.core.database import get_db
from app.models.user import User
from app.models.auth_token import AuthToken
from app.services.sms_service import SMSService
from app.services.email_service import EmailService
from app.utils.token_utils import TokenUtils
from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthService:
    """
    Handles authentication operations for the chatbot platform.
    
    This service manages:
    - OTP generation and validation
    - User lookup and creation
    - Session management
    - Token lifecycle
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.sms_service = SMSService()
        self.email_service = EmailService()
        self.token_utils = TokenUtils()
    
    async def request_auth_token(
        self,
        contact_method: str,
        contact_value: str,
        session_id: str
    ) -> Dict[str, str]:
        """
        Generate and send an authentication token to user.
        
        Args:
            contact_method: 'sms' or 'email'
            contact_value: Phone number or email address
            session_id: Current conversation session ID
            
        Returns:
            Dict containing success status and delivery method
            
        Raises:
            HTTPException: If contact method is invalid or delivery fails
        """
        try:
            # Validate contact method
            if contact_method not in ['sms', 'email']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Contact method must be 'sms' or 'email'"
                )
            
            # Validate contact value format
            if contact_method == 'sms':
                if not self._validate_phone_number(contact_value):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid phone number format"
                    )
            elif contact_method == 'email':
                if not self._validate_email(contact_value):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid email address format"
                    )
            
            # Find or create user
            user = await self._get_or_create_user(contact_method, contact_value)
            
            # Invalidate any existing tokens for this session
            await self._invalidate_session_tokens(session_id)
            
            # Generate new token
            token_value = self.token_utils.generate_secure_token()
            expires_at = datetime.utcnow() + timedelta(
                minutes=settings.OTP_EXPIRY_MINUTES
            )
            
            # Store token in database
            auth_token = AuthToken(
                id=uuid4(),
                user_id=user.id,
                token=self.token_utils.hash_token(token_value),
                expires_at=expires_at,
                used=False,
                session_id=session_id
            )
            
            self.db.add(auth_token)
            self.db.commit()
            
            # Send token via requested method
            if contact_method == 'sms':
                await self.sms_service.send_otp(contact_value, token_value)
                delivery_message = f"Authentication code sent to {self._mask_phone(contact_value)}"
            else:
                await self.email_service.send_otp(contact_value, token_value)
                delivery_message = f"Authentication code sent to {self._mask_email(contact_value)}"
            
            logger.info(
                f"Auth token requested for session {session_id}, "
                f"method: {contact_method}, user: {user.id}"
            )
            
            return {
                "success": True,
                "message": delivery_message,
                "expires_in_minutes": settings.OTP_EXPIRY_MINUTES
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error requesting auth token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send authentication code"
            )
    
    async def verify_auth_token(
        self,
        session_id: str,
        token_value: str
    ) -> Dict[str, any]:
        """
        Verify an authentication token and create session.
        
        Args:
            session_id: Current conversation session ID
            token_value: User-provided token
            
        Returns:
            Dict containing verification status and user info
            
        Raises:
            HTTPException: If token is invalid, expired, or already used
        """
        try:
            # Hash the provided token for comparison
            token_hash = self.token_utils.hash_token(token_value)
            
            # Find the token
            auth_token = self.db.query(AuthToken).filter(
                AuthToken.session_id == session_id,
                AuthToken.token == token_hash,
                AuthToken.used == False
            ).first()
            
            if not auth_token:
                logger.warning(f"Invalid token attempt for session {session_id}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication code"
                )
            
            # Check if token has expired
            if datetime.utcnow() > auth_token.expires_at:
                logger.warning(f"Expired token used for session {session_id}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication code has expired"
                )
            
            # Mark token as used
            auth_token.used = True
            
            # Update user's last authenticated timestamp
            user = self.db.query(User).filter(User.id == auth_token.user_id).first()
            if user:
                user.last_authenticated = datetime.utcnow()
            
            self.db.commit()
            
            logger.info(
                f"Successful authentication for session {session_id}, "
                f"user: {auth_token.user_id}"
            )
            
            return {
                "success": True,
                "user_id": str(auth_token.user_id),
                "authenticated_at": datetime.utcnow().isoformat(),
                "session_timeout_minutes": settings.AUTH_SESSION_TIMEOUT_MINUTES
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error verifying auth token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to verify authentication code"
            )
    
    async def is_session_authenticated(self, session_id: str) -> Tuple[bool, Optional[UUID]]:
        """
        Check if a session is currently authenticated.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            Tuple of (is_authenticated, user_id)
        """
        try:
            # Look for recent successful authentication
            cutoff_time = datetime.utcnow() - timedelta(
                minutes=settings.AUTH_SESSION_TIMEOUT_MINUTES
            )
            
            recent_auth = self.db.query(AuthToken).join(User).filter(
                AuthToken.session_id == session_id,
                AuthToken.used == True,
                User.last_authenticated >= cutoff_time
            ).first()
            
            if recent_auth:
                return True, recent_auth.user_id
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error checking session authentication: {str(e)}")
            return False, None
    
    async def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate all authentication tokens for a session.
        
        Args:
            session_id: Session to invalidate
            
        Returns:
            True if successful
        """
        try:
            # Mark all unused tokens for this session as used
            self.db.query(AuthToken).filter(
                AuthToken.session_id == session_id,
                AuthToken.used == False
            ).update({"used": True})
            
            self.db.commit()
            
            logger.info(f"Session {session_id} invalidated")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating session: {str(e)}")
            return False
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Remove expired authentication tokens from database.
        
        Returns:
            Number of tokens cleaned up
        """
        try:
            expired_tokens = self.db.query(AuthToken).filter(
                AuthToken.expires_at < datetime.utcnow()
            )
            
            count = expired_tokens.count()
            expired_tokens.delete()
            self.db.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens")
            
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {str(e)}")
            return 0
    
    async def _get_or_create_user(self, contact_method: str, contact_value: str) -> User:
        """Find existing user or create new one."""
        if contact_method == 'sms':
            user = self.db.query(User).filter(User.mobile_number == contact_value).first()
            if not user:
                user = User(
                    id=uuid4(),
                    mobile_number=contact_value,
                    created_at=datetime.utcnow()
                )
                self.db.add(user)
                self.db.commit()
                self.db.refresh(user)
        else:  # email
            user = self.db.query(User).filter(User.email == contact_value).first()
            if not user:
                user = User(
                    id=uuid4(),
                    email=contact_value,
                    created_at=datetime.utcnow()
                )
                self.db.add(user)
                self.db.commit()
                self.db.refresh(user)
        
        return user
    
    async def _invalidate_session_tokens(self, session_id: str):
        """Mark all unused tokens for a session as used."""
        self.db.query(AuthToken).filter(
            AuthToken.session_id == session_id,
            AuthToken.used == False
        ).update({"used": True})
        self.db.commit()
    
    def _validate_phone_number(self, phone: str) -> bool:
        """Basic phone number validation."""
        # Remove common formatting characters
        cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
        
        # Should be between 10-15 digits, possibly with country code
        return 10 <= len(cleaned.replace('+', '')) <= 15
    
    def _validate_email(self, email: str) -> bool:
        """Basic email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for display."""
        if len(phone) >= 4:
            return f"***-***-{phone[-4:]}"
        return "***-***-****"
    
    def _mask_email(self, email: str) -> str:
        """Mask email address for display."""
        if '@' in email:
            local, domain = email.split('@', 1)
            if len(local) > 2:
                masked_local = local[:2] + '*' * (len(local) - 2)
            else:
                masked_local = '*' * len(local)
            return f"{masked_local}@{domain}"
        return email
