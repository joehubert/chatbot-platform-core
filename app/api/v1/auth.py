"""
Authentication API endpoints.

This module provides authentication functionality including OTP generation,
verification, and session management.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, check_rate_limit
from app.schemas.auth import (
    AuthRequest,
    AuthResponse,
    AuthVerification,
    SessionStatusResponse,
)
from app.services.auth_service import AuthService
from app.services.otp_service import OTPService
from app.services.session_service import SessionService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/request", response_model=AuthResponse)
async def request_auth_token(
    request: AuthRequest,
    db: Session = Depends(get_db),
    _: bool = Depends(check_rate_limit),
):
    """
    Request an authentication token (OTP) to be sent via SMS or email.

    This endpoint triggers when MCP servers or tools require authentication.
    The OTP is sent to the user's registered contact method.
    """
    try:
        auth_service = AuthService(db)
        otp_service = OTPService()

        # Validate session exists
        session_service = SessionService(db)
        session = session_service.get_session(request.session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Find or create user based on contact method
        user = auth_service.find_or_create_user(
            contact_method=request.contact_method, contact_value=request.contact_value
        )

        # Generate and send OTP
        otp_token = await otp_service.generate_and_send_otp(
            user=user,
            session_id=request.session_id,
            contact_method=request.contact_method,
            contact_value=request.contact_value,
        )

        logger.info(
            f"OTP requested for session {request.session_id} via {request.contact_method}"
        )

        return AuthResponse(
            success=True,
            message=f"Authentication code sent via {request.contact_method}",
            session_id=request.session_id,
            expires_in=otp_service.get_otp_expiry_seconds(),
            contact_method=request.contact_method,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting auth token: {str(e)}")
        raise HTTPException(status_code=500, detail="Error sending authentication code")


@router.post("/verify", response_model=AuthVerification)
async def verify_auth_token(
    request: AuthVerification,
    db: Session = Depends(get_db),
    _: bool = Depends(check_rate_limit),
):
    """
    Verify an authentication token (OTP) and establish an authenticated session.

    Upon successful verification, the user's session is marked as authenticated
    and can access protected resources.
    """
    try:
        auth_service = AuthService(db)
        otp_service = OTPService()
        session_service = SessionService(db)

        # Verify the OTP token
        verification_result = await otp_service.verify_otp(
            session_id=request.session_id, token=request.token
        )

        if not verification_result.valid:
            raise HTTPException(
                status_code=400,
                detail=verification_result.error_message
                or "Invalid or expired authentication code",
            )

        # Update session with authentication
        session = session_service.authenticate_session(
            session_id=request.session_id, user_id=verification_result.user_id
        )

        # Generate session token for continued authentication
        session_token = auth_service.generate_session_token(
            session_id=request.session_id, user_id=verification_result.user_id
        )

        logger.info(f"Authentication successful for session {request.session_id}")

        return AuthVerification(
            success=True,
            message="Authentication successful",
            session_id=request.session_id,
            session_token=session_token,
            expires_in=auth_service.get_session_expiry_seconds(),
            user_id=str(verification_result.user_id),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying auth token: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error verifying authentication code"
        )


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str, db: Session = Depends(get_db)):
    """
    Check the authentication status of a session.

    Returns whether the session is authenticated and when it expires.
    """
    try:
        session_service = SessionService(db)
        auth_service = AuthService(db)

        session = session_service.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Check if session is authenticated and not expired
        is_authenticated = auth_service.is_session_authenticated(session_id)
        expires_at = auth_service.get_session_expiry(session_id)

        return SessionStatusResponse(
            session_id=session_id,
            authenticated=is_authenticated,
            expires_at=expires_at,
            user_id=str(session.user_id) if session.user_id else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking session status")


@router.post("/session/{session_id}/logout")
async def logout_session(session_id: str, db: Session = Depends(get_db)):
    """
    Log out a session, clearing authentication status.

    The session remains active but loses authentication privileges.
    """
    try:
        session_service = SessionService(db)
        auth_service = AuthService(db)

        # Clear authentication for the session
        success = auth_service.logout_session(session_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="Session not found or already logged out"
            )

        logger.info(f"Session {session_id} logged out successfully")

        return {
            "success": True,
            "message": "Successfully logged out",
            "session_id": session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging out session: {str(e)}")
        raise HTTPException(status_code=500, detail="Error logging out session")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """
    Completely delete a session and all associated data.

    This removes the session, authentication status, and conversation context.
    """
    try:
        session_service = SessionService(db)

        success = session_service.delete_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        logger.info(f"Session {session_id} deleted successfully")

        return {
            "success": True,
            "message": "Session deleted successfully",
            "session_id": session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting session")


@router.post("/sessions/{session_id}/extend")
async def extend_session(
    session_id: str,
    extension_minutes: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Extend the expiration time of an authenticated session.

    This is useful for keeping long conversations active without re-authentication.
    """
    try:
        auth_service = AuthService(db)

        # Use default extension if not specified
        if extension_minutes is None:
            extension_minutes = auth_service.get_default_extension_minutes()

        # Validate extension limits
        max_extension = auth_service.get_max_extension_minutes()
        if extension_minutes > max_extension:
            raise HTTPException(
                status_code=400,
                detail=f"Extension cannot exceed {max_extension} minutes",
            )

        new_expiry = auth_service.extend_session(session_id, extension_minutes)

        if not new_expiry:
            raise HTTPException(
                status_code=404, detail="Session not found or not authenticated"
            )

        return {
            "success": True,
            "message": f"Session extended by {extension_minutes} minutes",
            "session_id": session_id,
            "new_expiry": new_expiry.isoformat(),
            "extended_minutes": extension_minutes,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extending session: {str(e)}")
        raise HTTPException(status_code=500, detail="Error extending session")
