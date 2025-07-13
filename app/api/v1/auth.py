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
    TokenResponse,
    AuthStatus,
    SessionInfo,
)
from app.services.auth_service import AuthService
from app.services.session_service import SessionService
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/request", response_model=AuthResponse)
async def request_auth_token(request: AuthRequest, db: Session = Depends(get_db)):
    auth_service = AuthService(db)

    # Use AuthService method instead
    result = await auth_service.request_auth_token(
        contact_method=request.contact_method,
        contact_value=request.contact_value,
        session_id=request.session_id,
    )

    return AuthResponse(
        success=True,
        message=f"Authentication code sent via {request.contact_method}",
        expires_in=300,  # 5 minutes from settings
        retry_after=None,
    )


@router.post("/verify", response_model=TokenResponse)  # ✅ Correct response model
async def verify_auth_token(request: AuthVerification, db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    session_service = SessionService(db)

    # Use AuthService for verification
    verification_result = await auth_service.verify_token(
        session_id=request.session_id, token=request.token
    )

    if not verification_result.success:
        raise HTTPException(status_code=400, detail=verification_result.message)

    return TokenResponse(
        success=True,
        message="Authentication successful",
        session_token=verification_result.session_token,
        expires_at=verification_result.expires_at,
        user_id=verification_result.user_id,
    )


@router.get("/session/{session_id}/status", response_model=AuthStatus)
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

        return AuthStatus(
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


@router.get("/status/{session_id}", response_model=AuthStatus)
async def get_auth_status(session_id: str):
    """Get authentication status for a session"""


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information"""
