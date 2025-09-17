from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from app.dependencies import authenticate_user, create_user_token
from app.schemas import Token, ErrorResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])
security = HTTPBearer()


@router.post(
    "/auth/login",
    response_model=Token,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"}
    },
    summary="User Login",
    description="""
    **Đăng nhập người dùng và nhận JWT token**
    
    Test credentials:
    - username: `testuser`
    - password: `testpassword`
    
    Token sẽ hết hạn sau 30 phút (mặc định).
    """
)
async def login(
    username: str = Form(..., description="Username", example="testuser"),
    password: str = Form(..., description="Password", example="testpassword")
):
    """Login endpoint to get JWT token"""
    user = authenticate_user(username, password)
    if not user:
        logger.warning(f"Failed login attempt for user: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_user_token(user.user_id)
    logger.info(f"User {username} logged in successfully")
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.post(
    "/auth/refresh",
    response_model=Token,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or expired token"}
    },
    summary="Refresh Token",
    description="Làm mới JWT token với token hiện tại"
)
async def refresh_token(current_user = Depends(authenticate_user)):
    """Refresh JWT token"""
    access_token = create_user_token(current_user.user_id)
    logger.info(f"Token refreshed for user {current_user.user_id}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
