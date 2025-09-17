from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import TokenData, User
from app.config import settings
import logging

# Logging setup
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Verify and decode JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("Token does not contain user_id")
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
        return token_data
    except JWTError as e:
        logger.warning(f"JWT verification failed: {str(e)}")
        raise credentials_exception


def get_current_user(token_data: TokenData = Depends(verify_token)) -> User:
    """Get current user from token"""
    if token_data.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real application, you would fetch user details from database
    # For this example, we'll create a user object from token data
    user = User(user_id=token_data.user_id)
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def create_user_token(user_id: str) -> str:
    """Create access token for a user"""
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user_id}, expires_delta=access_token_expires
    )
    return access_token


# Optional: Simple user database (in production, use proper database)
fake_users_db = {
    "testuser": {
        "user_id": "testuser",
        "email": "test@example.com",
        "hashed_password": get_password_hash("testpassword"),
        "is_active": True,
    }
}


def authenticate_user(user_id: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user_dict = fake_users_db.get(user_id)
    if not user_dict:
        return None
    if not verify_password(password, user_dict["hashed_password"]):
        return None
    return User(**user_dict)


# Dependency for database session with user context
def get_db_with_user(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get database session with authenticated user context"""
    return db, current_user
