import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import get_db, Base
from app.config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"

# Create test engine
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
async def client():
    """Create test client"""
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Get authentication headers for testing"""
    # This would normally create a test token
    # For now, we'll use a mock token
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def sample_session_data():
    """Sample EKYC session data for testing"""
    return {
        "user_id": "test_user_123"
    }


@pytest.fixture
def sample_id_card_data():
    """Sample ID card data for testing"""
    return {
        "id_number": "123456789012",
        "full_name": "NGUYEN VAN TEST",
        "date_of_birth": "01/01/1990",
        "gender": "Nam",
        "nationality": "Việt Nam",
        "address": "123 Test Street, Ho Chi Minh City"
    }
