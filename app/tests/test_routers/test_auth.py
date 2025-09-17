import pytest
from httpx import AsyncClient


class TestAuthRouter:
    """Test authentication endpoints"""
    
    async def test_login_success(self, client: AsyncClient):
        """Test successful login"""
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "testuser", "password": "testpassword"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    async def test_login_invalid_credentials(self, client: AsyncClient):
        """Test login with invalid credentials"""
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "invalid", "password": "invalid"}
        )
        assert response.status_code == 401
    
    async def test_login_missing_fields(self, client: AsyncClient):
        """Test login with missing fields"""
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "testuser"}
        )
        assert response.status_code == 422
