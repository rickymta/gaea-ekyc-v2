import pytest
from httpx import AsyncClient
from uuid import uuid4


class TestSessionsRouter:
    """Test EKYC sessions endpoints"""
    
    async def test_create_session_success(self, client: AsyncClient, auth_headers, sample_session_data):
        """Test successful session creation"""
        response = await client.post(
            "/api/v1/sessions",
            json=sample_session_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["user_id"] == sample_session_data["user_id"]
        assert data["status"] == "pending"
    
    async def test_create_session_unauthorized(self, client: AsyncClient, sample_session_data):
        """Test session creation without auth"""
        response = await client.post(
            "/api/v1/sessions",
            json=sample_session_data
        )
        assert response.status_code == 401
    
    async def test_get_session_success(self, client: AsyncClient, auth_headers):
        """Test getting session by ID"""
        # First create a session
        create_response = await client.post(
            "/api/v1/sessions",
            json={"user_id": "test_user"},
            headers=auth_headers
        )
        session_id = create_response.json()["id"]
        
        # Then get it
        response = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
    
    async def test_get_session_not_found(self, client: AsyncClient, auth_headers):
        """Test getting non-existent session"""
        fake_id = str(uuid4())
        response = await client.get(
            f"/api/v1/sessions/{fake_id}",
            headers=auth_headers
        )
        assert response.status_code == 404
    
    async def test_list_user_sessions(self, client: AsyncClient, auth_headers):
        """Test listing user sessions"""
        response = await client.get(
            "/api/v1/sessions",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
