import pytest
from uuid import uuid4
from app.services.session_service import SessionService
from app.schemas import EKYCSessionCreate, EKYCStatus


class TestSessionService:
    """Test session service functionality"""
    
    def test_create_session(self, db):
        """Test creating a new session"""
        session_data = EKYCSessionCreate(user_id="test_user_123")
        session = SessionService.create_session(db, session_data)
        
        assert session.id is not None
        assert session.user_id == "test_user_123"
        assert session.status == EKYCStatus.PENDING.value
        assert session.processing_stages == {}
    
    def test_get_session(self, db):
        """Test getting session by ID"""
        # Create session first
        session_data = EKYCSessionCreate(user_id="test_user_123")
        created_session = SessionService.create_session(db, session_data)
        
        # Get session
        retrieved_session = SessionService.get_session(db, created_session.id)
        
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        assert retrieved_session.user_id == "test_user_123"
    
    def test_get_session_not_found(self, db):
        """Test getting non-existent session"""
        fake_id = uuid4()
        session = SessionService.get_session(db, fake_id)
        assert session is None
    
    def test_update_session_status(self, db):
        """Test updating session status"""
        # Create session
        session_data = EKYCSessionCreate(user_id="test_user_123")
        session = SessionService.create_session(db, session_data)
        
        # Update status
        updated_session = SessionService.update_session_status(
            db, session.id, EKYCStatus.IN_PROGRESS
        )
        
        assert updated_session is not None
        assert updated_session.status == EKYCStatus.IN_PROGRESS.value
    
    def test_update_processing_stage(self, db):
        """Test updating processing stage"""
        # Create session
        session_data = EKYCSessionCreate(user_id="test_user_123")
        session = SessionService.create_session(db, session_data)
        
        # Update stage
        updated_session = SessionService.update_processing_stage(
            db, session.id, "id_front_uploaded", True
        )
        
        assert updated_session is not None
        assert updated_session.processing_stages["id_front_uploaded"] is True
    
    def test_get_user_sessions(self, db):
        """Test getting user sessions with pagination"""
        user_id = "test_user_123"
        
        # Create multiple sessions
        for i in range(3):
            session_data = EKYCSessionCreate(user_id=user_id)
            SessionService.create_session(db, session_data)
        
        # Get sessions
        sessions = SessionService.get_user_sessions(db, user_id, skip=0, limit=2)
        assert len(sessions) == 2
        
        # Get count
        count = SessionService.get_user_sessions_count(db, user_id)
        assert count == 3
