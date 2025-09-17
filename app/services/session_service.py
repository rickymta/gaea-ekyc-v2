from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from uuid import UUID
from app.models import EKYCSession, EKYCAsset
from app.schemas import EKYCSessionCreate, EKYCStatus, FinalDecision
import logging

logger = logging.getLogger(__name__)


class SessionService:
    @staticmethod
    def create_session(db: Session, session_data: EKYCSessionCreate) -> EKYCSession:
        """Create a new EKYC session"""
        db_session = EKYCSession(
            user_id=session_data.user_id,
            status=EKYCStatus.PENDING.value,
            processing_stages={}
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        logger.info(f"Created EKYC session {db_session.id} for user {session_data.user_id}")
        return db_session

    @staticmethod
    def get_session(db: Session, session_id: UUID) -> Optional[EKYCSession]:
        """Get EKYC session by ID"""
        return db.query(EKYCSession).filter(EKYCSession.id == session_id).first()

    @staticmethod
    def get_user_sessions(
        db: Session, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 10
    ) -> List[EKYCSession]:
        """Get all sessions for a user with pagination"""
        return (
            db.query(EKYCSession)
            .filter(EKYCSession.user_id == user_id)
            .order_by(desc(EKYCSession.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_user_sessions_count(db: Session, user_id: str) -> int:
        """Get total count of sessions for a user"""
        return db.query(func.count(EKYCSession.id)).filter(EKYCSession.user_id == user_id).scalar()

    @staticmethod
    def update_session_status(
        db: Session, 
        session_id: UUID, 
        status: EKYCStatus,
        error_message: Optional[str] = None
    ) -> Optional[EKYCSession]:
        """Update session status"""
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if session:
            session.status = status.value
            if error_message:
                session.error_message = error_message
            db.commit()
            db.refresh(session)
            logger.info(f"Updated session {session_id} status to {status.value}")
        return session

    @staticmethod
    def update_processing_stage(
        db: Session, 
        session_id: UUID, 
        stage: str, 
        completed: bool = True
    ) -> Optional[EKYCSession]:
        """Update a processing stage"""
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if session:
            if session.processing_stages is None:
                session.processing_stages = {}
            session.processing_stages[stage] = completed
            db.commit()
            db.refresh(session)
            logger.info(f"Updated session {session_id} stage {stage} to {completed}")
        return session

    @staticmethod
    def update_id_card_data(
        db: Session, 
        session_id: UUID, 
        id_card_data: Dict[str, Any]
    ) -> Optional[EKYCSession]:
        """Update ID card data for session"""
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if session:
            session.id_card_data = id_card_data
            db.commit()
            db.refresh(session)
            logger.info(f"Updated session {session_id} ID card data")
        return session

    @staticmethod
    def update_face_match_score(
        db: Session, 
        session_id: UUID, 
        face_match_score: float,
        liveness_score: Optional[float] = None
    ) -> Optional[EKYCSession]:
        """Update face match and liveness scores"""
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if session:
            session.face_match_score = face_match_score
            if liveness_score is not None:
                session.liveness_score = liveness_score
            db.commit()
            db.refresh(session)
            logger.info(f"Updated session {session_id} face scores")
        return session

    @staticmethod
    def finalize_session(
        db: Session, 
        session_id: UUID, 
        final_decision: FinalDecision
    ) -> Optional[EKYCSession]:
        """Finalize session with decision"""
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if session:
            session.final_decision = final_decision.value
            session.status = EKYCStatus.COMPLETED.value
            db.commit()
            db.refresh(session)
            logger.info(f"Finalized session {session_id} with decision {final_decision.value}")
        return session

    @staticmethod
    def delete_session(db: Session, session_id: UUID) -> bool:
        """Delete EKYC session"""
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            logger.info(f"Deleted session {session_id}")
            return True
        return False
