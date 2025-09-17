from sqlalchemy import Column, String, DateTime, Float, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.database import Base


class EKYCSession(Base):
    __tablename__ = "ekyc_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending")
    id_card_data = Column(JSONB, nullable=True)
    face_match_score = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    final_decision = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    processing_stages = Column(JSONB, nullable=True, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship
    assets = relationship("EKYCAsset", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<EKYCSession(id={self.id}, user_id={self.user_id}, status={self.status})>"


class EKYCAsset(Base):
    __tablename__ = "ekyc_assets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("ekyc_sessions.id"), nullable=False)
    asset_type = Column(String(50), nullable=False)  # 'id_front', 'id_back', 'selfie'
    file_path = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_size = Column(Float, nullable=True)
    mime_type = Column(String(100), nullable=True)
    processed = Column(Boolean, default=False)
    processing_result = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    session = relationship("EKYCSession", back_populates="assets")
    
    def __repr__(self):
        return f"<EKYCAsset(id={self.id}, session_id={self.session_id}, asset_type={self.asset_type})>"
