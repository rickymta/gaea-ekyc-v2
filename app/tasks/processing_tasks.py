from celery import Celery
from celery.utils.log import get_task_logger
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import json
from typing import Dict, Any, Optional, List
import os
import time
import uuid
import tempfile
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.models import EKYCSession, EKYCAsset
from app.schemas import IDCardData, ProcessingTaskResult
from app.services.face_service import face_engine, load_image_from_path
from app.services.storage_service import file_manager
from app.config import settings
from app.webhooks.notifier import webhook_notifier

# Logger setup
logger = get_task_logger(__name__)

# Database setup for tasks
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# OCR engine (initialize once)
ocr_engine = None

def get_ocr_engine():
    """Get or initialize OCR engine"""
    global ocr_engine
    if ocr_engine is None:
        try:
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='vi')  # Vietnamese language
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')  # Fallback to English
    return ocr_engine


def get_file_for_processing_sync(object_name: str) -> Optional[str]:
    """Download file to temporary location for processing (sync version)"""
    import tempfile
    try:
        temp_path = os.path.join(tempfile.gettempdir(), f"ekyc_{uuid.uuid4()}")
        success = file_manager.storage.download_file(object_name, temp_path)
        return temp_path if success else None
    except Exception as e:
        logger.error(f"Error downloading file for processing: {str(e)}")
        return None


def cleanup_temp_file_sync(file_path: str):
    """Clean up temporary file (sync version)"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp file {file_path}: {str(e)}")


def update_session_status(session_id: str, status: str, **kwargs):
    """Update session status and other fields"""
    db = SessionLocal()
    try:
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if session:
            session.status = status
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            db.commit()
            logger.info(f"Updated session {session_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update session status: {str(e)}")
        db.rollback()
    finally:
        db.close()


def update_asset_processing(asset_id: str, processed: bool, result: Dict[str, Any]):
    """Update asset processing status and result"""
    db = SessionLocal()
    try:
        asset = db.query(EKYCAsset).filter(EKYCAsset.id == asset_id).first()
        if asset:
            asset.processed = processed
            asset.processing_result = result
            db.commit()
            logger.info(f"Updated asset {asset_id} processing status")
    except Exception as e:
        logger.error(f"Failed to update asset processing: {str(e)}")
        db.rollback()
    finally:
        db.close()


# Import celery app after defining dependencies
from app.tasks.celery_app import celery_app


@celery_app.task(bind=True, name='process_id_card_task')
def process_id_card_task(self, asset_id: str):
    """Process ID card image with OCR"""
    logger.info(f"Starting ID card processing for asset {asset_id}")
    start_time = time.time()
    
    db = SessionLocal()
    try:
        # Get asset information
        asset = db.query(EKYCAsset).filter(EKYCAsset.id == asset_id).first()
        if not asset:
            raise Exception(f"Asset {asset_id} not found")
        
        session_id = str(asset.session_id)
        
        # Update session status
        update_session_status(session_id, "in_progress")
        
        # Download file for processing
        temp_file_path = get_file_for_processing_sync(asset.file_path)
        if not temp_file_path:
            raise Exception("Failed to download file for processing")
        
        # Process with OCR
        ocr_result = extract_id_card_info(temp_file_path)
        
        # Clean up temp file
        cleanup_temp_file_sync(temp_file_path)
        
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "task_type": "id_card_processing",
            "success": True,
            "ocr_result": ocr_result,
            "processing_time": processing_time,
            "timestamp": int(time.time())
        }
        
        # Update asset
        update_asset_processing(asset_id, True, result)
        
        # Send webhook notification
        webhook_notifier.notify_asset_processed(
            session_id=session_id,
            asset_id=asset_id,
            asset_type=asset.asset_type,
            user_id=asset.session.user_id if hasattr(asset, 'session') else "unknown",
            success=True,
            processing_result=result
        )
        
        # Update session with ID card data
        if ocr_result.get("extracted_data"):
            db_session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
            if db_session:
                db_session.id_card_data = ocr_result["extracted_data"]
                
                # Update processing stage
                stages = db_session.processing_stages or {}
                stages[f"{asset.asset_type}_processed"] = True
                db_session.processing_stages = stages
                
                db.commit()
        
        logger.info(f"ID card processing completed for asset {asset_id}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"ID card processing failed for asset {asset_id}: {error_msg}")
        
        # Update asset with error
        result = {
            "task_type": "id_card_processing",
            "success": False,
            "error_message": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": int(time.time())
        }
        update_asset_processing(asset_id, True, result)
        
        # Update session status
        update_session_status(session_id, "failed", error_message=error_msg)
        
        raise self.retry(exc=e, countdown=60, max_retries=3)
        
    finally:
        db.close()


@celery_app.task(bind=True, name='process_selfie_task')
def process_selfie_task(self, asset_id: str):
    """Process selfie image for face detection and liveness"""
    logger.info(f"Starting selfie processing for asset {asset_id}")
    start_time = time.time()
    
    db = SessionLocal()
    try:
        # Get asset information
        asset = db.query(EKYCAsset).filter(EKYCAsset.id == asset_id).first()
        if not asset:
            raise Exception(f"Asset {asset_id} not found")
        
        session_id = str(asset.session_id)
        
        # Update session status
        update_session_status(session_id, "in_progress")
        
        # Download file for processing
        temp_file_path = get_file_for_processing_sync(asset.file_path)
        if not temp_file_path:
            raise Exception("Failed to download file for processing")
        
        # Load and process image
        image = load_image_from_path(temp_file_path)
        if image is None:
            raise Exception("Failed to load image")
        
        # Face detection
        faces = face_engine.detect_faces(image)
        if not faces:
            raise Exception("No face detected in selfie")
        
        # Liveness detection
        liveness_result = face_engine.detect_liveness(image)
        
        # Extract face embedding
        face_embedding = face_engine.extract_face_embedding(image)
        
        # Clean up temp file
        cleanup_temp_file_sync(temp_file_path)
        
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "task_type": "selfie_processing",
            "success": True,
            "faces_detected": len(faces),
            "face_data": faces[0] if faces else None,
            "liveness_result": liveness_result,
            "face_embedding": face_embedding.tolist() if face_embedding is not None else None,
            "processing_time": processing_time,
            "timestamp": int(time.time())
        }
        
        # Update asset
        update_asset_processing(asset_id, True, result)
        
        # Send webhook notification
        webhook_notifier.notify_asset_processed(
            session_id=session_id,
            asset_id=asset_id,
            asset_type=asset.asset_type,
            user_id=asset.session.user_id if hasattr(asset, 'session') else "unknown",
            success=True,
            processing_result=result
        )
        
        # Update session with liveness score
        db_session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if db_session:
            db_session.liveness_score = liveness_result.get("score", 0.0)
            
            # Update processing stage
            stages = db_session.processing_stages or {}
            stages["selfie_processed"] = True
            db_session.processing_stages = stages
            
            db.commit()
        
        # Check if we can perform face matching
        check_and_perform_face_matching_sync(session_id)
        
        logger.info(f"Selfie processing completed for asset {asset_id}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Selfie processing failed for asset {asset_id}: {error_msg}")
        
        # Update asset with error
        result = {
            "task_type": "selfie_processing",
            "success": False,
            "error_message": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": int(time.time())
        }
        update_asset_processing(asset_id, True, result)
        
        # Update session status
        update_session_status(session_id, "failed", error_message=error_msg)
        
        raise self.retry(exc=e, countdown=60, max_retries=3)
        
    finally:
        db.close()


def extract_id_card_info(image_path: str) -> Dict[str, Any]:
    """Extract information from ID card using OCR"""
    try:
        ocr = get_ocr_engine()
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to read image")
        
        # Run OCR
        result = ocr.ocr(image, cls=True)
        
        # Extract text
        extracted_texts = []
        confidence_scores = []
        
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                confidence = word_info[1][1]
                extracted_texts.append(text)
                confidence_scores.append(confidence)
        
        # Parse extracted information
        id_card_data = parse_vietnamese_id_card(extracted_texts)
        
        return {
            "extracted_data": id_card_data,
            "confidence_score": np.mean(confidence_scores) if confidence_scores else 0.0,
            "raw_texts": extracted_texts
        }
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise


def parse_vietnamese_id_card(texts: List[str]) -> Dict[str, Any]:
    """Parse Vietnamese ID card information from OCR texts"""
    id_data = {}
    confidence_scores = {}
    
    # Join all texts for easier pattern matching
    full_text = " ".join(texts).upper()
    
    # Patterns for Vietnamese ID cards
    patterns = {
        'id_number': [
            r'(?:SỐ|SO|NUMBER)[:\s]*(\d{9,12})',
            r'(\d{9,12})',  # Fallback: any 9-12 digit sequence
        ],
        'full_name': [
            r'(?:HỌ VÀ TÊN|HO VA TEN|NAME)[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]+)',
        ],
        'date_of_birth': [
            r'(?:SINH|BIRTH)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
        ],
        'gender': [
            r'(?:GIỚI TÍNH|GENDER)[:\s]*(NAM|NU|NỮ|MALE|FEMALE)',
        ],
        'nationality': [
            r'(?:QUỐC TỊCH|NATIONALITY)[:\s]*(VIỆT NAM|VIETNAM)',
        ],
        'place_of_origin': [
            r'(?:QUÁN QUÁN|PLACE OF ORIGIN)[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s,]+)',
        ],
        'place_of_residence': [
            r'(?:THƯỜNG TRÚ|RESIDENCE)[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s,]+)',
        ]
    }
    
    # Extract information using patterns
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, full_text)
            if match:
                value = match.group(1).strip()
                if value:
                    id_data[field] = value
                    confidence_scores[field] = 0.8  # Base confidence
                    break
    
    # Clean and validate extracted data
    if 'full_name' in id_data:
        id_data['full_name'] = clean_name(id_data['full_name'])
    
    if 'date_of_birth' in id_data:
        id_data['date_of_birth'] = normalize_date(id_data['date_of_birth'])
    
    if 'gender' in id_data:
        id_data['gender'] = normalize_gender(id_data['gender'])
    
    # Add confidence scores
    id_data['confidence_scores'] = confidence_scores
    
    return id_data


def clean_name(name: str) -> str:
    """Clean and normalize name"""
    # Remove extra spaces and special characters
    name = re.sub(r'[^\w\s]', '', name)
    name = ' '.join(name.split())
    return name.title()


def normalize_date(date_str: str) -> str:
    """Normalize date format"""
    # Convert various date formats to DD/MM/YYYY
    date_str = re.sub(r'[\-\.]', '/', date_str)
    return date_str


def normalize_gender(gender: str) -> str:
    """Normalize gender"""
    gender = gender.upper()
    if gender in ['NAM', 'MALE']:
        return 'Nam'
    elif gender in ['NU', 'NỮ', 'FEMALE']:
        return 'Nữ'
    return gender


def check_and_perform_face_matching_sync(session_id: str):
    """Check if face matching can be performed and execute it"""
    db = SessionLocal()
    try:
        session = db.query(EKYCSession).filter(EKYCSession.id == session_id).first()
        if not session:
            return
        
        # Get ID card asset with face
        id_asset = db.query(EKYCAsset).filter(
            EKYCAsset.session_id == session_id,
            EKYCAsset.asset_type.in_(['id_front']),
            EKYCAsset.processed == True
        ).first()
        
        # Get selfie asset
        selfie_asset = db.query(EKYCAsset).filter(
            EKYCAsset.session_id == session_id,
            EKYCAsset.asset_type == 'selfie',
            EKYCAsset.processed == True
        ).first()
        
        if not id_asset or not selfie_asset:
            logger.info(f"Face matching not ready for session {session_id}")
            return
        
        # Get face embeddings from processing results
        id_result = id_asset.processing_result or {}
        selfie_result = selfie_asset.processing_result or {}
        
        id_embedding = id_result.get('face_embedding')
        selfie_embedding = selfie_result.get('face_embedding')
        
        if not id_embedding or not selfie_embedding:
            logger.warning(f"Missing face embeddings for session {session_id}")
            return
        
        # Perform face matching
        id_emb = np.array(id_embedding)
        selfie_emb = np.array(selfie_embedding)
        
        is_match, similarity = face_engine.is_same_person(id_emb, selfie_emb)
        
        # Update session with face match result
        session.face_match_score = similarity
        
        # Update processing stage
        stages = session.processing_stages or {}
        stages['face_match_completed'] = True
        session.processing_stages = stages
        
        # Determine final decision
        liveness_passed = session.liveness_score >= settings.liveness_threshold
        face_match_passed = similarity >= settings.face_match_threshold
        
        if liveness_passed and face_match_passed:
            session.final_decision = 'approved'
            session.status = 'completed'
        else:
            session.final_decision = 'rejected'
            session.status = 'completed'
            
            # Set rejection reason
            reasons = []
            if not liveness_passed:
                reasons.append('liveness check failed')
            if not face_match_passed:
                reasons.append('face match failed')
            session.error_message = f"Rejected: {', '.join(reasons)}"
        
        db.commit()
        
        # Send webhook notifications
        webhook_notifier.notify_face_match_completed(
            session_id=session_id,
            user_id=session.user_id,
            similarity_score=similarity,
            is_match=is_match
        )
        
        # Send session completion notification
        if session.final_decision == 'approved':
            webhook_notifier.notify_session_completed(
                session_id=session_id,
                user_id=session.user_id,
                final_decision=session.final_decision,
                face_match_score=session.face_match_score,
                liveness_score=session.liveness_score
            )
        else:
            webhook_notifier.notify_session_failed(
                session_id=session_id,
                user_id=session.user_id,
                error_message=session.error_message or "EKYC verification failed"
            )
        
        logger.info(f"Face matching completed for session {session_id}: {session.final_decision}")
        
    except Exception as e:
        logger.error(f"Face matching failed for session {session_id}: {str(e)}")
        db.rollback()
    finally:
        db.close()
