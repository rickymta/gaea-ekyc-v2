"""
Enhanced EKYC API Endpoints
API hoàn chỉnh cho verification với liveness detection, OCR và quality assessment
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image
import io
import logging
import uuid
from datetime import datetime
import asyncio

from app.services.face_service import enhanced_face_engine
from app.database.models import EKYCVerification
from app.database.connection import get_db_connection
from app.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create FastAPI router
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1/ekyc", tags=["Enhanced EKYC"])


def process_uploaded_image(upload_file: UploadFile) -> np.ndarray:
    """Convert uploaded file to OpenCV image format"""
    try:
        # Read file content
        contents = upload_file.file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
        
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


def process_uploaded_video(upload_file: UploadFile) -> bytes:
    """Process uploaded video file"""
    try:
        # Read video content
        contents = upload_file.file.read()
        
        # Validate video size (max 10MB for 3s video)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Video file too large (max 10MB)")
        
        return contents
        
    except Exception as e:
        logger.error(f"Error processing uploaded video: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid video format: {str(e)}")


async def save_verification_to_db(verification_result: dict, session_id: str):
    """Save verification result to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert verification record
        insert_query = """
        INSERT INTO ekyc_verifications (
            session_id, 
            overall_decision, 
            confidence_score, 
            processing_time_ms,
            selfie_quality_score,
            id_card_quality_score,
            face_match_score,
            liveness_score,
            ocr_success,
            checks_passed,
            total_checks,
            failed_checks,
            recommendations,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            session_id,
            verification_result.get("overall_decision"),
            verification_result.get("confidence_score", 0.0),
            verification_result.get("processing_time_ms", 0),
            verification_result.get("quality_assessment", {}).get("selfie", {}).get("score", 0.0),
            verification_result.get("quality_assessment", {}).get("id_card", {}).get("score", 0.0),
            verification_result.get("face_matching", {}).get("similarity_score", 0.0),
            verification_result.get("liveness_detection", {}).get("confidence", 0.0),
            verification_result.get("ocr_results", {}).get("success", False),
            verification_result.get("verification_summary", {}).get("checks_passed", 0),
            verification_result.get("verification_summary", {}).get("total_checks", 7),
            str(verification_result.get("verification_summary", {}).get("failed_checks", [])),
            str(verification_result.get("recommendations", [])),
            datetime.now()
        )
        
        cursor.execute(insert_query, values)
        conn.commit()
        
        logger.info(f"Verification result saved to database for session: {session_id}")
        
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


@router.post("/verify-complete")
async def verify_complete_ekyc(
    background_tasks: BackgroundTasks,
    selfie_image: UploadFile = File(..., description="Selfie image file"),
    id_front_image: UploadFile = File(..., description="ID card front image"),
    id_back_image: Optional[UploadFile] = File(None, description="ID card back image (optional)"),
    liveness_video: Optional[UploadFile] = File(None, description="3-second video for liveness detection (optional)"),
    session_id: Optional[str] = Form(None, description="Optional session ID")
):
    """
    Complete EKYC Verification với đầy đủ tính năng:
    - Liveness detection từ video hoặc selfie
    - OCR thông tin CCCD
    - Quality assessment cho selfie và ID card
    - Face matching giữa selfie và CCCD
    - Comprehensive decision making
    
    Args:
        selfie_image: Ảnh selfie (required)
        id_front_image: Ảnh mặt trước CCCD (required)
        id_back_image: Ảnh mặt sau CCCD (optional)
        liveness_video: Video 3s cho liveness detection (optional)
        session_id: ID session (auto-generated if not provided)
    
    Returns:
        Comprehensive verification result với decision, scores, và recommendations
    """
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logger.info(f"Starting complete EKYC verification for session: {session_id}")
    
    try:
        # Process uploaded images
        logger.info("Processing uploaded files...")
        
        # Required images
        selfie_cv = process_uploaded_image(selfie_image)
        id_front_cv = process_uploaded_image(id_front_image)
        
        # Optional images/video
        id_back_cv = None
        if id_back_image:
            id_back_cv = process_uploaded_image(id_back_image)
        
        liveness_video_bytes = None
        if liveness_video:
            liveness_video_bytes = process_uploaded_video(liveness_video)
        
        # Perform complete EKYC verification
        logger.info("Performing complete EKYC verification...")
        verification_result = await enhanced_face_engine.perform_complete_ekyc_verification(
            selfie_image=selfie_cv,
            id_front_image=id_front_cv,
            id_back_image=id_back_cv,
            liveness_video=liveness_video_bytes,
            session_id=session_id
        )
        
        # Save to database in background
        background_tasks.add_task(save_verification_to_db, verification_result, session_id)
        
        # Return result
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "EKYC verification completed successfully",
                "data": verification_result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in complete EKYC verification: {e}")
        
        error_result = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "overall_decision": "ERROR",
            "confidence_score": 0.0,
            "error": str(e),
            "verification_summary": {
                "checks_passed": 0,
                "total_checks": 7,
                "failed_checks": ["system_error"],
                "warnings": []
            },
            "recommendations": ["Please try again or contact support"]
        }
        
        # Save error to database in background
        background_tasks.add_task(save_verification_to_db, error_result, session_id)
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error during EKYC verification",
                "data": error_result
            }
        )


@router.post("/verify-liveness-only")
async def verify_liveness_only(
    image_or_video: UploadFile = File(..., description="Image or video file for liveness detection"),
    session_id: Optional[str] = Form(None, description="Optional session ID")
):
    """
    Liveness Detection Only
    Chỉ thực hiện liveness detection từ ảnh hoặc video
    """
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Check if uploaded file is video or image based on content type
        content_type = image_or_video.content_type
        
        if content_type and content_type.startswith('video/'):
            # Process as video
            video_bytes = process_uploaded_video(image_or_video)
            
            from app.services.liveness_service import liveness_detection_service
            result = await liveness_detection_service.detect_liveness_from_video(
                video_bytes,
                confidence_threshold=settings.liveness_threshold
            )
        else:
            # Process as image
            image_cv = process_uploaded_image(image_or_video)
            
            from app.services.liveness_service import liveness_detection_service
            result = await liveness_detection_service.detect_liveness_from_image(
                image_cv,
                confidence_threshold=settings.liveness_threshold
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Liveness detection completed successfully",
                "data": {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "liveness_result": result
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in liveness detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Liveness detection failed: {str(e)}"
        )


@router.post("/extract-id-info")
async def extract_id_card_info(
    id_front_image: UploadFile = File(..., description="ID card front image"),
    id_back_image: Optional[UploadFile] = File(None, description="ID card back image (optional)"),
    session_id: Optional[str] = Form(None, description="Optional session ID")
):
    """
    OCR ID Card Information
    Trích xuất thông tin từ CCCD sử dụng OCR
    """
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Process uploaded images
        id_front_cv = process_uploaded_image(id_front_image)
        
        id_back_cv = None
        if id_back_image:
            id_back_cv = process_uploaded_image(id_back_image)
        
        # Extract ID card information
        from app.services.ocr_service import vietnam_id_ocr_service
        ocr_result = await vietnam_id_ocr_service.extract_id_card_info(
            id_front_cv,
            id_back_cv
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "ID card information extracted successfully",
                "data": {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "ocr_result": ocr_result
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in ID card OCR: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ID card OCR failed: {str(e)}"
        )


@router.post("/assess-image-quality")
async def assess_image_quality(
    image: UploadFile = File(..., description="Image to assess quality"),
    image_type: str = Form("selfie", description="Type of image: 'selfie' or 'id_card'"),
    session_id: Optional[str] = Form(None, description="Optional session ID")
):
    """
    Image Quality Assessment
    Đánh giá chất lượng ảnh (selfie hoặc CCCD)
    """
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if image_type not in ["selfie", "id_card"]:
        raise HTTPException(
            status_code=400,
            detail="image_type must be either 'selfie' or 'id_card'"
        )
    
    try:
        # Process uploaded image
        image_cv = process_uploaded_image(image)
        
        # Assess image quality
        from app.services.image_quality_service import image_quality_service
        
        if image_type == "selfie":
            quality_result = await image_quality_service.assess_selfie_quality(image_cv)
        else:
            quality_result = await image_quality_service.assess_id_card_quality(image_cv)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"{image_type.capitalize()} quality assessment completed successfully",
                "data": {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "image_type": image_type,
                    "quality_result": quality_result
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in image quality assessment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image quality assessment failed: {str(e)}"
        )


@router.post("/face-match")
async def face_match_only(
    selfie_image: UploadFile = File(..., description="Selfie image"),
    id_image: UploadFile = File(..., description="ID card image"),
    session_id: Optional[str] = Form(None, description="Optional session ID")
):
    """
    Face Matching Only
    Chỉ thực hiện so sánh khuôn mặt giữa selfie và CCCD
    """
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Process uploaded images
        selfie_cv = process_uploaded_image(selfie_image)
        id_cv = process_uploaded_image(id_image)
        
        # Detect faces
        selfie_faces = enhanced_face_engine.detect_faces(selfie_cv)
        id_faces = enhanced_face_engine.detect_faces(id_cv)
        
        if not selfie_faces:
            raise HTTPException(status_code=400, detail="No face detected in selfie image")
        
        if not id_faces:
            raise HTTPException(status_code=400, detail="No face detected in ID card image")
        
        # Perform face matching
        face_match_result = await enhanced_face_engine._perform_face_matching(
            selfie_faces[0],
            id_faces[0],
            selfie_cv,
            id_cv
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Face matching completed successfully",
                "data": {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "selfie_faces_detected": len(selfie_faces),
                    "id_faces_detected": len(id_faces),
                    "face_matching": face_match_result
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face matching: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Face matching failed: {str(e)}"
        )


@router.post("/face-match-simple", tags=["Simple Face Matching"])
async def face_match_simple(
    selfie_image: UploadFile = File(..., description="Selfie image"),
    id_image: UploadFile = File(..., description="ID card image"),
    session_id: Optional[str] = Form(None, description="Optional session ID"),
    skip_quality_check: bool = Form(False, description="Skip image quality assessment"),
    return_detailed_analysis: bool = Form(False, description="Return detailed face analysis")
):
    """
    Simple Face Matching API (No Liveness Detection)
    
    API đơn giản chỉ thực hiện face matching giữa selfie và CCCD mà không có liveness detection.
    Phù hợp cho các trường hợp chỉ cần xác minh khuôn mặt cơ bản.
    
    Features:
    - ✅ Face detection và extraction
    - ✅ Face similarity comparison  
    - ✅ Configurable quality checks
    - ✅ Detailed face analysis (optional)
    - ❌ NO liveness detection
    - ❌ NO OCR processing
    - ❌ NO comprehensive EKYC workflow
    
    Args:
        selfie_image: Ảnh selfie (required)
        id_image: Ảnh CCCD/ID (required)
        session_id: ID session (auto-generated if not provided)
        skip_quality_check: Bỏ qua kiểm tra chất lượng ảnh (default: False)
        return_detailed_analysis: Trả về phân tích chi tiết (default: False)
    
    Returns:
        Simplified face matching result without liveness detection
    """
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logger.info(f"Starting simple face matching for session: {session_id}")
    
    try:
        start_time = datetime.now()
        
        # Process uploaded images
        logger.info("Processing uploaded images...")
        selfie_cv = process_uploaded_image(selfie_image)
        id_cv = process_uploaded_image(id_image)
        
        result = {
            "session_id": session_id,
            "timestamp": start_time.isoformat(),
            "api_type": "face-match-simple",
            "success": False,
            "processing_time_ms": 0,
            
            # Core results
            "face_matching": {},
            "selfie_analysis": {},
            "id_analysis": {},
            
            # Optional detailed analysis
            "detailed_analysis": None,
            "quality_assessment": None,
            
            # Summary
            "summary": {
                "is_match": False,
                "confidence_level": "LOW",
                "recommendation": ""
            }
        }
        
        # 1. Image Quality Assessment (if not skipped)
        if not skip_quality_check:
            logger.info("Performing basic quality checks...")
            try:
                from app.services.image_quality_service import image_quality_service
                
                selfie_quality = await image_quality_service.assess_selfie_quality(selfie_cv)
                id_quality = await image_quality_service.assess_id_card_quality(id_cv)
                
                result["quality_assessment"] = {
                    "selfie": {
                        "score": selfie_quality.get("score", 0.0),
                        "quality": selfie_quality.get("overall_quality", "unknown"),
                        "main_issues": selfie_quality.get("recommendations", [])[:3]  # Top 3 issues
                    },
                    "id_card": {
                        "score": id_quality.get("score", 0.0),
                        "quality": id_quality.get("overall_quality", "unknown"),
                        "main_issues": id_quality.get("recommendations", [])[:3]  # Top 3 issues
                    }
                }
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")
                result["quality_assessment"] = {"error": "Quality assessment unavailable"}
        
        # 2. Face Detection - Selfie
        logger.info("Detecting faces in selfie...")
        selfie_faces = enhanced_face_engine.detect_faces(selfie_cv)
        result["selfie_analysis"] = {
            "faces_detected": len(selfie_faces),
            "primary_face": selfie_faces[0] if selfie_faces else None,
            "status": "success" if len(selfie_faces) == 1 else "error",
            "message": (
                "Face detected successfully" if len(selfie_faces) == 1
                else f"Expected 1 face, found {len(selfie_faces)}"
            )
        }
        
        # 3. Face Detection - ID Card
        logger.info("Detecting faces in ID card...")
        id_faces = enhanced_face_engine.detect_faces(id_cv)
        result["id_analysis"] = {
            "faces_detected": len(id_faces),
            "primary_face": id_faces[0] if id_faces else None,
            "status": "success" if len(id_faces) == 1 else "error",
            "message": (
                "Face detected successfully" if len(id_faces) == 1
                else f"Expected 1 face, found {len(id_faces)}"
            )
        }
        
        # 4. Face Matching (if both faces detected)
        if selfie_faces and id_faces:
            logger.info("Performing face matching...")
            face_match_result = await enhanced_face_engine._perform_face_matching(
                selfie_faces[0],
                id_faces[0],
                selfie_cv,
                id_cv
            )
            
            result["face_matching"] = face_match_result
            result["success"] = True
            
            # Update summary
            result["summary"]["is_match"] = face_match_result.get("is_match", False)
            result["summary"]["confidence_level"] = face_match_result.get("confidence_level", "LOW")
            
            # Generate recommendation
            if face_match_result.get("is_match", False):
                if face_match_result.get("similarity_score", 0) >= 0.8:
                    result["summary"]["recommendation"] = "Strong match - Identity verified with high confidence"
                elif face_match_result.get("similarity_score", 0) >= 0.6:
                    result["summary"]["recommendation"] = "Good match - Identity verified with medium confidence"
                else:
                    result["summary"]["recommendation"] = "Weak match - Consider additional verification"
            else:
                result["summary"]["recommendation"] = "No match - Different persons detected"
                
        else:
            # Face detection failed
            result["face_matching"] = {
                "is_match": False,
                "similarity_score": 0.0,
                "confidence_level": "LOW",
                "error": "Insufficient faces detected for comparison"
            }
            result["summary"]["recommendation"] = "Cannot perform matching - Face detection failed"
        
        # 5. Detailed Analysis (if requested)
        if return_detailed_analysis and selfie_faces and id_faces:
            logger.info("Generating detailed analysis...")
            
            selfie_face = selfie_faces[0]
            id_face = id_faces[0]
            
            result["detailed_analysis"] = {
                "selfie_face_details": {
                    "bbox": selfie_face.get("bbox"),
                    "confidence": selfie_face.get("confidence", 0.0),
                    "age": selfie_face.get("age"),
                    "gender": selfie_face.get("gender"),
                    "has_landmarks": selfie_face.get("landmarks") is not None,
                    "has_embedding": selfie_face.get("embedding") is not None
                },
                "id_face_details": {
                    "bbox": id_face.get("bbox"),
                    "confidence": id_face.get("confidence", 0.0),
                    "age": id_face.get("age"),
                    "gender": id_face.get("gender"),
                    "has_landmarks": id_face.get("landmarks") is not None,
                    "has_embedding": id_face.get("embedding") is not None
                },
                "comparison_metrics": {
                    "similarity_score": result["face_matching"].get("similarity_score", 0.0),
                    "threshold_used": result["face_matching"].get("threshold_used", 0.6),
                    "confidence_delta": result["face_matching"].get("similarity_score", 0.0) - result["face_matching"].get("threshold_used", 0.6),
                    "face_quality_difference": abs(selfie_face.get("confidence", 0.0) - id_face.get("confidence", 0.0))
                }
            }
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result["processing_time_ms"] = int(processing_time)
        
        logger.info(f"Simple face matching completed in {processing_time:.0f}ms - Match: {result['summary']['is_match']}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Simple face matching completed successfully",
                "data": result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simple face matching: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000 if 'start_time' in locals() else 0
        
        error_result = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "api_type": "face-match-simple",
            "success": False,
            "processing_time_ms": int(processing_time),
            "error": str(e),
            "face_matching": {
                "is_match": False,
                "similarity_score": 0.0,
                "confidence_level": "LOW",
                "error": str(e)
            },
            "summary": {
                "is_match": False,
                "confidence_level": "LOW",
                "recommendation": "Processing error - Please try again"
            }
        }
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Simple face matching failed",
                "data": error_result
            }
        )


@router.get("/verification-status/{session_id}")
async def get_verification_status(session_id: str):
    """
    Get Verification Status
    Lấy trạng thái verification theo session ID
    """
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query verification result
        query = """
        SELECT * FROM ekyc_verifications 
        WHERE session_id = ? 
        ORDER BY created_at DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (session_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No verification found for session ID: {session_id}"
            )
        
        # Convert result to dict
        columns = [description[0] for description in cursor.description]
        verification_data = dict(zip(columns, result))
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Verification status retrieved successfully",
                "data": verification_data
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving verification status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve verification status: {str(e)}"
        )
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/health")
async def health_check():
    """
    Health Check
    Kiểm tra tình trạng hoạt động của Enhanced EKYC API
    """
    
    try:
        # Test services
        services_status = {
            "enhanced_face_engine": "OK" if enhanced_face_engine else "ERROR",
            "liveness_service": "OK",
            "ocr_service": "OK", 
            "image_quality_service": "OK",
            "database": "OK"
        }
        
        # Test database connection
        try:
            conn = get_db_connection()
            conn.close()
        except:
            services_status["database"] = "ERROR"
        
        overall_status = "OK" if all(status == "OK" for status in services_status.values()) else "DEGRADED"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Enhanced EKYC API Health Check",
                "data": {
                    "overall_status": overall_status,
                    "timestamp": datetime.now().isoformat(),
                    "services": services_status,
                    "version": "2.0.0"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Health check failed",
                "data": {
                    "overall_status": "ERROR",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
        )
