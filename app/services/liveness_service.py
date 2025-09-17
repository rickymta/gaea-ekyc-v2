"""
Liveness Detection Service
Sử dụng silent_face.anti_spoof_predict để phát hiện liveness
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import os
from pathlib import Path
import logging

try:
    from silent_face.anti_spoof_predict import AntiSpoofPredict
    from silent_face.generate_patches import CropImage
    from silent_face.utility import parse_model_name
except ImportError:
    # Fallback nếu không có silent_face
    AntiSpoofPredict = None
    CropImage = None
    parse_model_name = None

from app.config import settings

logger = logging.getLogger(__name__)

class LivenessDetectionService:
    """Service để phát hiện liveness từ video hoặc ảnh"""
    
    def __init__(self):
        self.model_path = Path(settings.models_dir) / "silent_face"
        self.device_id = 0  # CPU
        self.model = None
        self.image_cropper = None
        
        # Thresholds cho các model
        self.thresholds = {
            "2.7": 0.7,
            "4.0": 0.9
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Khởi tạo model Silent Face Anti-Spoofing"""
        try:
            if AntiSpoofPredict is None:
                logger.warning("Silent Face library not available, using fallback detection")
                return
                
            # Tạo thư mục model nếu chưa có
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Download model nếu chưa có
            self._download_models_if_needed()
            
            # Khởi tạo model
            self.model = AntiSpoofPredict(self.device_id)
            self.image_cropper = CropImage()
            
            logger.info("Silent Face Anti-Spoofing model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize liveness detection model: {e}")
            self.model = None
    
    def _download_models_if_needed(self):
        """Download models nếu chưa có"""
        model_files = [
            "2.7_80x80_MiniFASNetV2.pth",
            "4_0_0_80x80_MiniFASNetV1SE.pth"
        ]
        
        for model_file in model_files:
            model_file_path = self.model_path / model_file
            if not model_file_path.exists():
                logger.info(f"Model {model_file} not found, please download it manually")
                # Bạn có thể thêm logic download tự động ở đây
    
    async def detect_liveness_from_video(
        self, 
        video_data: bytes,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Phát hiện liveness từ video ngắn 3s
        
        Args:
            video_data: Video data bytes
            confidence_threshold: Threshold để xác định live/spoof
            
        Returns:
            Dict với kết quả liveness detection
        """
        try:
            # Lưu video tạm
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_data)
                temp_video_path = temp_video.name
            
            try:
                # Phân tích video
                result = await self._analyze_video(temp_video_path, confidence_threshold)
                return result
                
            finally:
                # Xóa file tạm
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                    
        except Exception as e:
            logger.error(f"Error in liveness detection from video: {e}")
            return {
                "is_live": False,
                "confidence": 0.0,
                "error": str(e),
                "analysis": {
                    "total_frames": 0,
                    "live_frames": 0,
                    "spoof_frames": 0,
                    "scores": []
                }
            }
    
    async def _analyze_video(
        self, 
        video_path: str, 
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Phân tích video để phát hiện liveness"""
        
        if self.model is None:
            # Fallback detection nếu không có model
            return await self._fallback_liveness_detection(video_path)
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            frames_analyzed = 0
            live_frames = 0
            spoof_frames = 0
            scores = []
            
            # Lấy thông tin video
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Chỉ phân tích một số frame để tối ưu
            frame_skip = max(1, int(fps / 10))  # Phân tích 10 frames/giây
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames để tối ưu
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Phát hiện liveness cho frame này
                frame_result = await self._detect_frame_liveness(frame)
                
                if frame_result["face_detected"]:
                    frames_analyzed += 1
                    score = frame_result["liveness_score"]
                    scores.append(score)
                    
                    if score >= confidence_threshold:
                        live_frames += 1
                    else:
                        spoof_frames += 1
                
                frame_idx += 1
                
                # Tối đa phân tích 30 frames
                if frames_analyzed >= 30:
                    break
            
            cap.release()
            
            # Tính toán kết quả cuối cùng
            if frames_analyzed == 0:
                return {
                    "is_live": False,
                    "confidence": 0.0,
                    "error": "No faces detected in video",
                    "analysis": {
                        "total_frames": total_frames,
                        "analyzed_frames": 0,
                        "live_frames": 0,
                        "spoof_frames": 0,
                        "scores": []
                    }
                }
            
            # Tính confidence trung bình
            avg_confidence = np.mean(scores)
            
            # Xác định live nếu > 70% frames là live
            live_ratio = live_frames / frames_analyzed
            is_live = live_ratio >= 0.7 and avg_confidence >= confidence_threshold
            
            return {
                "is_live": is_live,
                "confidence": float(avg_confidence),
                "live_ratio": float(live_ratio),
                "analysis": {
                    "total_frames": total_frames,
                    "analyzed_frames": frames_analyzed,
                    "live_frames": live_frames,
                    "spoof_frames": spoof_frames,
                    "scores": scores,
                    "avg_score": float(avg_confidence),
                    "threshold_used": confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video for liveness: {e}")
            raise
    
    async def _detect_frame_liveness(self, frame: np.ndarray) -> Dict[str, Any]:
        """Phát hiện liveness cho một frame"""
        try:
            # Crop và chuẩn bị ảnh
            image_bbox = self.model.get_bbox(frame)
            
            if image_bbox is None:
                return {
                    "face_detected": False,
                    "liveness_score": 0.0
                }
            
            # Prediction
            prediction = np.zeros((1, 3))
            
            # Test với các model khác nhau
            for model_name in ["2.7", "4.0"]:
                try:
                    # Crop ảnh theo model
                    param = {
                        "org_img": frame,
                        "bbox": image_bbox,
                        "scale": parse_model_name(model_name, 0, "scale"),
                        "out_w": parse_model_name(model_name, 0, "out_w"),
                        "out_h": parse_model_name(model_name, 0, "out_h"),
                        "crop": True,
                    }
                    
                    if "org_img" in param:
                        cropped_img = self.image_cropper.crop(**param)
                        pred = self.model.predict(cropped_img, model_name)
                        prediction += pred
                        
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {e}")
                    continue
            
            # Tính toán kết quả
            prediction_avg = prediction / len(["2.7", "4.0"])
            label = np.argmax(prediction_avg)
            value = prediction_avg[0][label]
            
            # Label 1 = Live, Label 0 = Spoof
            is_live = label == 1
            confidence = float(value)
            
            return {
                "face_detected": True,
                "liveness_score": confidence if is_live else 1.0 - confidence,
                "is_live": is_live,
                "raw_prediction": prediction_avg.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in frame liveness detection: {e}")
            return {
                "face_detected": False,
                "liveness_score": 0.0,
                "error": str(e)
            }
    
    async def _fallback_liveness_detection(self, video_path: str) -> Dict[str, Any]:
        """
        Fallback liveness detection khi không có Silent Face model
        Sử dụng các heuristics đơn giản
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            # Khởi tạo face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            frames_with_faces = 0
            total_frames = 0
            movement_scores = []
            brightness_scores = []
            
            prev_gray = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    frames_with_faces += 1
                    
                    # Tính movement (optical flow)
                    if prev_gray is not None:
                        flow = cv2.calcOpticalFlowPyrLK(
                            prev_gray, gray, None, None
                        )
                        if flow[0] is not None:
                            movement = np.mean(np.abs(flow[0]))
                            movement_scores.append(movement)
                    
                    # Tính brightness variation
                    face_region = gray[faces[0][1]:faces[0][1]+faces[0][3], 
                                     faces[0][0]:faces[0][0]+faces[0][2]]
                    brightness = np.mean(face_region)
                    brightness_scores.append(brightness)
                
                prev_gray = gray.copy()
                
                # Giới hạn frames để tối ưu
                if total_frames >= 90:  # ~3 seconds at 30fps
                    break
            
            cap.release()
            
            # Tính toán heuristics
            if frames_with_faces < total_frames * 0.5:
                return {
                    "is_live": False,
                    "confidence": 0.3,
                    "reason": "Insufficient face detection",
                    "analysis": {
                        "total_frames": total_frames,
                        "frames_with_faces": frames_with_faces,
                        "detection_ratio": frames_with_faces / max(total_frames, 1)
                    }
                }
            
            # Đánh giá movement (liveness thường có movement tự nhiên)
            movement_score = 0.5
            if movement_scores:
                avg_movement = np.mean(movement_scores)
                movement_score = min(1.0, avg_movement / 10.0)  # Normalize
            
            # Đánh giá brightness variation
            brightness_score = 0.5
            if brightness_scores:
                brightness_var = np.var(brightness_scores)
                brightness_score = min(1.0, brightness_var / 100.0)  # Normalize
            
            # Tổng hợp score
            overall_confidence = (movement_score + brightness_score) / 2.0
            is_live = overall_confidence >= 0.6
            
            return {
                "is_live": is_live,
                "confidence": float(overall_confidence),
                "analysis": {
                    "total_frames": total_frames,
                    "frames_with_faces": frames_with_faces,
                    "movement_score": float(movement_score),
                    "brightness_score": float(brightness_score),
                    "method": "fallback_heuristics"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fallback liveness detection: {e}")
            return {
                "is_live": False,
                "confidence": 0.0,
                "error": str(e),
                "analysis": {"method": "fallback_heuristics"}
            }
    
    async def detect_liveness_from_image(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Phát hiện liveness từ ảnh đơn
        
        Args:
            image: Ảnh input
            confidence_threshold: Threshold để xác định live/spoof
            
        Returns:
            Dict với kết quả liveness detection
        """
        try:
            if self.model is None:
                # Fallback cho ảnh đơn
                return {
                    "is_live": True,  # Giả định live cho ảnh đơn
                    "confidence": 0.5,
                    "method": "fallback_single_image"
                }
            
            result = await self._detect_frame_liveness(image)
            
            if not result["face_detected"]:
                return {
                    "is_live": False,
                    "confidence": 0.0,
                    "error": "No face detected in image"
                }
            
            is_live = result["liveness_score"] >= confidence_threshold
            
            return {
                "is_live": is_live,
                "confidence": result["liveness_score"],
                "face_detected": True,
                "threshold_used": confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in image liveness detection: {e}")
            return {
                "is_live": False,
                "confidence": 0.0,
                "error": str(e)
            }


# Singleton instance
liveness_detection_service = LivenessDetectionService()
