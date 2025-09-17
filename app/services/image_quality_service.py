"""
Image Quality Assessment Service
Đánh giá chất lượng ảnh selfie và CCCD theo các tiêu chí cụ thể
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageQualityAssessment:
    """Service đánh giá chất lượng ảnh cho EKYC"""
    
    def __init__(self):
        # Khởi tạo face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Thêm cascade cho mắt và profile face
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize face detectors: {e}")
            self.face_cascade = None
            self.eye_cascade = None
            self.profile_cascade = None
    
    async def assess_selfie_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Đánh giá chất lượng ảnh selfie theo các tiêu chí:
        - Đủ sáng, không tối
        - Khuôn mặt không bị che hoặc cản trở
        - Không đeo kính
        - Không đội mũ
        - Có mặc áo (không cởi trần)
        
        Args:
            image: Ảnh selfie input
            
        Returns:
            Dict với kết quả đánh giá chất lượng
        """
        try:
            result = {
                "overall_quality": "good",
                "score": 0.0,
                "checks": {
                    "brightness": {"passed": False, "score": 0.0, "message": ""},
                    "face_detection": {"passed": False, "score": 0.0, "message": ""},
                    "face_obstruction": {"passed": False, "score": 0.0, "message": ""},
                    "glasses_detection": {"passed": False, "score": 0.0, "message": ""},
                    "hat_detection": {"passed": False, "score": 0.0, "message": ""},
                    "clothing_detection": {"passed": False, "score": 0.0, "message": ""}
                },
                "recommendations": []
            }
            
            if image is None or image.size == 0:
                result["overall_quality"] = "poor"
                result["checks"]["face_detection"]["message"] = "Invalid image"
                return result
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. Kiểm tra độ sáng
            brightness_result = self._check_brightness(image, gray)
            result["checks"]["brightness"] = brightness_result
            
            # 2. Detect faces
            faces = self._detect_faces(gray)
            
            if len(faces) == 0:
                result["overall_quality"] = "poor"
                result["checks"]["face_detection"] = {
                    "passed": False, 
                    "score": 0.0, 
                    "message": "No face detected"
                }
                result["recommendations"].append("Ensure your face is clearly visible in the image")
                return result
            
            elif len(faces) > 1:
                result["overall_quality"] = "poor"
                result["checks"]["face_detection"] = {
                    "passed": False, 
                    "score": 0.0, 
                    "message": "Multiple faces detected"
                }
                result["recommendations"].append("Only one person should be in the photo")
                return result
            
            # Lấy face chính
            main_face = faces[0]
            result["checks"]["face_detection"] = {
                "passed": True, 
                "score": 1.0, 
                "message": "Single face detected successfully"
            }
            
            # 3. Kiểm tra face obstruction
            obstruction_result = self._check_face_obstruction(image, gray, main_face)
            result["checks"]["face_obstruction"] = obstruction_result
            
            # 4. Kiểm tra kính
            glasses_result = self._detect_glasses(gray, main_face)
            result["checks"]["glasses_detection"] = glasses_result
            
            # 5. Kiểm tra mũ
            hat_result = self._detect_hat(image, main_face)
            result["checks"]["hat_detection"] = hat_result
            
            # 6. Kiểm tra áo
            clothing_result = self._detect_clothing(image, main_face)
            result["checks"]["clothing_detection"] = clothing_result
            
            # Tính overall score và quality
            overall_score, overall_quality = self._calculate_overall_quality(result["checks"])
            result["score"] = overall_score
            result["overall_quality"] = overall_quality
            
            # Tạo recommendations
            result["recommendations"] = self._generate_recommendations(result["checks"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in selfie quality assessment: {e}")
            return {
                "overall_quality": "poor",
                "score": 0.0,
                "checks": {},
                "recommendations": [f"Error in quality assessment: {str(e)}"],
                "error": str(e)
            }
    
    async def assess_id_card_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Đánh giá chất lượng ảnh CCCD theo các tiêu chí:
        - Chụp nét, không bị mờ
        - Không bị cắt góc
        - Không bị dán đè ảnh
        - Đầy đủ thông tin có thể đọc được
        
        Args:
            image: Ảnh CCCD input
            
        Returns:
            Dict với kết quả đánh giá chất lượng
        """
        try:
            result = {
                "overall_quality": "good",
                "score": 0.0,
                "checks": {
                    "sharpness": {"passed": False, "score": 0.0, "message": ""},
                    "corners": {"passed": False, "score": 0.0, "message": ""},
                    "occlusion": {"passed": False, "score": 0.0, "message": ""},
                    "readability": {"passed": False, "score": 0.0, "message": ""},
                    "aspect_ratio": {"passed": False, "score": 0.0, "message": ""}
                },
                "recommendations": []
            }
            
            if image is None or image.size == 0:
                result["overall_quality"] = "poor"
                result["checks"]["sharpness"]["message"] = "Invalid image"
                return result
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. Kiểm tra độ nét
            sharpness_result = self._check_sharpness(gray)
            result["checks"]["sharpness"] = sharpness_result
            
            # 2. Kiểm tra góc cắt
            corners_result = self._check_corners(gray)
            result["checks"]["corners"] = corners_result
            
            # 3. Kiểm tra che khuất
            occlusion_result = self._check_occlusion(image)
            result["checks"]["occlusion"] = occlusion_result
            
            # 4. Kiểm tra khả năng đọc
            readability_result = self._check_readability(gray)
            result["checks"]["readability"] = readability_result
            
            # 5. Kiểm tra tỷ lệ khung hình
            aspect_result = self._check_aspect_ratio(image)
            result["checks"]["aspect_ratio"] = aspect_result
            
            # Tính overall score và quality
            overall_score, overall_quality = self._calculate_overall_quality(result["checks"])
            result["score"] = overall_score
            result["overall_quality"] = overall_quality
            
            # Tạo recommendations
            result["recommendations"] = self._generate_recommendations(result["checks"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ID card quality assessment: {e}")
            return {
                "overall_quality": "poor",
                "score": 0.0,
                "checks": {},
                "recommendations": [f"Error in quality assessment: {str(e)}"],
                "error": str(e)
            }
    
    def _detect_faces(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        if self.face_cascade is None:
            return []
        
        try:
            faces = self.face_cascade.detectMultiScale(
                gray_image, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def _check_brightness(self, color_image: np.ndarray, gray_image: np.ndarray) -> Dict[str, Any]:
        """Kiểm tra độ sáng của ảnh"""
        try:
            # Tính độ sáng trung bình
            mean_brightness = np.mean(gray_image)
            
            # Tính histogram để phân tích phân bố độ sáng
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            
            # Tính tỷ lệ pixel tối (< 50)
            dark_pixels = np.sum(hist[:50]) / (gray_image.shape[0] * gray_image.shape[1])
            
            # Tính tỷ lệ pixel sáng (> 200)
            bright_pixels = np.sum(hist[200:]) / (gray_image.shape[0] * gray_image.shape[1])
            
            # Đánh giá
            if mean_brightness < 80 or dark_pixels > 0.5:
                return {
                    "passed": False,
                    "score": 0.3,
                    "message": "Image too dark",
                    "mean_brightness": float(mean_brightness),
                    "dark_pixel_ratio": float(dark_pixels)
                }
            elif mean_brightness > 200 or bright_pixels > 0.3:
                return {
                    "passed": False,
                    "score": 0.5,
                    "message": "Image too bright",
                    "mean_brightness": float(mean_brightness),
                    "bright_pixel_ratio": float(bright_pixels)
                }
            else:
                score = min(1.0, (mean_brightness - 80) / 120)  # Scale 80-200 to 0-1
                return {
                    "passed": True,
                    "score": float(score),
                    "message": "Good brightness",
                    "mean_brightness": float(mean_brightness)
                }
                
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "message": f"Error checking brightness: {str(e)}"
            }
    
    def _check_face_obstruction(
        self, 
        color_image: np.ndarray, 
        gray_image: np.ndarray, 
        face_bbox: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Kiểm tra che khuất khuôn mặt"""
        try:
            x, y, w, h = face_bbox
            face_region = gray_image[y:y+h, x:x+w]
            
            # Detect eyes trong vùng mặt
            eyes = []
            if self.eye_cascade is not None:
                eyes = self.eye_cascade.detectMultiScale(face_region, 1.1, 3)
            
            # Tính độ đồng đều của vùng mặt (để detect shadow/obstruction)
            face_std = np.std(face_region)
            face_mean = np.mean(face_region)
            
            # Kiểm tra gradient (edge detection)
            edges = cv2.Canny(face_region, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            score = 1.0
            messages = []
            
            # Kiểm tra mắt
            if len(eyes) < 2:
                score -= 0.3
                messages.append("Eyes not clearly visible")
            
            # Kiểm tra độ đồng đều (quá thấp = có shadow/obstruction)
            if face_std < 15:
                score -= 0.2
                messages.append("Possible shadow or obstruction detected")
            
            # Kiểm tra edge density (quá thấp = blur/obstruction)
            if edge_density < 0.1:
                score -= 0.3
                messages.append("Face details not clear")
            
            passed = score >= 0.6
            message = "; ".join(messages) if messages else "Face clearly visible"
            
            return {
                "passed": passed,
                "score": max(0.0, float(score)),
                "message": message,
                "eyes_detected": len(eyes),
                "face_std": float(face_std),
                "edge_density": float(edge_density)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "message": f"Error checking face obstruction: {str(e)}"
            }
    
    def _detect_glasses(self, gray_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Detect kính đeo"""
        try:
            x, y, w, h = face_bbox
            
            # Lấy vùng mắt (1/3 trên của mặt)
            eye_region = gray_image[y:y+h//3, x:x+w]
            
            # Detect circular/rectangular shapes (glasses frames)
            circles = cv2.HoughCircles(
                eye_region, 
                cv2.HOUGH_GRADIENT, 
                1, 20, 
                param1=50, param2=30, 
                minRadius=10, maxRadius=50
            )
            
            # Detect edges cho frame
            edges = cv2.Canny(eye_region, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Đếm rectangular contours
            rectangular_shapes = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangle
                    rectangular_shapes += 1
            
            # Tính reflection (glasses often reflect light)
            bright_pixels = np.sum(eye_region > 200) / (eye_region.shape[0] * eye_region.shape[1])
            
            glasses_indicators = 0
            if circles is not None and len(circles[0]) > 0:
                glasses_indicators += 1
            if rectangular_shapes > 2:
                glasses_indicators += 1
            if bright_pixels > 0.1:
                glasses_indicators += 1
            
            glasses_detected = glasses_indicators >= 2
            
            return {
                "passed": not glasses_detected,
                "score": 0.0 if glasses_detected else 1.0,
                "message": "Glasses detected - please remove" if glasses_detected else "No glasses detected",
                "confidence": float(glasses_indicators / 3)
            }
            
        except Exception as e:
            return {
                "passed": True,
                "score": 0.5,
                "message": f"Cannot determine glasses: {str(e)}"
            }
    
    def _detect_hat(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Detect mũ/nón"""
        try:
            x, y, w, h = face_bbox
            
            # Mở rộng vùng lên trên để check hat
            hat_region_y = max(0, y - h//2)
            hat_region = image[hat_region_y:y, max(0, x-w//4):min(image.shape[1], x+w+w//4)]
            
            if hat_region.size == 0:
                return {"passed": True, "score": 1.0, "message": "No hat detected"}
            
            # Convert to HSV để detect màu đặc trưng của mũ
            hsv = cv2.cvtColor(hat_region, cv2.COLOR_BGR2HSV)
            
            # Detect edge patterns typical of hats
            gray_hat = cv2.cvtColor(hat_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_hat, 50, 150)
            
            # Horizontal lines (hat brim)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Đếm horizontal lines
            horizontal_line_pixels = np.sum(horizontal_lines > 0)
            total_pixels = horizontal_lines.shape[0] * horizontal_lines.shape[1]
            horizontal_ratio = horizontal_line_pixels / total_pixels
            
            # Kiểm tra uniformity of color (hats often have uniform color)
            std_deviation = np.std(gray_hat)
            
            hat_indicators = 0
            if horizontal_ratio > 0.05:  # Strong horizontal lines
                hat_indicators += 1
            if std_deviation < 20:  # Uniform color
                hat_indicators += 1
            
            hat_detected = hat_indicators >= 1
            
            return {
                "passed": not hat_detected,
                "score": 0.0 if hat_detected else 1.0,
                "message": "Hat detected - please remove" if hat_detected else "No hat detected",
                "confidence": float(hat_indicators / 2)
            }
            
        except Exception as e:
            return {
                "passed": True,
                "score": 0.5,
                "message": f"Cannot determine hat: {str(e)}"
            }
    
    def _detect_clothing(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Detect có mặc áo hay không"""
        try:
            x, y, w, h = face_bbox
            
            # Lấy vùng dưới mặt để check áo
            clothing_region_y = y + h
            clothing_region_height = min(h, image.shape[0] - clothing_region_y)
            
            if clothing_region_height <= 0:
                return {
                    "passed": False,
                    "score": 0.0,
                    "message": "Cannot determine clothing - insufficient image area"
                }
            
            clothing_region = image[clothing_region_y:clothing_region_y + clothing_region_height, x:x+w]
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2GRAY)
            
            # Detect skin-like colors (indicate bare chest)
            # Skin color ranges in HSV
            lower_skin1 = np.array([0, 20, 70])
            upper_skin1 = np.array([20, 255, 255])
            lower_skin2 = np.array([160, 20, 70])
            upper_skin2 = np.array([180, 255, 255])
            
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
            skin_ratio = skin_pixels / total_pixels
            
            # Detect fabric textures (edges, patterns)
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # Color variety (clothes usually have more color variation than skin)
            color_std = np.std(gray)
            
            clothing_indicators = 0
            
            # Low skin ratio = likely wearing clothes
            if skin_ratio < 0.3:
                clothing_indicators += 2
            elif skin_ratio < 0.5:
                clothing_indicators += 1
            
            # Good edge density = fabric texture
            if edge_density > 0.1:
                clothing_indicators += 1
            
            # Color variation = not uniform skin
            if color_std > 15:
                clothing_indicators += 1
            
            wearing_clothes = clothing_indicators >= 2
            
            return {
                "passed": wearing_clothes,
                "score": 1.0 if wearing_clothes else 0.0,
                "message": "Proper clothing detected" if wearing_clothes else "Please wear appropriate clothing",
                "skin_ratio": float(skin_ratio),
                "confidence": float(clothing_indicators / 4)
            }
            
        except Exception as e:
            return {
                "passed": True,  # Default to pass on error
                "score": 0.5,
                "message": f"Cannot determine clothing: {str(e)}"
            }
    
    def _check_sharpness(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Kiểm tra độ nét của ảnh"""
        try:
            # Sử dụng Laplacian để detect blur
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            sharpness_score = laplacian.var()
            
            # Threshold để xác định blur
            if sharpness_score < 100:
                return {
                    "passed": False,
                    "score": 0.0,
                    "message": "Image is too blurry",
                    "sharpness_score": float(sharpness_score)
                }
            elif sharpness_score < 300:
                score = (sharpness_score - 100) / 200  # Scale 100-300 to 0-1
                return {
                    "passed": True,
                    "score": float(score),
                    "message": "Moderate sharpness",
                    "sharpness_score": float(sharpness_score)
                }
            else:
                return {
                    "passed": True,
                    "score": 1.0,
                    "message": "Good sharpness",
                    "sharpness_score": float(sharpness_score)
                }
                
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "message": f"Error checking sharpness: {str(e)}"
            }
    
    def _check_corners(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Kiểm tra có bị cắt góc không"""
        try:
            # Detect corners using Harris corner detection
            corners = cv2.goodFeaturesToTrack(
                gray_image, 
                maxCorners=100, 
                qualityLevel=0.01, 
                minDistance=10
            )
            
            if corners is None:
                return {
                    "passed": False,
                    "score": 0.0,
                    "message": "No clear corners detected"
                }
            
            # Kiểm tra góc ở 4 vị trí của ảnh
            h, w = gray_image.shape
            corner_regions = [
                (0, 0, w//4, h//4),      # Top-left
                (3*w//4, 0, w, h//4),    # Top-right
                (0, 3*h//4, w//4, h),    # Bottom-left
                (3*w//4, 3*h//4, w, h)   # Bottom-right
            ]
            
            corners_found = 0
            for x1, y1, x2, y2 in corner_regions:
                region_corners = corners[
                    (corners[:, 0, 0] >= x1) & (corners[:, 0, 0] <= x2) &
                    (corners[:, 0, 1] >= y1) & (corners[:, 0, 1] <= y2)
                ]
                if len(region_corners) > 0:
                    corners_found += 1
            
            # Cần ít nhất 3/4 góc
            passed = corners_found >= 3
            score = corners_found / 4.0
            
            return {
                "passed": passed,
                "score": float(score),
                "message": f"Found {corners_found}/4 corners" if passed else "Missing corners - document may be cut",
                "corners_found": corners_found
            }
            
        except Exception as e:
            return {
                "passed": True,  # Default to pass
                "score": 0.5,
                "message": f"Cannot check corners: {str(e)}"
            }
    
    def _check_occlusion(self, color_image: np.ndarray) -> Dict[str, Any]:
        """Kiểm tra có bị che khuất/dán đè không"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Detect foreign objects (unusual colors/textures)
            # Look for very bright or very dark regions
            bright_mask = cv2.inRange(gray, 240, 255)
            dark_mask = cv2.inRange(gray, 0, 15)
            
            bright_ratio = np.sum(bright_mask > 0) / (gray.shape[0] * gray.shape[1])
            dark_ratio = np.sum(dark_mask > 0) / (gray.shape[0] * gray.shape[1])
            
            # Detect unusual color concentrations
            saturation = hsv[:, :, 1]
            high_sat_mask = cv2.inRange(saturation, 200, 255)
            high_sat_ratio = np.sum(high_sat_mask > 0) / (gray.shape[0] * gray.shape[1])
            
            occlusion_indicators = 0
            
            if bright_ratio > 0.1:  # Too many bright spots
                occlusion_indicators += 1
            if dark_ratio > 0.1:   # Too many dark spots
                occlusion_indicators += 1
            if high_sat_ratio > 0.15:  # Unusual colors
                occlusion_indicators += 1
            
            occluded = occlusion_indicators >= 2
            
            return {
                "passed": not occluded,
                "score": 0.0 if occluded else 1.0,
                "message": "Possible occlusion detected" if occluded else "No occlusion detected",
                "bright_ratio": float(bright_ratio),
                "dark_ratio": float(dark_ratio),
                "high_saturation_ratio": float(high_sat_ratio)
            }
            
        except Exception as e:
            return {
                "passed": True,
                "score": 0.5,
                "message": f"Cannot check occlusion: {str(e)}"
            }
    
    def _check_readability(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Kiểm tra khả năng đọc text"""
        try:
            # Check contrast
            std_dev = np.std(gray_image)
            mean_val = np.mean(gray_image)
            
            # Edge detection for text
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Text-like patterns (horizontal and vertical lines)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            text_pattern_score = (np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)) / (2 * edges.shape[0] * edges.shape[1])
            
            score = 0.0
            messages = []
            
            # Good contrast
            if std_dev > 30:
                score += 0.4
                messages.append("Good contrast")
            else:
                messages.append("Low contrast")
            
            # Good edge density
            if edge_density > 0.05:
                score += 0.3
                messages.append("Clear edges")
            else:
                messages.append("Blurry details")
            
            # Text patterns
            if text_pattern_score > 0.01:
                score += 0.3
                messages.append("Text patterns detected")
            else:
                messages.append("No clear text patterns")
            
            passed = score >= 0.6
            message = "; ".join(messages)
            
            return {
                "passed": passed,
                "score": float(score),
                "message": message,
                "contrast": float(std_dev),
                "edge_density": float(edge_density)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "message": f"Error checking readability: {str(e)}"
            }
    
    def _check_aspect_ratio(self, image: np.ndarray) -> Dict[str, Any]:
        """Kiểm tra tỷ lệ khung hình của CCCD"""
        try:
            h, w = image.shape[:2]
            aspect_ratio = w / h
            
            # CCCD Việt Nam có tỷ lệ khoảng 1.586 (85.6mm x 54mm)
            target_ratio = 1.586
            tolerance = 0.2
            
            ratio_diff = abs(aspect_ratio - target_ratio)
            
            if ratio_diff <= tolerance:
                score = 1.0 - (ratio_diff / tolerance) * 0.3  # Max penalty 30%
                return {
                    "passed": True,
                    "score": float(score),
                    "message": "Good aspect ratio",
                    "aspect_ratio": float(aspect_ratio),
                    "target_ratio": target_ratio
                }
            else:
                return {
                    "passed": False,
                    "score": 0.0,
                    "message": "Incorrect aspect ratio - image may be cropped incorrectly",
                    "aspect_ratio": float(aspect_ratio),
                    "target_ratio": target_ratio
                }
                
        except Exception as e:
            return {
                "passed": True,
                "score": 0.5,
                "message": f"Cannot check aspect ratio: {str(e)}"
            }
    
    def _calculate_overall_quality(self, checks: Dict[str, Dict[str, Any]]) -> Tuple[float, str]:
        """Tính điểm tổng thể và chất lượng"""
        try:
            total_score = 0.0
            total_weight = 0.0
            critical_failed = False
            
            # Trọng số cho các checks
            weights = {
                "brightness": 0.15,
                "face_detection": 0.25,
                "face_obstruction": 0.15,
                "glasses_detection": 0.15,
                "hat_detection": 0.15,
                "clothing_detection": 0.15,
                "sharpness": 0.25,
                "corners": 0.15,
                "occlusion": 0.2,
                "readability": 0.2,
                "aspect_ratio": 0.2
            }
            
            # Critical checks (must pass)
            critical_checks = ["face_detection", "sharpness"]
            
            for check_name, check_result in checks.items():
                if check_name in weights and "score" in check_result:
                    weight = weights[check_name]
                    score = check_result["score"]
                    
                    total_score += score * weight
                    total_weight += weight
                    
                    # Check critical failures
                    if check_name in critical_checks and not check_result.get("passed", False):
                        critical_failed = True
            
            if total_weight > 0:
                overall_score = total_score / total_weight
            else:
                overall_score = 0.0
            
            # Determine quality level
            if critical_failed or overall_score < 0.4:
                quality = "poor"
            elif overall_score < 0.7:
                quality = "fair"
            else:
                quality = "good"
            
            return float(overall_score), quality
            
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return 0.0, "poor"
    
    def _generate_recommendations(self, checks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Tạo recommendations dựa trên kết quả checks"""
        recommendations = []
        
        try:
            for check_name, check_result in checks.items():
                if not check_result.get("passed", True):
                    message = check_result.get("message", "")
                    if "dark" in message.lower():
                        recommendations.append("Improve lighting - use natural light or bright indoor lighting")
                    elif "bright" in message.lower():
                        recommendations.append("Reduce brightness - avoid direct flash or sunlight")
                    elif "face" in message.lower() and "not" in message.lower():
                        recommendations.append("Ensure your face is clearly visible and centered")
                    elif "glass" in message.lower():
                        recommendations.append("Remove glasses for better face recognition")
                    elif "hat" in message.lower():
                        recommendations.append("Remove hat or head covering")
                    elif "clothing" in message.lower():
                        recommendations.append("Wear appropriate clothing (shirt/top)")
                    elif "blur" in message.lower():
                        recommendations.append("Take a sharper photo - ensure camera is steady")
                    elif "corner" in message.lower():
                        recommendations.append("Capture the full document without cutting edges")
                    elif "occlusion" in message.lower():
                        recommendations.append("Remove any stickers, tape, or objects covering the document")
                    elif "aspect" in message.lower():
                        recommendations.append("Frame the document properly to match standard proportions")
            
            # Remove duplicates
            recommendations = list(set(recommendations))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Please retake the photo with better quality"]


# Singleton instance
image_quality_service = ImageQualityAssessment()
