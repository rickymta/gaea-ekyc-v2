"""
OCR Service cho việc đọc thông tin từ Căn cước công dân Việt Nam
Sử dụng PaddleOCR và EasyOCR để đọc chính xác thông tin
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re
import logging
from datetime import datetime
from pathlib import Path

try:
    import paddleocr
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASY_OCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASY_OCR_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)

class VietnamIDCardOCR:
    """Service để OCR căn cước công dân Việt Nam"""
    
    def __init__(self):
        self.paddle_ocr = None
        self.easy_ocr = None
        
        # Khởi tạo các OCR engines
        self._initialize_ocr_engines()
        
        # Regex patterns cho thông tin CCCD
        self.patterns = {
            "id_number": r"\b\d{12}\b",  # 12 số
            "full_name": r"Họ và tên[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]+)",
            "date_of_birth": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
            "gender": r"Giới tính[:\s]*(Nam|Nữ)",
            "nationality": r"Quốc tịch[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]+)",
            "place_of_origin": r"Quê quán[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ,\s]+)",
            "address": r"Nơi thường trú[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ,.\s\d]+)",
            "issue_date": r"Ngày cấp[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})",
            "expiry_date": r"Có giá trị đến[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})",
            "issue_place": r"Nơi cấp[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ,\s]+)"
        }
        
    def _initialize_ocr_engines(self):
        """Khởi tạo các OCR engines"""
        try:
            if PADDLE_AVAILABLE:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True, 
                    lang='vi',  # Tiếng Việt
                    use_gpu=False,
                    show_log=False
                )
                logger.info("PaddleOCR initialized successfully")
            
            if EASY_OCR_AVAILABLE:
                self.easy_ocr = easyocr.Reader(['vi', 'en'], gpu=False)
                logger.info("EasyOCR initialized successfully")
                
            if not PADDLE_AVAILABLE and not EASY_OCR_AVAILABLE:
                logger.warning("No OCR engines available")
                
        except Exception as e:
            logger.error(f"Failed to initialize OCR engines: {e}")
    
    async def extract_id_card_info(
        self, 
        front_image: np.ndarray,
        back_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Trích xuất thông tin từ ảnh CCCD
        
        Args:
            front_image: Ảnh mặt trước CCCD
            back_image: Ảnh mặt sau CCCD (optional)
            
        Returns:
            Dict chứa thông tin đã trích xuất
        """
        try:
            result = {
                "success": False,
                "extracted_info": {},
                "raw_text": {
                    "front": "",
                    "back": ""
                },
                "confidence_scores": {},
                "errors": []
            }
            
            # Tiền xử lý ảnh
            processed_front = self._preprocess_image(front_image)
            
            # OCR mặt trước
            front_text, front_confidence = await self._perform_ocr(processed_front)
            result["raw_text"]["front"] = front_text
            
            if not front_text.strip():
                result["errors"].append("Cannot extract text from front side")
                return result
            
            # Trích xuất thông tin từ mặt trước
            front_info = self._extract_front_info(front_text)
            result["extracted_info"].update(front_info)
            result["confidence_scores"]["front"] = front_confidence
            
            # OCR mặt sau nếu có
            if back_image is not None:
                processed_back = self._preprocess_image(back_image)
                back_text, back_confidence = await self._perform_ocr(processed_back)
                result["raw_text"]["back"] = back_text
                result["confidence_scores"]["back"] = back_confidence
                
                # Trích xuất thông tin từ mặt sau
                back_info = self._extract_back_info(back_text)
                result["extracted_info"].update(back_info)
            
            # Validation thông tin
            validation_result = self._validate_extracted_info(result["extracted_info"])
            result["validation"] = validation_result
            
            # Đánh giá độ tin cậy tổng thể
            overall_confidence = self._calculate_overall_confidence(
                result["extracted_info"], 
                result["confidence_scores"]
            )
            result["overall_confidence"] = overall_confidence
            
            # Xác định success
            required_fields = ["id_number", "full_name", "date_of_birth"]
            has_required = all(
                field in result["extracted_info"] and result["extracted_info"][field] 
                for field in required_fields
            )
            
            result["success"] = has_required and overall_confidence >= 0.7
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ID card OCR: {e}")
            return {
                "success": False,
                "extracted_info": {},
                "raw_text": {"front": "", "back": ""},
                "confidence_scores": {},
                "errors": [str(e)],
                "overall_confidence": 0.0
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh để cải thiện OCR"""
        try:
            # Chuyển sang grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Tăng kích thước nếu ảnh quá nhỏ
            height, width = gray.shape
            if height < 600 or width < 800:
                scale_factor = max(800/width, 600/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Khử nhiễu
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Tăng độ tương phản
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Làm sắc nét
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image
    
    async def _perform_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Thực hiện OCR với các engines có sẵn"""
        texts = []
        confidences = []
        
        # Thử PaddleOCR trước
        if self.paddle_ocr is not None:
            try:
                paddle_result = self.paddle_ocr.ocr(image, cls=True)
                
                if paddle_result and paddle_result[0]:
                    paddle_text = ""
                    paddle_confidences = []
                    
                    for line in paddle_result[0]:
                        if line:
                            text = line[1][0]
                            confidence = line[1][1]
                            paddle_text += text + " "
                            paddle_confidences.append(confidence)
                    
                    if paddle_text.strip():
                        texts.append(paddle_text.strip())
                        confidences.append(np.mean(paddle_confidences) if paddle_confidences else 0.0)
                        
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}")
        
        # Thử EasyOCR
        if self.easy_ocr is not None:
            try:
                easy_result = self.easy_ocr.readtext(image)
                
                if easy_result:
                    easy_text = ""
                    easy_confidences = []
                    
                    for detection in easy_result:
                        text = detection[1]
                        confidence = detection[2]
                        easy_text += text + " "
                        easy_confidences.append(confidence)
                    
                    if easy_text.strip():
                        texts.append(easy_text.strip())
                        confidences.append(np.mean(easy_confidences) if easy_confidences else 0.0)
                        
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Chọn kết quả tốt nhất
        if texts:
            best_idx = np.argmax(confidences)
            return texts[best_idx], confidences[best_idx]
        else:
            return "", 0.0
    
    def _extract_front_info(self, text: str) -> Dict[str, str]:
        """Trích xuất thông tin từ mặt trước CCCD"""
        info = {}
        
        try:
            # Làm sạch text
            cleaned_text = self._clean_text(text)
            
            # Trích xuất số CCCD
            id_match = re.search(self.patterns["id_number"], cleaned_text)
            if id_match:
                info["id_number"] = id_match.group()
            
            # Trích xuất họ tên
            name_match = re.search(self.patterns["full_name"], cleaned_text, re.IGNORECASE)
            if name_match:
                info["full_name"] = name_match.group(1).strip()
            else:
                # Fallback: tìm tên từ dòng chứa "Họ và tên"
                lines = cleaned_text.split('\n')
                for i, line in enumerate(lines):
                    if 'họ và tên' in line.lower() or 'ho va ten' in line.lower():
                        if i + 1 < len(lines):
                            info["full_name"] = lines[i + 1].strip()
                        break
            
            # Trích xuất ngày sinh
            dob_match = re.search(self.patterns["date_of_birth"], cleaned_text)
            if dob_match:
                info["date_of_birth"] = dob_match.group()
            
            # Trích xuất giới tính
            gender_match = re.search(self.patterns["gender"], cleaned_text, re.IGNORECASE)
            if gender_match:
                info["gender"] = gender_match.group(1)
            
            # Trích xuất quốc tịch
            nationality_match = re.search(self.patterns["nationality"], cleaned_text, re.IGNORECASE)
            if nationality_match:
                info["nationality"] = nationality_match.group(1).strip()
            
            # Trích xuất quê quán
            origin_match = re.search(self.patterns["place_of_origin"], cleaned_text, re.IGNORECASE)
            if origin_match:
                info["place_of_origin"] = origin_match.group(1).strip()
            
            # Trích xuất nơi thường trú
            address_match = re.search(self.patterns["address"], cleaned_text, re.IGNORECASE)
            if address_match:
                info["address"] = address_match.group(1).strip()
                
        except Exception as e:
            logger.error(f"Error extracting front info: {e}")
        
        return info
    
    def _extract_back_info(self, text: str) -> Dict[str, str]:
        """Trích xuất thông tin từ mặt sau CCCD"""
        info = {}
        
        try:
            cleaned_text = self._clean_text(text)
            
            # Trích xuất ngày cấp
            issue_date_match = re.search(self.patterns["issue_date"], cleaned_text, re.IGNORECASE)
            if issue_date_match:
                info["issue_date"] = issue_date_match.group(1)
            
            # Trích xuất ngày hết hạn
            expiry_date_match = re.search(self.patterns["expiry_date"], cleaned_text, re.IGNORECASE)
            if expiry_date_match:
                info["expiry_date"] = expiry_date_match.group(1)
            
            # Trích xuất nơi cấp
            issue_place_match = re.search(self.patterns["issue_place"], cleaned_text, re.IGNORECASE)
            if issue_place_match:
                info["issue_place"] = issue_place_match.group(1).strip()
                
        except Exception as e:
            logger.error(f"Error extracting back info: {e}")
        
        return info
    
    def _clean_text(self, text: str) -> str:
        """Làm sạch text OCR"""
        # Loại bỏ các ký tự đặc biệt
        cleaned = re.sub(r'[^\w\s\-/:.,()\n]', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Chuẩn hóa dấu gạch ngang và dấu /
        cleaned = re.sub(r'[-–—]', '-', cleaned)
        cleaned = re.sub(r'[/\\]', '/', cleaned)
        
        return cleaned.strip()
    
    def _validate_extracted_info(self, info: Dict[str, str]) -> Dict[str, Any]:
        """Validate thông tin đã trích xuất"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate số CCCD (12 số)
        if "id_number" in info:
            if not re.match(r'^\d{12}$', info["id_number"]):
                validation["errors"].append("Invalid ID number format")
                validation["is_valid"] = False
        else:
            validation["errors"].append("ID number not found")
            validation["is_valid"] = False
        
        # Validate họ tên
        if "full_name" in info:
            if len(info["full_name"]) < 2:
                validation["errors"].append("Full name too short")
                validation["is_valid"] = False
        else:
            validation["errors"].append("Full name not found")
            validation["is_valid"] = False
        
        # Validate ngày sinh
        if "date_of_birth" in info:
            try:
                # Thử parse ngày sinh
                dob_str = info["date_of_birth"].replace('-', '/').replace('.', '/')
                dob = datetime.strptime(dob_str, '%d/%m/%Y')
                
                # Kiểm tra hợp lý (tuổi từ 15-120)
                today = datetime.now()
                age = today.year - dob.year
                if age < 15 or age > 120:
                    validation["warnings"].append("Unusual age")
                    
            except ValueError:
                validation["errors"].append("Invalid date of birth format")
                validation["is_valid"] = False
        
        return validation
    
    def _calculate_overall_confidence(
        self, 
        extracted_info: Dict[str, str], 
        confidence_scores: Dict[str, float]
    ) -> float:
        """Tính độ tin cậy tổng thể"""
        try:
            # Trọng số cho các trường quan trọng
            field_weights = {
                "id_number": 0.3,
                "full_name": 0.25,
                "date_of_birth": 0.2,
                "address": 0.15,
                "gender": 0.1
            }
            
            total_weight = 0
            weighted_score = 0
            
            # Tính điểm dựa trên việc có field hay không
            for field, weight in field_weights.items():
                if field in extracted_info and extracted_info[field]:
                    total_weight += weight
                    weighted_score += weight
            
            # Nhân với confidence trung bình từ OCR
            ocr_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
            
            if total_weight > 0:
                field_score = weighted_score / total_weight
                overall_confidence = (field_score * 0.7) + (ocr_confidence * 0.3)
            else:
                overall_confidence = ocr_confidence * 0.3
            
            return float(overall_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0


# Singleton instance
vietnam_id_ocr_service = VietnamIDCardOCR()
