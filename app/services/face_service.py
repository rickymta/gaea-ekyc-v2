"""
Enhanced Face Service with Liveness Detection, OCR, and Quality Assessment
Tích hợp InsightFace, Silent Face Anti-Spoofing, OCR và đánh giá chất lượng ảnh
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime

from app.config import settings

# Logging setup
logger = logging.getLogger(__name__)


class EnhancedFaceEngine:
    """
    Enhanced Face Engine với đầy đủ tính năng EKYC:
    - Face detection & recognition (InsightFace)
    - Liveness detection (Silent Face Anti-Spoofing)
    - OCR cho CCCD (PaddleOCR + EasyOCR)
    - Image quality assessment
    """
    
    def __init__(self, model_name: str = None, model_path: str = None):
        self.model_name = model_name or settings.insightface_model
        self.model_path = model_path or settings.insightface_model_path
        self.face_analysis = None
        self.recognition_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize InsightFace models"""
        try:
            # Initialize face analysis (detection + recognition)
            self.face_analysis = FaceAnalysis(
                name=self.model_name,
                root=self.model_path,
                providers=['CPUExecutionProvider']  # Can be changed to CUDAExecutionProvider for GPU
            )
            self.face_analysis.prepare(
                ctx_id=0, 
                det_size=settings.face_detection_size
            )
            
            # Initialize dedicated recognition model for better performance
            model_file = os.path.join(self.model_path, self.model_name)
            if os.path.exists(model_file):
                self.recognition_model = get_model(model_file)
                if self.recognition_model:
                    self.recognition_model.prepare(ctx_id=0)
            
            logger.info(f"Enhanced Face Engine initialized successfully with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace models: {str(e)}")
            raise

    async def _perform_face_matching(
        self,
        selfie_face: Dict[str, Any],
        id_face: Dict[str, Any],
        selfie_image: np.ndarray,
        id_image: np.ndarray
    ) -> Dict[str, Any]:
        """Thực hiện face matching giữa selfie và CCCD"""
        try:
            # Extract embeddings if not already available
            selfie_embedding = None
            id_embedding = None
            
            if selfie_face.get("embedding"):
                selfie_embedding = np.array(selfie_face["embedding"])
            else:
                selfie_embedding = self.extract_face_embedding(selfie_image, selfie_face["bbox"])
            
            if id_face.get("embedding"):
                id_embedding = np.array(id_face["embedding"])
            else:
                id_embedding = self.extract_face_embedding(id_image, id_face["bbox"])
            
            if selfie_embedding is None or id_embedding is None:
                return {
                    "is_match": False,
                    "similarity_score": 0.0,
                    "confidence_level": "LOW",
                    "error": "Could not extract face embeddings for comparison"
                }
            
            # Calculate similarity
            similarity_score = self.compare_faces(selfie_embedding, id_embedding)
            
            # Determine if match based on threshold
            threshold = settings.face_match_threshold
            is_match = similarity_score >= threshold
            
            # Determine confidence level
            if similarity_score >= 0.8:
                confidence_level = "HIGH"
            elif similarity_score >= 0.6:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
            
            return {
                "is_match": is_match,
                "similarity_score": float(similarity_score),
                "threshold_used": float(threshold),
                "confidence_level": confidence_level,
                "selfie_face_quality": selfie_face.get("confidence", 0.0),
                "id_face_quality": id_face.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error in face matching: {e}")
            return {
                "is_match": False,
                "similarity_score": 0.0,
                "confidence_level": "LOW",
                "error": str(e)
            }

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        try:
            faces = self.face_analysis.get(image)
            
            results = []
            for face in faces:
                result = {
                    'bbox': face.bbox.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(face.det_score),
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                    'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None,
                    'age': float(face.age) if hasattr(face, 'age') else None,
                    'gender': int(face.gender) if hasattr(face, 'gender') else None
                }
                results.append(result)
            
            logger.info(f"Detected {len(results)} faces in image")
            return results
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def extract_face_embedding(self, image: np.ndarray, face_bbox: List[float] = None) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Input image as numpy array
            face_bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Face embedding as numpy array or None if no face found
        """
        try:
            faces = self.face_analysis.get(image)
            
            if not faces:
                logger.warning("No faces detected for embedding extraction")
                return None
            
            # If bbox is provided, find the closest face
            if face_bbox:
                target_face = self._find_closest_face(faces, face_bbox)
            else:
                # Use the largest face (most confident detection)
                target_face = max(faces, key=lambda x: x.det_score)
            
            if hasattr(target_face, 'embedding'):
                logger.info("Face embedding extracted successfully")
                return target_face.embedding
            else:
                logger.warning("Face detected but embedding not available")
                return None
                
        except Exception as e:
            logger.error(f"Face embedding extraction failed: {str(e)}")
            return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings and return similarity score
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            logger.info(f"Face comparison completed with similarity: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {str(e)}")
            return 0.0

    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                      threshold: float = None) -> bool:
        """
        Determine if two face embeddings belong to the same person
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (default from settings)
            
        Returns:
            True if same person, False otherwise
        """
        if threshold is None:
            threshold = settings.face_match_threshold
            
        similarity = self.compare_faces(embedding1, embedding2)
        return similarity >= threshold

    def _find_closest_face(self, faces: List, target_bbox: List[float]):
        """Find the face closest to the target bounding box"""
        if not faces:
            return None
            
        def bbox_overlap(bbox1, bbox2):
            """Calculate overlap between two bounding boxes"""
            x1_max = max(bbox1[0], bbox2[0])
            y1_max = max(bbox1[1], bbox2[1])
            x2_min = min(bbox1[2], bbox2[2])
            y2_min = min(bbox1[3], bbox2[3])
            
            if x2_min <= x1_max or y2_min <= y1_max:
                return 0.0
                
            intersection = (x2_min - x1_max) * (y2_min - y1_max)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Find face with highest overlap
        best_face = None
        best_overlap = 0.0
        
        for face in faces:
            overlap = bbox_overlap(face.bbox.tolist(), target_bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_face = face
        
        return best_face if best_face else faces[0]


# Singleton instance
enhanced_face_engine = EnhancedFaceEngine()
