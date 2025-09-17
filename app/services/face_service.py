import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
from app.config import settings

# Logging setup
logger = logging.getLogger(__name__)


class FaceEngine:
    """InsightFace wrapper for face detection, recognition and liveness detection"""
    
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
            
            logger.info(f"InsightFace models initialized successfully with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace models: {str(e)}")
            raise
    
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
                      threshold: float = None) -> Tuple[bool, float]:
        """
        Determine if two embeddings belong to the same person
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (uses config default if None)
            
        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        threshold = threshold or settings.face_match_threshold
        similarity = self.compare_faces(embedding1, embedding2)
        is_same = similarity >= threshold
        
        logger.info(f"Face match result: {is_same} (similarity: {similarity:.4f}, threshold: {threshold})")
        return is_same, similarity
    
    def detect_liveness(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Basic liveness detection based on face quality metrics
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with liveness detection results
        """
        try:
            faces = self.detect_faces(image)
            
            if not faces:
                return {
                    'is_live': False,
                    'score': 0.0,
                    'reason': 'No face detected',
                    'checks': {}
                }
            
            # Use the most confident face
            best_face = max(faces, key=lambda x: x['confidence'])
            
            # Basic liveness checks
            checks = {
                'face_confidence': best_face['confidence'],
                'face_size_adequate': self._check_face_size(best_face['bbox'], image.shape),
                'face_quality': self._assess_face_quality(image, best_face),
                'multiple_faces': len(faces) == 1  # Single face is better for liveness
            }
            
            # Calculate overall liveness score
            score = self._calculate_liveness_score(checks)
            is_live = score >= settings.liveness_threshold
            
            result = {
                'is_live': is_live,
                'score': score,
                'checks': checks,
                'reason': 'Passed basic liveness checks' if is_live else 'Failed liveness checks'
            }
            
            logger.info(f"Liveness detection completed: {is_live} (score: {score:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Liveness detection failed: {str(e)}")
            return {
                'is_live': False,
                'score': 0.0,
                'reason': f'Error during liveness detection: {str(e)}',
                'checks': {}
            }
    
    def _find_closest_face(self, faces: List, target_bbox: List[float]):
        """Find the face closest to the target bounding box"""
        min_distance = float('inf')
        closest_face = faces[0]
        
        target_center = [(target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2]
        
        for face in faces:
            face_center = [(face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2]
            distance = np.sqrt((target_center[0] - face_center[0])**2 + (target_center[1] - face_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_face = face
        
        return closest_face
    
    def _check_face_size(self, bbox: List[float], image_shape: Tuple) -> bool:
        """Check if face size is adequate"""
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        image_width, image_height = image_shape[1], image_shape[0]
        
        # Face should be at least 10% of image width/height
        min_size_ratio = 0.1
        width_ratio = face_width / image_width
        height_ratio = face_height / image_height
        
        return width_ratio >= min_size_ratio and height_ratio >= min_size_ratio
    
    def _assess_face_quality(self, image: np.ndarray, face_info: Dict) -> float:
        """Assess face quality based on various metrics"""
        try:
            bbox = face_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return 0.0
            
            # Calculate sharpness (Laplacian variance)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Normalize sharpness score (higher is better)
            sharpness_score = min(sharpness / 1000.0, 1.0)
            
            # Calculate brightness
            brightness = np.mean(gray_face) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Penalize too dark or too bright
            
            # Combine scores
            quality_score = (sharpness_score * 0.6 + brightness_score * 0.4)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Face quality assessment failed: {str(e)}")
            return 0.0
    
    def _calculate_liveness_score(self, checks: Dict) -> float:
        """Calculate overall liveness score from individual checks"""
        weights = {
            'face_confidence': 0.3,
            'face_size_adequate': 0.2,
            'face_quality': 0.3,
            'multiple_faces': 0.2
        }
        
        score = 0.0
        for check_name, weight in weights.items():
            if check_name in checks:
                if isinstance(checks[check_name], bool):
                    score += weight * (1.0 if checks[check_name] else 0.0)
                else:
                    score += weight * float(checks[check_name])
        
        return min(score, 1.0)
    
    def train_face_database(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train face recognition system with new identity data
        
        Args:
            training_data: List of training samples, each containing:
                - person_id: Unique identifier for the person
                - images: List of image paths or numpy arrays
                - metadata: Additional information about the person
                
        Returns:
            Training results and statistics
        """
        try:
            training_results = {
                'total_persons': 0,
                'total_images': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'person_embeddings': {},
                'errors': []
            }
            
            for person_data in training_data:
                person_id = person_data['person_id']
                images = person_data.get('images', [])
                metadata = person_data.get('metadata', {})
                
                person_embeddings = []
                person_errors = []
                
                for img_data in images:
                    try:
                        # Load image
                        if isinstance(img_data, str):
                            image = cv2.imread(img_data)
                        elif isinstance(img_data, np.ndarray):
                            image = img_data
                        else:
                            continue
                            
                        if image is None:
                            continue
                        
                        # Extract embedding
                        embedding = self.extract_face_embedding(image)
                        if embedding is not None:
                            person_embeddings.append(embedding)
                            training_results['successful_extractions'] += 1
                        else:
                            training_results['failed_extractions'] += 1
                            person_errors.append(f"No face detected in image")
                        
                        training_results['total_images'] += 1
                        
                    except Exception as e:
                        training_results['failed_extractions'] += 1
                        person_errors.append(f"Error processing image: {str(e)}")
                
                if person_embeddings:
                    # Calculate average embedding for the person
                    avg_embedding = np.mean(person_embeddings, axis=0)
                    training_results['person_embeddings'][person_id] = {
                        'embedding': avg_embedding,
                        'num_samples': len(person_embeddings),
                        'metadata': metadata,
                        'quality_score': self._calculate_embedding_quality(person_embeddings)
                    }
                    training_results['total_persons'] += 1
                
                if person_errors:
                    training_results['errors'].append({
                        'person_id': person_id,
                        'errors': person_errors
                    })
            
            logger.info(f"Training completed: {training_results['total_persons']} persons, "
                       f"{training_results['successful_extractions']} successful extractions")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def perform_ekyc_verification(self, id_card_image: np.ndarray, 
                                 selfie_image: np.ndarray,
                                 user_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform complete EKYC verification process
        
        Args:
            id_card_image: ID card/document image
            selfie_image: User selfie image
            user_data: Additional user information
            
        Returns:
            Complete EKYC verification results
        """
        try:
            verification_result = {
                'verification_id': self._generate_verification_id(),
                'timestamp': self._get_timestamp(),
                'user_data': user_data or {},
                'id_card_analysis': {},
                'selfie_analysis': {},
                'face_match': {},
                'liveness_check': {},
                'overall_result': {},
                'confidence_scores': {},
                'recommendations': []
            }
            
            # 1. Analyze ID card image
            logger.info("Starting ID card analysis...")
            id_faces = self.detect_faces(id_card_image)
            if not id_faces:
                verification_result['overall_result'] = {
                    'status': 'FAILED',
                    'reason': 'No face detected in ID card'
                }
                return verification_result
            
            id_face = max(id_faces, key=lambda x: x['confidence'])
            id_embedding = self.extract_face_embedding(id_card_image, id_face['bbox'])
            
            verification_result['id_card_analysis'] = {
                'faces_detected': len(id_faces),
                'primary_face': id_face,
                'face_quality': self._assess_face_quality(id_card_image, id_face),
                'embedding_extracted': id_embedding is not None
            }
            
            # 2. Analyze selfie image
            logger.info("Starting selfie analysis...")
            selfie_faces = self.detect_faces(selfie_image)
            if not selfie_faces:
                verification_result['overall_result'] = {
                    'status': 'FAILED',
                    'reason': 'No face detected in selfie'
                }
                return verification_result
            
            selfie_face = max(selfie_faces, key=lambda x: x['confidence'])
            selfie_embedding = self.extract_face_embedding(selfie_image, selfie_face['bbox'])
            
            verification_result['selfie_analysis'] = {
                'faces_detected': len(selfie_faces),
                'primary_face': selfie_face,
                'face_quality': self._assess_face_quality(selfie_image, selfie_face),
                'embedding_extracted': selfie_embedding is not None
            }
            
            # 3. Perform liveness detection
            logger.info("Performing liveness detection...")
            liveness_result = self.detect_liveness(selfie_image)
            verification_result['liveness_check'] = liveness_result
            
            # 4. Face matching
            logger.info("Performing face matching...")
            if id_embedding is not None and selfie_embedding is not None:
                is_match, similarity = self.is_same_person(id_embedding, selfie_embedding)
                verification_result['face_match'] = {
                    'is_match': is_match,
                    'similarity_score': float(similarity),
                    'threshold_used': settings.face_match_threshold,
                    'confidence_level': self._get_confidence_level(similarity)
                }
            else:
                verification_result['face_match'] = {
                    'is_match': False,
                    'error': 'Could not extract embeddings from one or both images'
                }
            
            # 5. Calculate overall confidence scores
            verification_result['confidence_scores'] = {
                'id_card_quality': verification_result['id_card_analysis'].get('face_quality', {}).get('overall_score', 0),
                'selfie_quality': verification_result['selfie_analysis'].get('face_quality', {}).get('overall_score', 0),
                'liveness_score': liveness_result.get('score', 0),
                'face_match_score': verification_result['face_match'].get('similarity_score', 0)
            }
            
            # 6. Determine overall result
            overall_confidence = self._calculate_overall_confidence(verification_result['confidence_scores'])
            verification_result['overall_result'] = {
                'status': 'PASSED' if self._meets_verification_criteria(verification_result) else 'FAILED',
                'confidence': overall_confidence,
                'decision_factors': self._get_decision_factors(verification_result)
            }
            
            # 7. Generate recommendations
            verification_result['recommendations'] = self._generate_recommendations(verification_result)
            
            logger.info(f"EKYC verification completed: {verification_result['overall_result']['status']}")
            return verification_result
            
        except Exception as e:
            logger.error(f"EKYC verification failed: {str(e)}")
            return {
                'verification_id': self._generate_verification_id(),
                'timestamp': self._get_timestamp(),
                'overall_result': {
                    'status': 'ERROR',
                    'error': str(e)
                }
            }
    
    def batch_face_recognition(self, query_image: np.ndarray, 
                             database_embeddings: Dict[str, np.ndarray],
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform batch face recognition against a database of known faces
        
        Args:
            query_image: Image to search for
            database_embeddings: Dictionary of {person_id: embedding}
            top_k: Number of top matches to return
            
        Returns:
            List of top matches with scores
        """
        try:
            query_embedding = self.extract_face_embedding(query_image)
            if query_embedding is None:
                return []
            
            matches = []
            for person_id, db_embedding in database_embeddings.items():
                similarity = self.compare_faces(query_embedding, db_embedding)
                matches.append({
                    'person_id': person_id,
                    'similarity': float(similarity),
                    'is_match': similarity >= settings.face_match_threshold
                })
            
            # Sort by similarity and return top-k
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Batch recognition failed: {str(e)}")
            return []
    
    def update_model_with_feedback(self, verification_results: List[Dict[str, Any]],
                                  feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update model performance based on user feedback and verification results
        
        Args:
            verification_results: List of verification results
            feedback_data: List of user feedback on verification accuracy
            
        Returns:
            Model update results and performance metrics
        """
        try:
            update_result = {
                'total_feedback': len(feedback_data),
                'correct_predictions': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'accuracy_improvement': 0,
                'threshold_adjustments': {},
                'recommendations': []
            }
            
            # Analyze feedback patterns
            for feedback in feedback_data:
                verification_id = feedback.get('verification_id')
                actual_result = feedback.get('actual_result')  # True/False
                system_result = feedback.get('system_result')  # True/False
                confidence = feedback.get('confidence', 0)
                
                if actual_result == system_result:
                    update_result['correct_predictions'] += 1
                elif system_result and not actual_result:
                    update_result['false_positives'] += 1
                elif not system_result and actual_result:
                    update_result['false_negatives'] += 1
            
            # Calculate accuracy
            total_feedback = len(feedback_data)
            if total_feedback > 0:
                accuracy = update_result['correct_predictions'] / total_feedback
                update_result['current_accuracy'] = accuracy
                
                # Suggest threshold adjustments
                if update_result['false_positives'] > update_result['false_negatives']:
                    # Too many false positives - increase threshold
                    suggested_threshold = min(settings.face_match_threshold + 0.05, 0.9)
                    update_result['threshold_adjustments']['face_match'] = suggested_threshold
                    update_result['recommendations'].append("Consider increasing face match threshold to reduce false positives")
                
                elif update_result['false_negatives'] > update_result['false_positives']:
                    # Too many false negatives - decrease threshold
                    suggested_threshold = max(settings.face_match_threshold - 0.05, 0.3)
                    update_result['threshold_adjustments']['face_match'] = suggested_threshold
                    update_result['recommendations'].append("Consider decreasing face match threshold to reduce false negatives")
            
            logger.info(f"Model feedback analysis completed: {update_result['current_accuracy']:.2%} accuracy")
            return update_result
            
        except Exception as e:
            logger.error(f"Model update failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # Helper methods for EKYC verification
    def _calculate_embedding_quality(self, embeddings: List[np.ndarray]) -> float:
        """Calculate quality score for a set of embeddings"""
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate consistency (lower variance = higher quality)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self.compare_faces(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _generate_verification_id(self) -> str:
        """Generate unique verification ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_confidence_level(self, similarity: float) -> str:
        """Get confidence level based on similarity score"""
        if similarity >= 0.8:
            return "HIGH"
        elif similarity >= 0.6:
            return "MEDIUM" 
        else:
            return "LOW"
    
    def _calculate_overall_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate overall confidence from individual scores"""
        weights = {
            'id_card_quality': 0.2,
            'selfie_quality': 0.2,
            'liveness_score': 0.3,
            'face_match_score': 0.3
        }
        
        weighted_sum = sum(scores.get(key, 0) * weight for key, weight in weights.items())
        return min(weighted_sum, 1.0)
    
    def _meets_verification_criteria(self, result: Dict[str, Any]) -> bool:
        """Check if verification result meets all criteria"""
        face_match = result.get('face_match', {})
        liveness = result.get('liveness_check', {})
        
        return (
            face_match.get('is_match', False) and
            liveness.get('is_live', False) and
            result.get('confidence_scores', {}).get('face_match_score', 0) >= settings.face_match_threshold
        )
    
    def _get_decision_factors(self, result: Dict[str, Any]) -> List[str]:
        """Get factors that influenced the verification decision"""
        factors = []
        
        if not result.get('face_match', {}).get('is_match', False):
            factors.append("Face does not match ID document")
        
        if not result.get('liveness_check', {}).get('is_live', False):
            factors.append("Liveness detection failed")
        
        if result.get('confidence_scores', {}).get('face_match_score', 0) < settings.face_match_threshold:
            factors.append("Face similarity below threshold")
        
        return factors
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on verification result"""
        recommendations = []
        
        id_quality = result.get('confidence_scores', {}).get('id_card_quality', 0)
        selfie_quality = result.get('confidence_scores', {}).get('selfie_quality', 0)
        
        if id_quality < 0.5:
            recommendations.append("Improve ID document image quality - ensure good lighting and clear focus")
        
        if selfie_quality < 0.5:
            recommendations.append("Improve selfie image quality - face camera directly with good lighting")
        
        if not result.get('liveness_check', {}).get('is_live', False):
            recommendations.append("Take a live selfie - avoid using photos of photos")
        
        if result.get('face_match', {}).get('similarity_score', 0) < 0.5:
            recommendations.append("Ensure the person in selfie matches the ID document")
        
        return recommendations


def load_image_from_path(image_path: str) -> Optional[np.ndarray]:
    """Load image from file path"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image from {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Load image from bytes data"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to decode image from bytes")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image from bytes: {str(e)}")
        return None


# Global face engine instance
face_engine = FaceEngine()
