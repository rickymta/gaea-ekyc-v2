"""
Training API Router
Handles model training, evaluation, and optimization endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

from app.services.training_service import training_service
from app.services.face_service import face_engine, load_image_from_bytes
from app.core.dependencies import get_current_user
from app.schemas.training import (
    TrainingDataSource, TrainingConfig, TrainingResult,
    EvaluationResult, OptimizationConfig, OptimizationResult
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])


@router.post("/prepare-dataset", response_model=Dict[str, Any])
async def prepare_training_dataset(
    data_sources: List[TrainingDataSource],
    current_user = Depends(get_current_user)
):
    """
    Prepare training dataset from various sources
    """
    try:
        logger.info(f"User {current_user.username} preparing training dataset with {len(data_sources)} sources")
        
        sources_config = [source.dict() for source in data_sources]
        result = training_service.prepare_training_dataset(sources_config)
        
        if not result.get('success', False):
            raise HTTPException(status_code=400, detail=result.get('error', 'Dataset preparation failed'))
        
        return {
            "message": "Dataset prepared successfully",
            "dataset_file": result['dataset_file'],
            "statistics": result['statistics']
        }
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-model", response_model=TrainingResult)
async def train_recognition_model(
    dataset_file: str = Form(...),
    training_config: str = Form(...),
    current_user = Depends(get_current_user)
):
    """
    Train face recognition model with prepared dataset
    """
    try:
        logger.info(f"User {current_user.username} starting model training")
        
        config = json.loads(training_config)
        result = training_service.train_recognition_model(dataset_file, config)
        
        if not result.get('success', True):
            raise HTTPException(status_code=400, detail=result.get('error', 'Training failed'))
        
        return TrainingResult(**result)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid training configuration JSON")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-model", response_model=EvaluationResult)
async def evaluate_model_performance(
    test_dataset: str = Form(...),
    model_config: str = Form(...),
    current_user = Depends(get_current_user)
):
    """
    Evaluate model performance on test dataset
    """
    try:
        logger.info(f"User {current_user.username} evaluating model performance")
        
        config = json.loads(model_config)
        result = training_service.evaluate_model_performance(test_dataset, config)
        
        if not result.get('success', True):
            raise HTTPException(status_code=400, detail=result.get('error', 'Evaluation failed'))
        
        return EvaluationResult(**result)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid model configuration JSON")
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continuous-learning", response_model=Dict[str, Any])
async def continuous_learning_update(
    new_training_data: str = Form(None),
    feedback_data: str = Form(None),
    current_user = Depends(get_current_user)
):
    """
    Update model with new data and user feedback
    """
    try:
        logger.info(f"User {current_user.username} performing continuous learning update")
        
        new_data = json.loads(new_training_data) if new_training_data else []
        feedback = json.loads(feedback_data) if feedback_data else []
        
        result = training_service.continuous_learning_update(new_data, feedback)
        
        if not result.get('success', True):
            raise HTTPException(status_code=400, detail=result.get('error', 'Update failed'))
        
        return {
            "message": "Continuous learning update completed",
            "update_id": result.get('update_id'),
            "improvements": result.get('improvements', {}),
            "new_metrics": result.get('new_metrics', {})
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Continuous learning update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-production", response_model=OptimizationResult)
async def optimize_for_production(
    optimization_config: OptimizationConfig,
    current_user = Depends(get_current_user)
):
    """
    Optimize model for production deployment
    """
    try:
        logger.info(f"User {current_user.username} optimizing model for production")
        
        result = training_service.optimize_for_production(optimization_config.dict())
        
        if not result.get('success', True):
            raise HTTPException(status_code=400, detail=result.get('error', 'Optimization failed'))
        
        return OptimizationResult(**result)
        
    except Exception as e:
        logger.error(f"Production optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-embeddings", response_model=Dict[str, Any])
async def extract_face_embeddings(
    images: List[UploadFile] = File(...),
    person_id: str = Form(...),
    current_user = Depends(get_current_user)
):
    """
    Extract face embeddings from uploaded images
    """
    try:
        logger.info(f"User {current_user.username} extracting embeddings for person {person_id}")
        
        results = {
            'person_id': person_id,
            'images_processed': 0,
            'successful_extractions': 0,
            'embeddings': [],
            'errors': []
        }
        
        for i, image_file in enumerate(images):
            try:
                # Read image
                image_bytes = await image_file.read()
                image = load_image_from_bytes(image_bytes)
                
                if image is None:
                    results['errors'].append(f"Could not load image {image_file.filename}")
                    continue
                
                # Extract embedding
                embedding = face_engine.extract_face_embedding(image)
                
                if embedding is not None:
                    results['embeddings'].append({
                        'filename': image_file.filename,
                        'embedding': embedding.tolist(),
                        'embedding_size': len(embedding)
                    })
                    results['successful_extractions'] += 1
                else:
                    results['errors'].append(f"No face detected in {image_file.filename}")
                
                results['images_processed'] += 1
                
            except Exception as e:
                results['errors'].append(f"Error processing {image_file.filename}: {str(e)}")
        
        return {
            "message": f"Processed {results['images_processed']} images",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-recognition", response_model=Dict[str, Any])
async def batch_face_recognition(
    query_image: UploadFile = File(...),
    database_embeddings: str = Form(...),
    top_k: int = Form(5),
    current_user = Depends(get_current_user)
):
    """
    Perform batch face recognition against database
    """
    try:
        logger.info(f"User {current_user.username} performing batch face recognition")
        
        # Load query image
        image_bytes = await query_image.read()
        image = load_image_from_bytes(image_bytes)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not load query image")
        
        # Parse database embeddings
        db_embeddings = json.loads(database_embeddings)
        
        # Convert embeddings back to numpy arrays
        import numpy as np
        for person_id, embedding_data in db_embeddings.items():
            if isinstance(embedding_data, list):
                db_embeddings[person_id] = np.array(embedding_data)
            elif isinstance(embedding_data, dict) and 'embedding' in embedding_data:
                db_embeddings[person_id] = np.array(embedding_data['embedding'])
        
        # Perform recognition
        matches = face_engine.batch_face_recognition(image, db_embeddings, top_k)
        
        return {
            "message": f"Found {len(matches)} matches",
            "query_image": query_image.filename,
            "matches": matches,
            "total_database_size": len(db_embeddings)
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid database embeddings JSON")
    except Exception as e:
        logger.error(f"Batch recognition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-history", response_model=Dict[str, Any])
async def get_training_history(
    limit: int = 10,
    current_user = Depends(get_current_user)
):
    """
    Get training history and metrics
    """
    try:
        import os
        import glob
        
        metrics_path = training_service.metrics_path
        
        # Get all training files
        training_files = glob.glob(os.path.join(metrics_path, "training_*.json"))
        training_files.sort(key=os.path.getmtime, reverse=True)
        
        history = []
        for file_path in training_files[:limit]:
            try:
                with open(file_path, 'r') as f:
                    training_data = json.load(f)
                    history.append({
                        'training_id': training_data.get('training_id'),
                        'start_time': training_data.get('start_time'),
                        'end_time': training_data.get('end_time'),
                        'metrics': training_data.get('metrics', {}),
                        'model_info': training_data.get('model_info', {})
                    })
            except Exception as e:
                logger.warning(f"Could not load training file {file_path}: {str(e)}")
        
        return {
            "message": f"Retrieved {len(history)} training records",
            "history": history,
            "total_available": len(training_files)
        }
        
    except Exception as e:
        logger.error(f"Failed to get training history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-metrics", response_model=Dict[str, Any])
async def get_current_model_metrics(
    current_user = Depends(get_current_user)
):
    """
    Get current model performance metrics
    """
    try:
        # This would typically load the latest model metrics
        # For now, return basic model information
        
        model_info = {
            'model_type': 'InsightFace R100',
            'model_path': training_service.model_path,
            'embedding_size': 512,
            'current_threshold': 0.6,
            'last_training': None,
            'performance_metrics': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        }
        
        # Try to load latest metrics
        import os
        import glob
        
        metrics_files = glob.glob(os.path.join(training_service.metrics_path, "training_*.json"))
        if metrics_files:
            latest_file = max(metrics_files, key=os.path.getmtime)
            try:
                with open(latest_file, 'r') as f:
                    latest_data = json.load(f)
                    model_info['last_training'] = latest_data.get('training_id')
                    model_info['performance_metrics'] = latest_data.get('metrics', {})
            except Exception as e:
                logger.warning(f"Could not load latest metrics: {str(e)}")
        
        return {
            "message": "Current model metrics",
            "model_info": model_info,
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"Failed to get model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
