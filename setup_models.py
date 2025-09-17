#!/usr/bin/env python3
"""
EKYC Model Setup Script
Automates the setup of InsightFace models and dependencies for EKYC system
"""

import os
import sys
import logging
import subprocess
import shutil
from pathlib import Path
import requests
import zipfile
import tarfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EKYCModelSetup:
    """Setup class for EKYC model configuration"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.models_dir = self.project_root / "models"
        self.r100_model_dir = self.models_dir / "model-r100-ii"
        
    def setup_directories(self):
        """Create necessary directories"""
        logger.info("Setting up directories...")
        
        directories = [
            self.models_dir,
            self.r100_model_dir,
            self.models_dir / "training_data",
            self.models_dir / "trained_models", 
            self.models_dir / "metrics",
            self.models_dir / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'insightface',
            'opencv-python',
            'numpy',
            'onnxruntime',
            'scikit-learn',
            'matplotlib'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} is not installed")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Please install missing packages with: pip install -r requirements.txt")
            return False
        
        return True
    
    def setup_insightface_model(self):
        """Setup InsightFace R100 model"""
        logger.info("Setting up InsightFace R100 model...")
        
        # Check if model files already exist
        required_files = [
            "model-0000.params",
            "model-symbol.json"
        ]
        
        all_files_exist = all(
            (self.r100_model_dir / file).exists() 
            for file in required_files
        )
        
        if all_files_exist:
            logger.info("✓ InsightFace R100 model files already exist")
            return True
        
        logger.info("Model files not found. Please ensure the following files are in models/model-r100-ii/:")
        for file in required_files:
            exists = "✓" if (self.r100_model_dir / file).exists() else "✗"
            logger.info(f"  {exists} {file}")
        
        if not all_files_exist:
            logger.warning("Some model files are missing. Please obtain the InsightFace R100 model files.")
            return False
        
        return True
    
    def test_model_loading(self):
        """Test if the model can be loaded successfully"""
        logger.info("Testing model loading...")
        
        try:
            # Test basic imports
            import insightface
            import cv2
            import numpy as np
            
            logger.info("✓ Basic imports successful")
            
            # Test model initialization
            from app.services.face_service import FaceEngine
            face_engine = FaceEngine()
            
            logger.info("✓ FaceEngine initialization successful")
            
            # Test with a simple black image
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
            faces = face_engine.detect_faces(test_image)
            
            logger.info(f"✓ Face detection test completed (detected {len(faces)} faces in test image)")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading test failed: {str(e)}")
            return False
    
    def create_sample_training_data(self):
        """Create sample training data structure"""
        logger.info("Creating sample training data structure...")
        
        training_data_dir = self.models_dir / "training_data" / "sample"
        training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample person directories
        sample_persons = ["person_001", "person_002", "person_003"]
        
        for person in sample_persons:
            person_dir = training_data_dir / person
            person_dir.mkdir(exist_ok=True)
            
            # Create a README file with instructions
            readme_file = person_dir / "README.txt"
            readme_content = f"""
Sample Training Data for {person}

Instructions:
1. Place face images for {person} in this directory
2. Use clear, well-lit photos showing the full face
3. Include multiple angles and expressions if possible
4. Supported formats: .jpg, .jpeg, .png
5. Recommended: 3-10 images per person

Example filenames:
- {person}_front.jpg
- {person}_left.jpg
- {person}_right.jpg
- {person}_smile.jpg
"""
            
            with open(readme_file, 'w') as f:
                f.write(readme_content)
        
        logger.info(f"✓ Created sample training structure in {training_data_dir}")
        
        # Create training configuration template
        config_template = {
            "validation_split": 0.2,
            "augmentation": {
                "enabled": False,
                "rotation_range": 10,
                "brightness_range": 0.1,
                "zoom_range": 0.1
            },
            "optimization": {
                "threshold_range": [0.3, 0.9],
                "threshold_step": 0.05
            },
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001
        }
        
        import json
        config_file = self.models_dir / "training_config_template.json"
        with open(config_file, 'w') as f:
            json.dump(config_template, f, indent=2)
        
        logger.info(f"✓ Created training configuration template: {config_file}")
    
    def create_environment_file(self):
        """Create environment configuration for model settings"""
        logger.info("Creating environment configuration...")
        
        env_content = f"""
# InsightFace Model Configuration
INSIGHTFACE_MODEL=model-r100-ii
INSIGHTFACE_MODEL_PATH=./models
FACE_MATCH_THRESHOLD=0.6
LIVENESS_THRESHOLD=0.5
FACE_DETECTION_SIZE=(640,640)
FACE_EMBEDDING_SIZE=512

# Model Performance Settings
MAX_CONCURRENT_REQUESTS=5
BATCH_PROCESSING=false
MODEL_PRECISION=fp32

# Training Settings
TRAINING_DATA_PATH=./models/training_data
TRAINED_MODELS_PATH=./models/trained_models
METRICS_PATH=./models/metrics

# Cache Settings
MODEL_CACHE_PATH=./models/cache
ENABLE_EMBEDDING_CACHE=true
CACHE_EXPIRY_HOURS=24
"""
        
        env_file = self.project_root / ".env.model"
        with open(env_file, 'w') as f:
            f.write(env_content.strip())
        
        logger.info(f"✓ Created model environment file: {env_file}")
    
    def setup_model_permissions(self):
        """Set up proper permissions for model files"""
        logger.info("Setting up model permissions...")
        
        try:
            # Make sure model directory is readable
            os.chmod(self.models_dir, 0o755)
            
            for root, dirs, files in os.walk(self.models_dir):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o755)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o644)
            
            logger.info("✓ Model permissions set successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Could not set permissions: {str(e)}")
            return False
    
    def run_setup(self):
        """Run complete setup process"""
        logger.info("=" * 60)
        logger.info("EKYC Model Setup Starting...")
        logger.info("=" * 60)
        
        steps = [
            ("Setting up directories", self.setup_directories),
            ("Checking dependencies", self.check_dependencies),
            ("Setting up InsightFace model", self.setup_insightface_model),
            ("Testing model loading", self.test_model_loading),
            ("Creating sample training data", self.create_sample_training_data),
            ("Creating environment configuration", self.create_environment_file),
            ("Setting up permissions", self.setup_model_permissions)
        ]
        
        success_count = 0
        total_steps = len(steps)
        
        for step_name, step_func in steps:
            logger.info(f"\n[{success_count + 1}/{total_steps}] {step_name}...")
            try:
                if step_func():
                    success_count += 1
                    logger.info(f"✓ {step_name} completed successfully")
                else:
                    logger.error(f"✗ {step_name} failed")
            except Exception as e:
                logger.error(f"✗ {step_name} failed with error: {str(e)}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Setup completed: {success_count}/{total_steps} steps successful")
        
        if success_count == total_steps:
            logger.info("🎉 EKYC Model setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Verify model files are in models/model-r100-ii/")
            logger.info("2. Add training data to models/training_data/sample/")
            logger.info("3. Test the EKYC API endpoints")
            logger.info("4. Start training with your data if needed")
        else:
            logger.warning("⚠️  Setup completed with some issues. Please review the logs above.")
        
        logger.info("=" * 60)
        
        return success_count == total_steps


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EKYC Model Setup Script")
    parser.add_argument(
        "--project-root", 
        type=str, 
        default=None,
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run model loading test"
    )
    
    args = parser.parse_args()
    
    setup = EKYCModelSetup(args.project_root)
    
    if args.test_only:
        logger.info("Running model loading test only...")
        if setup.test_model_loading():
            logger.info("✓ Model loading test passed!")
            return 0
        else:
            logger.error("✗ Model loading test failed!")
            return 1
    else:
        if setup.run_setup():
            return 0
        else:
            return 1


if __name__ == "__main__":
    sys.exit(main())
