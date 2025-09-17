# Enhanced EKYC Service v2.0

Hệ thống Enhanced EKYC với đầy đủ tính năng liveness detection, OCR và quality assessment cho xác minh danh tính điện tử.

## 🌟 Tính năng chính

### ✅ Hoàn thành
- **🔥 Liveness Detection**: Sử dụng Silent Face Anti-Spoofing để phát hiện video/ảnh thật
- **📄 OCR Vietnamese ID Cards**: PaddleOCR + EasyOCR cho CCCD Việt Nam
- **📸 Image Quality Assessment**: Đánh giá chất lượng ảnh selfie và CCCD
- **👤 Face Recognition**: InsightFace cho detection và matching
- **🎯 Complete EKYC Pipeline**: Workflow hoàn chỉnh từ ảnh đến quyết định
- **🚀 FastAPI Endpoints**: RESTful API với async processing
- **💾 Database Storage**: Lưu trữ kết quả verification

### 🎯 Use Cases
- Xác minh danh tính cho ngân hàng số
- Onboarding khách hàng KYC/AML
- Xác thực tài khoản fintech
- Verification cho ví điện tử

## 🏗️ Kiến trúc hệ thống

```
Enhanced EKYC Service
├── 📱 API Layer (FastAPI)
│   ├── /verify-complete (Main endpoint)
│   ├── /verify-liveness-only
│   ├── /extract-id-info
│   ├── /assess-image-quality
│   └── /face-match
├── 🧠 Service Layer
│   ├── EnhancedFaceEngine (Core)
│   ├── LivenessDetectionService
│   ├── VietnamIDCardOCR
│   └── ImageQualityAssessment
├── 🔧 Core Technologies
│   ├── Silent Face Anti-Spoofing
│   ├── InsightFace (ArcFace)
│   ├── PaddleOCR + EasyOCR
│   └── OpenCV + PIL
└── 💾 Data Layer
    ├── PostgreSQL (Results)
    └── File Storage (Images/Videos)
```

## 🚀 Quick Start

### 1. Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd gaea-ekyc-v2

# Install Python dependencies
pip install -r requirements.txt

# Install additional system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-vie
sudo apt-get install ffmpeg

# Windows: Download tesseract và ffmpeg manually
```

### 2. Cấu hình environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Cấu hình chính trong `.env`:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ekyc_db

# Enhanced EKYC Settings
LIVENESS_CONFIDENCE_THRESHOLD=0.7
FACE_MATCH_THRESHOLD=0.6
PADDLEOCR_USE_GPU=false
EASYOCR_GPU=false

# InsightFace Models
INSIGHTFACE_MODEL_PATH=./models
INSIGHTFACE_MODEL=model-r100-ii

# Image Quality
MIN_FACE_SIZE=80
MIN_BRIGHTNESS=80.0
MIN_SHARPNESS=100.0
```

### 3. Download AI Models

```bash
# Create models directory
mkdir -p models

# Download InsightFace models
# Tự động download qua InsightFace library khi khởi chạy

# Optional: Download Silent Face Anti-Spoofing models
# Models sẽ được download automatically khi sử dụng lần đầu
```

### 4. Khởi chạy service

```bash
# Start the server
cd app
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test API

```bash
# Run test suite
python test_enhanced_ekyc_api.py

# Manual test với curl
curl -X GET "http://localhost:8000/api/v1/ekyc/health"
```

## 📚 API Documentation

### Main Endpoint: Complete EKYC Verification

```bash
POST /api/v1/ekyc/verify-complete
```

**Input:**
- `selfie_image` (file, required): Ảnh selfie
- `id_front_image` (file, required): Ảnh mặt trước CCCD  
- `id_back_image` (file, optional): Ảnh mặt sau CCCD
- `liveness_video` (file, optional): Video 3s cho liveness detection
- `session_id` (string, optional): ID session

**Output:**
```json
{
  "status": "success",
  "data": {
    "session_id": "uuid",
    "overall_decision": "APPROVED|REJECTED|MANUAL_REVIEW|ERROR",
    "confidence_score": 0.85,
    "processing_time_ms": 1250,
    
    "verification_summary": {
      "checks_passed": 6,
      "total_checks": 7,
      "failed_checks": ["ocr_extraction"],
      "warnings": []
    },
    
    "quality_assessment": {
      "selfie": {
        "overall_quality": "good",
        "score": 0.82,
        "checks": { /* detailed checks */ }
      },
      "id_card": {
        "overall_quality": "excellent", 
        "score": 0.91
      }
    },
    
    "face_matching": {
      "is_match": true,
      "similarity_score": 0.87,
      "confidence_level": "HIGH"
    },
    
    "liveness_detection": {
      "is_live": true,
      "confidence": 0.89,
      "method": "video_analysis"
    },
    
    "ocr_results": {
      "success": true,
      "extracted_info": {
        "id_number": "123456789012",
        "full_name": "NGUYEN VAN A",
        "date_of_birth": "01/01/1990",
        "place_of_birth": "Ha Noi",
        "address": "123 ABC Street, Ha Noi"
      },
      "overall_confidence": 0.76
    },
    
    "recommendations": []
  }
}
```

### Individual Service Endpoints

#### 1. Liveness Detection Only
```bash
POST /api/v1/ekyc/verify-liveness-only
Content-Type: multipart/form-data

# Upload image or video file
image_or_video: file
session_id: optional
```

#### 2. OCR ID Card Information  
```bash
POST /api/v1/ekyc/extract-id-info
Content-Type: multipart/form-data

id_front_image: file (required)
id_back_image: file (optional)
session_id: optional
```

#### 3. Image Quality Assessment
```bash
POST /api/v1/ekyc/assess-image-quality
Content-Type: multipart/form-data

image: file
image_type: "selfie" | "id_card"  
session_id: optional
```

#### 4. Face Matching Only
```bash
POST /api/v1/ekyc/face-match
Content-Type: multipart/form-data

selfie_image: file
id_image: file
session_id: optional
```

#### 5. Verification Status
```bash
GET /api/v1/ekyc/verification-status/{session_id}
```

## 🔧 Configuration Reference

### Liveness Detection Settings
```python
# Confidence threshold for liveness detection
liveness_confidence_threshold: float = 0.7

# Enable fallback heuristic methods  
liveness_fallback_enabled: bool = True

# Model path for Silent Face Anti-Spoofing
liveness_model_path: str = "./models/liveness"
```

### OCR Settings
```python
# PaddleOCR configuration
paddleocr_use_gpu: bool = False
paddleocr_lang: str = "vi"

# EasyOCR configuration  
easyocr_gpu: bool = False
easyocr_languages: List[str] = ["vi", "en"]

# Tesseract path (auto-detect if empty)
tesseract_path: str = ""
```

### Image Quality Settings
```python
# Face size constraints
min_face_size: int = 80
max_face_size: int = 400

# Lighting thresholds
min_brightness: float = 80.0
max_brightness: float = 220.0

# Sharpness requirements
min_sharpness: float = 100.0
blur_threshold: float = 100.0
```

### Video Processing Settings
```python
# Maximum video file size
max_video_size_mb: int = 10

# Frame analysis rate
video_frame_rate: int = 5  # fps

# Duration constraints
min_video_duration: float = 2.0  # seconds
max_video_duration: float = 5.0  # seconds
```

## 🎯 Decision Making Logic

### Quality Checks (7 total)
1. **Selfie Quality**: Lighting, face detection, obstruction
2. **ID Card Quality**: Corners, readability, lighting
3. **Selfie Face Detection**: Exactly 1 face required
4. **ID Card Face Detection**: Exactly 1 face required  
5. **Face Matching**: Similarity >= threshold
6. **Liveness Detection**: Real person verification
7. **OCR Extraction**: Readable ID information

### Decision Rules
- **APPROVED**: All critical checks pass + confidence >= 0.8
- **MANUAL_REVIEW**: Critical checks pass + confidence 0.6-0.8
- **REJECTED**: Any critical check fails OR confidence < 0.6
- **ERROR**: System/processing error

### Critical Checks (Must Pass)
- Selfie face detection
- ID card face detection  
- Face matching
- Liveness detection

## 🔍 Quality Assessment Criteria

### Selfie Quality Checks
- ✅ Face detection and size
- ✅ Lighting conditions
- ✅ Face obstruction (glasses, mask, hat)
- ✅ Clothing requirements
- ✅ Image sharpness and blur
- ✅ Background analysis

### ID Card Quality Checks  
- ✅ Four corners visible
- ✅ Text readability
- ✅ Photo clarity
- ✅ Overall lighting
- ✅ Document authenticity indicators

## 🚨 Error Handling

### Common Error Codes
- `400`: Invalid input (bad image format, missing required fields)
- `404`: Session not found
- `500`: Internal processing error (model loading, OCR failure)

### Retry Logic
- OCR: Fallback từ PaddleOCR sang EasyOCR
- Liveness: Fallback từ AI model sang heuristic methods
- Face detection: Multiple detection thresholds

## 📊 Performance Metrics

### Typical Processing Times
- **Liveness Detection**: 200-500ms
- **OCR Extraction**: 500-1500ms  
- **Face Matching**: 100-300ms
- **Quality Assessment**: 200-400ms
- **Complete Pipeline**: 1-3 seconds

### Resource Requirements
- **RAM**: 2-4GB (depending on models)
- **CPU**: 2+ cores recommended
- **GPU**: Optional (significant speedup)
- **Storage**: 1-2GB for models

## 🔒 Security Considerations

### Data Privacy
- Images/videos processed in memory
- Optional permanent storage
- Configurable retention period
- GDPR compliance ready

### API Security
- Rate limiting recommended
- File size restrictions
- Input validation
- Session-based tracking

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# CV2 not found
pip install opencv-python

# InsightFace installation
pip install insightface

# OCR libraries
pip install paddlepaddle paddleocr easyocr
```

#### 2. Model Download Issues
```bash
# Clear model cache
rm -rf ~/.insightface/
rm -rf ~/.paddleocr/

# Manual model download
python -c "import insightface; app = insightface.app.FaceAnalysis()"
```

#### 3. Memory Issues
```bash
# Reduce batch size
export OMP_NUM_THREADS=1

# Disable GPU if not enough VRAM
PADDLEOCR_USE_GPU=false
EASYOCR_GPU=false
```

#### 4. Database Connection
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Test connection
python -c "import psycopg2; psycopg2.connect('postgresql://...')"
```

## 📈 Monitoring & Analytics

### Health Check Endpoint
```bash
GET /api/v1/ekyc/health

# Response includes service status
{
  "overall_status": "OK",
  "services": {
    "enhanced_face_engine": "OK",
    "liveness_service": "OK", 
    "ocr_service": "OK",
    "image_quality_service": "OK",
    "database": "OK"
  }
}
```

### Metrics Collection
- Processing times per component
- Success/failure rates
- Quality score distributions
- Common failure reasons

## 🔄 Development Workflow

### Adding New Features
1. Update service layer (`app/services/`)
2. Add new endpoints (`app/api/`)
3. Update configuration (`app/config.py`)
4. Add tests (`test_*.py`)
5. Update documentation

### Testing
```bash
# Unit tests
python -m pytest tests/

# Integration tests  
python test_enhanced_ekyc_api.py

# Load testing
python load_test.py
```

## 📞 Support & Contributing

### Issues & Bug Reports
- Create detailed issue reports
- Include error logs and configurations
- Provide sample images (if possible)

### Feature Requests
- Describe use case clearly
- Consider backwards compatibility
- Provide implementation suggestions

## 📄 License

MIT License - see LICENSE file for details.

---

**Enhanced EKYC Service v2.0** - Powered by AI for reliable identity verification 🚀
