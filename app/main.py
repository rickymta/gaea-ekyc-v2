from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.config import settings
from app.database import create_tables
from app.routers import auth, sessions, assets, monitoring, training, ekyc_verification

# Logging setup
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app with comprehensive OpenAPI documentation
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="""
## EKYC Service API

Dịch vụ xác minh danh tính điện tử (eKYC) sử dụng AI và xử lý tài liệu.

### Tính năng chính:
- 🔐 **JWT Authentication** - Xác thực an toàn
- 📄 **OCR Processing** - Trích xuất thông tin từ CMND/CCCD
- 👤 **Face Recognition** - Nhận diện khuôn mặt với InsightFace
- 🔍 **Liveness Detection** - Phát hiện ảnh selfie thật
- 📊 **Async Processing** - Xử lý bất đồng bộ với Celery
- 💾 **Secure Storage** - Lưu trữ file với MinIO

### Workflow:
1. **Đăng nhập** để lấy JWT token
2. **Tạo session** EKYC mới
3. **Upload ảnh CMND** (mặt trước/sau)
4. **Upload ảnh selfie** để xác minh
5. **Nhận kết quả** xác minh tự động

### API Response Format:
- Tất cả API trả về JSON format
- Lỗi được standardized với error codes
- Async tasks có thể theo dõi qua task_id
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "Đăng nhập và quản lý JWT tokens"
        },
        {
            "name": "EKYC Sessions", 
            "description": "Quản lý sessions xác minh danh tính"
        },
        {
            "name": "File Upload",
            "description": "Upload và xử lý ảnh CMND, selfie"
        },
        {
            "name": "Monitoring",
            "description": "Health check và monitoring tasks"
        }
    ],
    contact={
        "name": "EKYC API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(assets.router, prefix="/api/v1")
app.include_router(monitoring.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(ekyc_verification.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting EKYC Service...")
    
    # Create database tables
    create_tables()
    logger.info("Database tables created/verified")
    
    logger.info("EKYC Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down EKYC Service...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
