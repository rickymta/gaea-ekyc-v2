# EKYC Service v2

Dịch vụ xác minh danh tính điện tử (eKYC) sử dụng FastAPI, InsightFace, PostgreSQL, MinIO, và Celery.

## 🚀 Tính năng

- **JWT Authentication** - Xác thực an toàn
- **OCR Processing** - Trích xuất thông tin từ CMND/CCCD
- **Face Recognition** - Nhận diện khuôn mặt với InsightFace
- **Liveness Detection** - Phát hiện ảnh selfie thật
- **Async Processing** - Xử lý bất đồng bộ với Celery
- **Secure Storage** - Lưu trữ file với MinIO
- **Clean Architecture** - Cấu trúc code chuyên nghiệp

## 📁 Cấu trúc dự án

```
gaea-ekyc-v2/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration settings
│   ├── database.py             # Database setup
│   ├── models.py               # SQLAlchemy models
│   ├── schemas.py              # Pydantic schemas
│   ├── dependencies.py         # Authentication dependencies
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py             # Authentication routes
│   │   ├── sessions.py         # EKYC session routes
│   │   ├── assets.py           # File upload routes
│   │   └── monitoring.py       # Health check & metrics
│   ├── services/
│   │   ├── __init__.py
│   │   ├── session_service.py  # Session business logic
│   │   ├── asset_service.py    # Asset management
│   │   ├── storage_service.py  # MinIO file operations
│   │   └── face_service.py     # Face recognition engine
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── celery_app.py       # Celery configuration
│   │   └── processing_tasks.py # Background tasks
│   ├── utils/
│   │   └── __init__.py
│   ├── webhooks/
│   │   └── __init__.py
│   └── tests/
│       └── __init__.py
├── migrations/
│   └── __init__.py
├── docs/
├── scripts/
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml
```

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd gaea-ekyc-v2
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env với thông tin database, Redis, MinIO
```

### 5. Cài đặt database

```bash
# Tạo database PostgreSQL
createdb ekyc_db

# Chạy migrations (tùy chọn)
alembic upgrade head
```

### 6. Chạy services

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: MinIO
minio server /data --console-address ":9001"

# Terminal 3: Celery worker
celery -A app.tasks.celery_app worker --loglevel=info

# Terminal 4: FastAPI
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

Sau khi chạy service, truy cập:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Sử dụng API

### 1. Đăng nhập

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -d "username=testuser&password=testpassword"
```

### 2. Tạo EKYC Session

```bash
curl -X POST "http://localhost:8000/api/v1/sessions" \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "testuser"}'
```

### 3. Upload ảnh CMND

```bash
curl -X POST "http://localhost:8000/api/v1/sessions/<session_id>/upload" \\
  -H "Authorization: Bearer <token>" \\
  -F "asset_type=id_front" \\
  -F "file=@cmnd_front.jpg"
```

### 4. Upload selfie

```bash
curl -X POST "http://localhost:8000/api/v1/sessions/<session_id>/upload" \\
  -H "Authorization: Bearer <token>" \\
  -F "asset_type=selfie" \\
  -F "file=@selfie.jpg"
```

## 🐳 Docker

```bash
# Chạy tất cả services
docker-compose up -d

# Chỉ chạy dependencies
docker-compose up -d postgres redis minio

# View logs
docker-compose logs -f
```

## 🧪 Testing

```bash
# Chạy tests
pytest

# Test với coverage
pytest --cov=app

# Test specific module
pytest app/tests/test_sessions.py
```

## 🔒 Bảo mật

- JWT tokens với expiration
- File upload validation
- Size limits và type checking
- SQL injection protection với SQLAlchemy
- CORS configuration
- Secure file storage với MinIO

## 📊 Monitoring

- Health check: `GET /health`
- Metrics: `GET /api/v1/metrics`
- Task status: `GET /api/v1/tasks/{task_id}/status`
- System info: `GET /api/v1/system/info`

## 🚀 Production Deployment

### 1. Environment Variables

```bash
export ENVIRONMENT="production"
export DEBUG=false
export SECRET_KEY="strong-secret-key"
export DATABASE_URL="postgresql://user:pass@db:5432/ekyc"
```

### 2. Gunicorn

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Nginx (tùy chọn)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - xem [LICENSE](LICENSE) file.

## 📞 Support

- Email: support@example.com
- Issues: GitHub Issues
- Docs: [Documentation](docs/)
