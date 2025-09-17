# Hướng dẫn sử dụng Postman Collection

## 📥 Import Collection và Environment

### 1. Import Collection
1. Mở Postman
2. Click **Import** button
3. Chọn file: `postman/EKYC_Service_API.postman_collection.json`
4. Click **Import**

### 2. Import Environment
1. Click **Import** button
2. Chọn file: `postman/EKYC_Service.postman_environment.json`
3. Click **Import**
4. Chọn environment **"EKYC Service Environment"** từ dropdown

## 🚀 Quy trình Testing

### Bước 1: Kiểm tra Service
```
GET /health
```
- Kiểm tra service có đang chạy không
- Không cần authentication

### Bước 2: Đăng nhập
```
POST /api/v1/auth/login
```
- Username: `testuser`
- Password: `testpassword`
- **Auto-save token:** Collection tự động lưu access_token vào biến môi trường

### Bước 3: Tạo Session
```
POST /api/v1/ekyc/sessions
```
- **Auto-save session_id:** Collection tự động lưu session_id
- User_id: `testuser`

### Bước 4: Upload CMND
```
POST /api/v1/ekyc/sessions/{session_id}/id-document
```
- Chọn file ảnh CMND mặt trước (bắt buộc)
- Chọn file ảnh CMND mặt sau (tùy chọn)
- **Format:** JPG, JPEG, PNG
- **Dung lượng:** Tối đa 10MB

### Bước 5: Upload Selfie
```
POST /api/v1/ekyc/sessions/{session_id}/selfie
```
- Chọn file ảnh selfie
- Phải upload CMND trước

### Bước 6: Kiểm tra kết quả
```
GET /api/v1/ekyc/sessions/{session_id}
```
- Xem chi tiết session và kết quả xác minh
- Kiểm tra `final_decision`: approved/rejected/pending_review

## 🔧 Cấu hình Environment Variables

| Variable | Mô tả | Default Value |
|----------|-------|---------------|
| `base_url` | URL của API server | `http://localhost:8000` |
| `access_token` | JWT token (auto-saved) | _(empty)_ |
| `session_id` | Session ID (auto-saved) | _(empty)_ |
| `task_id` | Task ID để theo dõi | _(empty)_ |
| `user_id` | Username | `testuser` |
| `password` | Password | `testpassword` |

## 📋 Collection Structure

```
EKYC Service API/
├── 🏥 Health Check
├── 🔐 Authentication/
│   └── Login
├── 📋 EKYC Sessions/
│   ├── Create Session
│   ├── Get Session Details
│   └── List Sessions
├── 📎 File Upload/
│   ├── Upload ID Document (Front Only)
│   ├── Upload ID Document (Front + Back)
│   └── Upload Selfie
└── 📊 Monitoring/
    └── Get Task Status
```

## 🎯 Tips và Best Practices

### Authentication
- Token có hiệu lực **30 phút**
- Collection tự động thêm Bearer token vào header
- Nếu token hết hạn, chạy lại request **Login**

### File Upload
- **Chuẩn bị ảnh test:**
  - Ảnh CMND rõ nét, không bị mờ
  - Ảnh selfie có ánh sáng tốt
  - Avoid ảnh quá nhỏ hoặc quá lớn

### Error Handling
- **401:** Token hết hạn → Login lại
- **400:** File không hợp lệ → Kiểm tra format/size
- **404:** Session không tồn tại → Tạo session mới
- **403:** Không có quyền truy cập

### Workflow Testing
1. **Sequential Testing:** Chạy requests theo thứ tự
2. **Auto Variables:** Collection tự động lưu token và session_id
3. **Response Validation:** Kiểm tra response status và data

## 🔍 Monitoring và Debug

### Response Logs
- Collection tự động log response status và time
- Check Console tab để xem detailed logs

### Task Monitoring
```javascript
// Script để poll task status
const taskId = pm.response.json().task_id;
if (taskId) {
    pm.collectionVariables.set('task_id', taskId);
    // Có thể setup polling request
}
```

### Session Status Tracking
```javascript
// Check processing stages
const session = pm.response.json();
console.log('Processing stages:', session.processing_stages);
console.log('Final decision:', session.final_decision);
```

## 📊 Example Responses

### Login Success
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
}
```

### Session Created
```json
{
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "user_id": "testuser",
    "status": "pending",
    "processing_stages": {
        "id_document_uploaded": false,
        "id_document_processed": false,
        "selfie_uploaded": false,
        "selfie_processed": false,
        "face_match_completed": false
    },
    "created_at": "2024-01-01T00:00:00Z"
}
```

### Final Result
```json
{
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "completed",
    "id_card_data": {
        "id_number": "123456789012",
        "full_name": "NGUYEN VAN A",
        "date_of_birth": "01/01/1990",
        "gender": "Nam"
    },
    "face_match_score": 0.85,
    "liveness_score": 0.92,
    "final_decision": "approved"
}
```

## 🛠️ Troubleshooting

### Common Issues

**1. Connection Error**
```
Error: connect ECONNREFUSED 127.0.0.1:8000
```
- **Solution:** Khởi động API server trước: `python main.py`

**2. Authentication Failed**
```
401 Unauthorized
```
- **Solution:** Chạy Login request để lấy token mới

**3. File Upload Failed**
```
400 Bad Request - Invalid file
```
- **Solution:** Kiểm tra format file (JPG/PNG) và dung lượng (<10MB)

**4. Task Processing Timeout**
```
Session stuck in "in_progress" status
```
- **Solution:** Kiểm tra Celery worker: `python celery_worker.py`

### Debug Steps
1. Check service health: `/health`
2. Verify authentication: Login request
3. Check request format và headers
4. Monitor server logs for errors
5. Verify file upload requirements
