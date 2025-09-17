# 🚀 HƯỚNG DẪN NHANH - SETUP DATABASE VỚI PGADMIN

## Cách đơn giản nhất để setup database GAEA EKYC v2

### Bước 1: Mở pgAdmin
1. Mở pgAdmin 4
2. Kết nối tới PostgreSQL server của bạn
3. Tạo database mới (nếu chưa có): 
   - Right-click "Databases" → "Create" → "Database"
   - Đặt tên: `gaea_ekyc` (hoặc tên khác)

### Bước 2: Chạy script setup
1. Right-click vào database `gaea_ekyc` → "Query Tool"
2. Mở file `pgadmin_complete_setup.sql` (File → Open)
3. Click nút "Execute" (▶️) hoặc nhấn F5
4. Chờ script chạy hoàn tất (khoảng 1-2 phút)

### Bước 3: Kiểm tra kết quả
Sau khi script chạy xong, bạn sẽ thấy:

```
🎉 GAEA EKYC v2 Database Setup Complete!
====================================================
📊 Database Statistics:
   • Tables Created: 15
   • Views Created: 0  
   • Functions Created: 2
   • Sample Users: 3
   • Configuration Entries: 14
```

### Bước 4: Kiểm tra database structure
Refresh database trong pgAdmin và bạn sẽ thấy:

📁 **Schemas:**
- `ekyc` - Chứa các bảng chính cho EKYC
- `training` - Chứa dữ liệu training AI  
- `audit` - Chứa logs và audit trails

📁 **Tables trong schema `ekyc`:**
- `users` - Tài khoản người dùng
- `ekyc_sessions` - Phiên xác minh EKYC
- `ekyc_assets` - File upload (ảnh CCCD, selfie)
- `ekyc_verifications` - Kết quả xác minh
- `database_config` - Cấu hình hệ thống

### Bước 5: Test connection
Cập nhật file `.env` trong project với thông tin kết nối:

```env
DATABASE_URL=postgresql://ekyc_user:ekyc_user_pass_2025@192.168.1.127:5432/gaea_ekyc
```

**⚠️ Lưu ý:**
- Thay `192.168.1.127` bằng IP của PostgreSQL server
- Thay `gaea_ekyc` bằng tên database bạn đã tạo
- Đổi password mặc định trong production!

### 🔑 Thông tin đăng nhập

**Database Roles được tạo:**
- `ekyc_admin` / `ekyc_admin_pass_2025` - Full quyền admin
- `ekyc_user` / `ekyc_user_pass_2025` - Quyền ứng dụng
- `ekyc_readonly` / `ekyc_readonly_pass_2025` - Chỉ đọc

**Test Accounts:**
- `admin@ekyc.local` / `test123` - Admin user
- `user1@test.com` / `test123` - Test user 1  
- `user2@test.com` / `test123` - Test user 2

### ✅ Hoàn tất!

Database đã sẵn sàng! Bạn có thể:
1. Khởi động ứng dụng FastAPI
2. Test các API endpoints
3. Sử dụng test accounts để thử nghiệm EKYC

### 🛠️ Nếu có lỗi

**Lỗi thường gặp:**
1. **Permission denied**: Đảm bảo user PostgreSQL có quyền tạo database/schema
2. **Extension missing**: Chạy `CREATE EXTENSION IF NOT EXISTS "uuid-ossp";` với superuser
3. **Role exists**: Script sẽ bỏ qua nếu role đã tồn tại

**Kiểm tra logs:**
- Xem tab "Messages" trong pgAdmin Query Tool
- Script sẽ hiển thị progress và kết quả chi tiết

### 📞 Cần hỗ trợ?

Nếu script báo lỗi, hãy:
1. Copy toàn bộ error message
2. Kiểm tra quyền user PostgreSQL
3. Đảm bảo database trống hoặc không có conflict

---
**Script version**: 1.0.0  
**Compatible**: PostgreSQL 12+, pgAdmin 4+
