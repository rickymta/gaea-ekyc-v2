# GAEA EKYC v2 - Database Setup Guide

Complete guide for setting up the PostgreSQL database for GAEA EKYC v2 system.

## 📋 Prerequisites

- PostgreSQL 12+ installed and running
- pgAdmin 4 or psql command-line tool available
- Administrative access to PostgreSQL server
- At least 1GB free disk space

## 🚀 Quick Setup

### Option 1: pgAdmin Complete Setup (Recommended)

**Easiest way - Single script for pgAdmin:**

1. Open pgAdmin and connect to your PostgreSQL server
2. Right-click on your database → Query Tool
3. Open the file `pgadmin_complete_setup.sql`
4. Execute the entire script (F5)

This single script will:
- ✅ Create all schemas, tables, and indexes
- ✅ Set up user roles and permissions  
- ✅ Load sample data for testing
- ✅ Create basic utility functions
- ✅ Configure the database completely

### Option 2: Command Line Setup

Run the master initialization script:

```bash
# Navigate to SQL directory
cd sql/

# Run complete initialization
psql -U postgres -d your_database_name -f 00_initialize_database.sql
psql -U postgres -d your_database_name -f 01_create_tables.sql
psql -U postgres -d your_database_name -f 02_sample_data.sql
psql -U postgres -d your_database_name -f 03_management_utilities.sql
```

### Option 3: Step-by-Step Setup

If you prefer to run scripts individually in pgAdmin:

1. **Initialize database**: Run `00_initialize_database.sql`
2. **Create tables**: Run `01_create_tables.sql`  
3. **Load sample data**: Run `02_sample_data.sql` (optional)
4. **Add utilities**: Run `03_management_utilities.sql` (optional)

## 🗂️ Database Structure

### Schemas

The database is organized into three main schemas:

- **`ekyc`** - Core EKYC operations (sessions, assets, verifications)
- **`training`** - Machine learning and training data
- **`audit`** - Logging, monitoring, and audit trails

### Key Tables

#### EKYC Schema
- `users` - User accounts and authentication
- `ekyc_sessions` - EKYC verification sessions
- `ekyc_assets` - Uploaded documents and images
- `ekyc_verifications` - Verification results and analysis
- `webhook_deliveries` - Webhook notifications
- `api_keys` - API access management
- `task_queue` - Background task processing

#### Training Schema
- `face_identities` - Known person identities
- `face_embeddings` - Face recognition embeddings (512-dim)
- `datasets` - Training datasets
- `training_sessions` - Model training records
- `user_feedback` - User feedback for model improvement

#### Audit Schema
- `activity_logs` - User and system activity tracking
- `security_events` - Security-related events

## 🔐 Database Roles

Three roles are automatically created:

| Role | Permissions | Use Case |
|------|------------|----------|
| `ekyc_admin` | Full access to all schemas | Database administration, maintenance |
| `ekyc_user` | Read/write on operational tables | Application runtime user |
| `ekyc_readonly` | Read-only access | Reporting, analytics, monitoring |

### Default Passwords

**⚠️ Change these passwords in production!**

- `ekyc_admin`: `ekyc_admin_pass_2025`
- `ekyc_user`: `ekyc_user_pass_2025`
- `ekyc_readonly`: `ekyc_readonly_pass_2025`

## 📊 Sample Data

The initialization includes sample data for testing:

- 3 test users (including admin)
- 3 EKYC sessions (successful, failed, pending)
- Sample face identities and embeddings
- Training datasets and sessions
- Audit logs and activity records

### Test Accounts

- **Admin**: `admin@ekyc.local` / password: `test123`
- **User 1**: `user1@test.com` / password: `test123`
- **User 2**: `user2@test.com` / password: `test123`

## 🛠️ Management Utilities

### Monitoring Views

Query these views for system insights:

```sql
-- Session performance over time
SELECT * FROM session_performance_stats;

-- Training model performance
SELECT * FROM training_performance_stats;

-- Storage usage by asset type
SELECT * FROM asset_storage_stats;

-- User feedback analysis
SELECT * FROM user_feedback_analysis;

-- Overall system health
SELECT * FROM system_health_monitor;
```

### Utility Functions

```sql
-- Clean up old sessions (older than 30 days)
SELECT * FROM cleanup_old_sessions(30);

-- Get database statistics
SELECT * FROM get_database_stats();

-- Find faces similar to a given embedding
SELECT * FROM find_similar_faces(
    ARRAY[0.1, 0.2, ...], -- 512-dimensional embedding
    0.6,  -- similarity threshold
    10    -- max results
);

-- Archive completed sessions
SELECT * FROM archive_completed_sessions(7);

-- Audit suspicious activities
SELECT * FROM audit_suspicious_activities(7);
```

### Maintenance Tasks

```sql
-- Rebuild all indexes
SELECT * FROM rebuild_all_indexes();

-- Update table statistics
SELECT * FROM update_table_statistics();

-- Export user data
SELECT * FROM export_user_data('user_uuid_here');

-- Analyze query performance
SELECT * FROM analyze_query_performance();
```

## ⚡ Performance Optimization

### Indexes

The database includes optimized indexes for:

- Fast session lookups by user and status
- Efficient face embedding similarity searches
- Quick asset retrieval by session and type
- Audit log searches by user and timeframe

### Configuration

Key performance settings in `database_config` table:

```sql
-- View current configuration
SELECT * FROM ekyc.database_config ORDER BY config_key;

-- Update configuration
UPDATE ekyc.database_config 
SET config_value = '0.65' 
WHERE config_key = 'face_similarity_threshold';
```

## 🔍 Monitoring & Health Checks

### System Health Dashboard

```sql
-- Quick health overview
SELECT * FROM system_health_monitor;
```

### Performance Metrics

```sql
-- Sessions processed today
SELECT COUNT(*) as sessions_today 
FROM ekyc_sessions 
WHERE DATE(created_at) = CURRENT_DATE;

-- Average processing time
SELECT AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_seconds
FROM ekyc_sessions 
WHERE status = 'completed' 
AND DATE(created_at) = CURRENT_DATE;

-- Success rate
SELECT 
    ROUND(
        COUNT(*) FILTER (WHERE final_decision = 'APPROVED')::FLOAT / 
        COUNT(*)::FLOAT * 100, 1
    ) as success_rate_percent
FROM ekyc_sessions 
WHERE status = 'completed' 
AND DATE(created_at) = CURRENT_DATE;
```

## 🛡️ Security Features

### Audit Logging

All user activities are automatically logged:

```sql
-- Recent user activities
SELECT * FROM audit.activity_logs 
WHERE created_at >= CURRENT_DATE 
ORDER BY created_at DESC 
LIMIT 50;
```

### Suspicious Activity Detection

```sql
-- Check for suspicious patterns
SELECT * FROM audit_suspicious_activities(7);
```

### Data Protection

- Sensitive data (passwords, API keys) are hashed
- Face embeddings are stored as encrypted arrays
- Comprehensive audit trails for compliance

## 🔧 Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Ensure proper role permissions
   GRANT ALL PRIVILEGES ON SCHEMA ekyc TO your_user;
   ```

2. **Extension Missing**
   ```sql
   -- Install required extensions
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   CREATE EXTENSION IF NOT EXISTS "pgcrypto";
   ```

3. **Large Embeddings Performance**
   ```sql
   -- Check embedding index usage
   EXPLAIN ANALYZE SELECT * FROM find_similar_faces(your_embedding, 0.6, 10);
   ```

### Validation Queries

```sql
-- Verify table counts
SELECT 
    schemaname,
    COUNT(*) as table_count
FROM pg_tables 
WHERE schemaname IN ('ekyc', 'training', 'audit')
GROUP BY schemaname;

-- Check sample data
SELECT 
    (SELECT COUNT(*) FROM ekyc.users) as users,
    (SELECT COUNT(*) FROM ekyc.ekyc_sessions) as sessions,
    (SELECT COUNT(*) FROM training.face_identities) as identities;
```

## 📈 Scaling Considerations

### For High Volume

1. **Partitioning**: Consider partitioning large tables by date
2. **Connection Pooling**: Use pgBouncer for connection management
3. **Read Replicas**: Set up read replicas for reporting queries
4. **Archiving**: Regularly archive old data using provided utilities

### Storage Management

```sql
-- Monitor storage usage
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname IN ('ekyc', 'training', 'audit')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## 🎯 Next Steps

After database setup:

1. **Update Configuration**: Modify `database_config` for your environment
2. **Change Passwords**: Update all default passwords
3. **Setup Monitoring**: Configure your monitoring tools to use the provided views
4. **Test Integration**: Run application tests against the database
5. **Performance Tuning**: Adjust PostgreSQL settings for your hardware

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review PostgreSQL logs for detailed error messages
3. Validate permissions and role assignments
4. Ensure all required extensions are installed

---

**Database Version**: 1.0.0  
**Compatible with**: PostgreSQL 12+  
**Last Updated**: September 2025
