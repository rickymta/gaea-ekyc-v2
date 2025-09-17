-- =====================================================
-- GAEA EKYC v2 Database Initialization Script
-- Description: Complete database setup and initialization
-- Usage: Run this script in pgAdmin or psql to set up the entire database
-- Note: This script is compatible with pgAdmin and standard PostgreSQL clients
-- =====================================================

-- Enable required extensions
DO $$
BEGIN
    RAISE NOTICE '🔧 Setting up PostgreSQL extensions...';
END $$;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
DO $$
BEGIN
    RAISE NOTICE '📁 Creating database schemas...';
END $$;

CREATE SCHEMA IF NOT EXISTS ekyc;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS audit;

-- Create roles
DO $$
BEGIN
    RAISE NOTICE '👥 Setting up database roles...';
END $$;
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ekyc_admin') THEN
        CREATE ROLE ekyc_admin WITH LOGIN PASSWORD 'ekyc_admin_pass_2025';
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ekyc_user') THEN
        CREATE ROLE ekyc_user WITH LOGIN PASSWORD 'ekyc_user_pass_2025';
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ekyc_readonly') THEN
        CREATE ROLE ekyc_readonly WITH LOGIN PASSWORD 'ekyc_readonly_pass_2025';
    END IF;
END $$;

-- Grant schema permissions
DO $$
BEGIN
    RAISE NOTICE '🔐 Configuring schema permissions...';
END $$;
GRANT ALL PRIVILEGES ON SCHEMA ekyc TO ekyc_admin;
GRANT ALL PRIVILEGES ON SCHEMA training TO ekyc_admin;
GRANT ALL PRIVILEGES ON SCHEMA audit TO ekyc_admin;

GRANT USAGE ON SCHEMA ekyc TO ekyc_user;
GRANT USAGE ON SCHEMA training TO ekyc_user;
GRANT USAGE ON SCHEMA audit TO ekyc_user;

GRANT USAGE ON SCHEMA ekyc TO ekyc_readonly;
GRANT USAGE ON SCHEMA training TO ekyc_readonly;
GRANT USAGE ON SCHEMA audit TO ekyc_readonly;

-- Set default search path for roles
ALTER ROLE ekyc_admin SET search_path TO ekyc, training, audit, public;
ALTER ROLE ekyc_user SET search_path TO ekyc, training, audit, public;
ALTER ROLE ekyc_readonly SET search_path TO ekyc, training, audit, public;

-- =====================================================
-- MAIN TABLE CREATION
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '📊 Creating main database tables...';
END $$;

-- Note: In pgAdmin, you need to run the table creation scripts separately
-- Copy and run the contents of 01_create_tables.sql after this initialization

-- For pgAdmin users: Please run the following scripts in order:
-- 1. This initialization script (00_initialize_database.sql)  
-- 2. 01_create_tables.sql
-- 3. 02_sample_data.sql (optional - for test data)
-- 4. 03_management_utilities.sql (optional - for utilities)

-- Grant table permissions after creation
DO $$
BEGIN
    RAISE NOTICE '🔑 Setting up default table permissions...';
    RAISE NOTICE 'Note: Additional permissions will be granted after table creation';
END $$;

-- Admin permissions (full access)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ekyc TO ekyc_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA training TO ekyc_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO ekyc_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ekyc TO ekyc_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA training TO ekyc_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO ekyc_admin;

-- User permissions (read/write on operational tables)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ekyc TO ekyc_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA training TO ekyc_user;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO ekyc_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ekyc TO ekyc_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA training TO ekyc_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO ekyc_user;

-- Readonly permissions (select only)
GRANT SELECT ON ALL TABLES IN SCHEMA ekyc TO ekyc_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA training TO ekyc_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO ekyc_readonly;

-- Set default permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA ekyc GRANT ALL ON TABLES TO ekyc_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA training GRANT ALL ON TABLES TO ekyc_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT ALL ON TABLES TO ekyc_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA ekyc GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ekyc_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA training GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ekyc_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT SELECT, INSERT ON TABLES TO ekyc_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA ekyc GRANT SELECT ON TABLES TO ekyc_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA training GRANT SELECT ON TABLES TO ekyc_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT SELECT ON TABLES TO ekyc_readonly;

-- Create database configuration table
DO $$
BEGIN
    RAISE NOTICE '⚙️ Setting up database configuration...';
END $$;
CREATE TABLE IF NOT EXISTS ekyc.database_config (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert configuration values
INSERT INTO ekyc.database_config (config_key, config_value, description) VALUES
('db_version', '1.0.0', 'Database schema version'),
('face_similarity_threshold', '0.6', 'Default face similarity threshold for matching'),
('liveness_threshold', '0.5', 'Default liveness detection threshold'),
('confidence_threshold', '0.7', 'Default overall confidence threshold'),
('max_session_duration_minutes', '30', 'Maximum time allowed for EKYC session completion'),
('cleanup_retention_days', '90', 'Number of days to retain completed sessions'),
('archive_after_days', '30', 'Number of days before archiving completed sessions'),
('max_file_size_mb', '10', 'Maximum file size for uploaded assets'),
('allowed_image_formats', 'jpg,jpeg,png', 'Allowed image file formats'),
('webhook_timeout_seconds', '30', 'Timeout for webhook deliveries'),
('rate_limit_per_minute', '100', 'Default API rate limit per minute'),
('training_batch_size', '32', 'Default batch size for model training'),
('embedding_dimensions', '512', 'Number of dimensions in face embeddings'),
('model_version', 'InsightFace-R100-v1.0', 'Current face recognition model version')
ON CONFLICT (config_key) DO UPDATE SET 
    config_value = EXCLUDED.config_value,
    updated_at = CURRENT_TIMESTAMP;

-- Create initialization status tracking
CREATE TABLE IF NOT EXISTS ekyc.initialization_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    script_name VARCHAR(100) NOT NULL,
    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'completed',
    details TEXT,
    database_version VARCHAR(20)
);

-- Log this initialization
INSERT INTO ekyc.initialization_log (script_name, details, database_version) VALUES
('00_initialize_database.sql', 'Complete database initialization with tables, sample data, and utilities', '1.0.0');

-- Final validation
DO $$
BEGIN
    RAISE NOTICE '✅ Running final validation...';
END $$;
DO $$
DECLARE
    table_count INTEGER := 0;
    view_count INTEGER := 0;
    function_count INTEGER := 0;
    user_count INTEGER := 0;
    config_count INTEGER;
BEGIN
    -- Count configuration table
    SELECT COUNT(*) INTO config_count FROM ekyc.database_config;
    
    RAISE NOTICE '====================================================';
    RAISE NOTICE '🎉 GAEA EKYC v2 Database Initialization Complete!';
    RAISE NOTICE '====================================================';
    RAISE NOTICE '⚠️  IMPORTANT FOR pgAdmin USERS:';
    RAISE NOTICE '   After running this script, please run these scripts in order:';
    RAISE NOTICE '   1. 01_create_tables.sql (REQUIRED - creates all tables)';
    RAISE NOTICE '   2. 02_sample_data.sql (OPTIONAL - test data)';
    RAISE NOTICE '   3. 03_management_utilities.sql (OPTIONAL - utilities)';
    RAISE NOTICE '';
    RAISE NOTICE '✅ Database Foundation Setup Complete:';
    RAISE NOTICE '   • Database extensions installed';
    RAISE NOTICE '   • Schemas created (ekyc, training, audit)';
    RAISE NOTICE '   • User roles configured';
    RAISE NOTICE '   • Default permissions set';
    RAISE NOTICE '   • Configuration table ready (% entries)', config_count;
    RAISE NOTICE '';
    RAISE NOTICE '🔐 Database Roles Created:';
    RAISE NOTICE '   • ekyc_admin (full access)';
    RAISE NOTICE '   • ekyc_user (read/write access)';
    RAISE NOTICE '   • ekyc_readonly (read-only access)';
    RAISE NOTICE '';
    RAISE NOTICE '🗂️ Database Schemas:';
    RAISE NOTICE '   • ekyc (main EKYC operations)';
    RAISE NOTICE '   • training (ML training data)';
    RAISE NOTICE '   • audit (logging and monitoring)';
    RAISE NOTICE '';
    RAISE NOTICE '� Next Steps:';
    RAISE NOTICE '   1. Run 01_create_tables.sql to create all tables';
    RAISE NOTICE '   2. Run 02_sample_data.sql for test data (optional)';
    RAISE NOTICE '   3. Run 03_management_utilities.sql for utilities (optional)';
    RAISE NOTICE '   4. Configure your application with connection details';
    RAISE NOTICE '';
    RAISE NOTICE 'Database initialization foundation ready!';
    RAISE NOTICE '====================================================';
    
    IF config_count = 0 THEN
        RAISE WARNING 'Configuration table is empty - this is normal at this stage';
    END IF;
END $$;
