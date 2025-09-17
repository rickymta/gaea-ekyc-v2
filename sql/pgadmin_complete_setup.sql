-- =====================================================
-- GAEA EKYC v2 - COMPLETE DATABASE SETUP FOR PGADMIN
-- Description: All-in-one script for pgAdmin import
-- Usage: Import and run this single script in pgAdmin
-- =====================================================

-- =====================================================
-- PART 1: INITIALIZATION
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '🚀 Starting GAEA EKYC v2 Database Setup...';
    RAISE NOTICE '====================================================';
END $$;

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

-- Set search path for current session
SET search_path TO ekyc, training, audit, public;

-- =====================================================
-- PART 2: CREATE ALL TABLES
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '📊 Creating all database tables...';
END $$;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(200),
    phone_number VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ekyc_sessions table
CREATE TABLE IF NOT EXISTS ekyc_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_reference VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    id_card_data JSONB,
    face_match_score FLOAT,
    liveness_score FLOAT,
    overall_confidence FLOAT,
    final_decision VARCHAR(20),
    processing_stages JSONB DEFAULT '{}',
    verification_results JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ekyc_assets table
CREATE TABLE IF NOT EXISTS ekyc_assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES ekyc_sessions(id) ON DELETE CASCADE,
    asset_type VARCHAR(20) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    original_filename VARCHAR(200),
    file_size BIGINT,
    mime_type VARCHAR(50),
    file_hash VARCHAR(100),
    processed BOOLEAN DEFAULT false,
    processing_result JSONB DEFAULT '{}',
    quality_score FLOAT,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ekyc_verifications table
CREATE TABLE IF NOT EXISTS ekyc_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES ekyc_sessions(id) ON DELETE CASCADE,
    verification_id VARCHAR(50) UNIQUE NOT NULL,
    verification_type VARCHAR(20) NOT NULL,
    id_card_analysis JSONB,
    selfie_analysis JSONB,
    face_match_result JSONB,
    liveness_result JSONB,
    overall_result JSONB,
    confidence_scores JSONB,
    recommendations TEXT[],
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create face_identities table in training schema
CREATE TABLE IF NOT EXISTS training.face_identities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id VARCHAR(50) UNIQUE NOT NULL,
    full_name VARCHAR(200) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create face_embeddings table in training schema
CREATE TABLE IF NOT EXISTS training.face_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identity_id UUID REFERENCES training.face_identities(id) ON DELETE CASCADE,
    embedding FLOAT[] NOT NULL,
    embedding_version VARCHAR(50) NOT NULL,
    source_type VARCHAR(20) NOT NULL,
    source_file_path VARCHAR(500),
    quality_score FLOAT,
    confidence_score FLOAT,
    face_bbox INTEGER[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create datasets table in training schema
CREATE TABLE IF NOT EXISTS training.datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    dataset_type VARCHAR(20) NOT NULL,
    file_path VARCHAR(500),
    total_images INTEGER DEFAULT 0,
    total_persons INTEGER DEFAULT 0,
    statistics JSONB DEFAULT '{}',
    quality_metrics JSONB DEFAULT '{}',
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create training_sessions table in training schema
CREATE TABLE IF NOT EXISTS training.training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_id VARCHAR(50) UNIQUE NOT NULL,
    dataset_id UUID REFERENCES training.datasets(id),
    training_config JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    metrics JSONB DEFAULT '{}',
    model_info JSONB DEFAULT '{}',
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user_feedback table in training schema
CREATE TABLE IF NOT EXISTS training.user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_id VARCHAR(50) UNIQUE NOT NULL,
    verification_id VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id),
    actual_result BOOLEAN NOT NULL,
    system_result BOOLEAN NOT NULL,
    confidence_score FLOAT,
    feedback_type VARCHAR(20) NOT NULL,
    user_comments TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create webhook_deliveries table
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES ekyc_sessions(id) ON DELETE CASCADE,
    webhook_url VARCHAR(500) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    response_code INTEGER,
    response_body TEXT,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    next_attempt_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP
);

-- Create api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(100) NOT NULL,
    api_key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions TEXT[] DEFAULT ARRAY['read'],
    rate_limit_per_minute INTEGER DEFAULT 60,
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP,
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Create task_queue table
CREATE TABLE IF NOT EXISTS task_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(100) UNIQUE NOT NULL,
    task_name VARCHAR(100) NOT NULL,
    task_args JSONB DEFAULT '[]',
    task_kwargs JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending',
    result JSONB,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create activity_logs table in audit schema
CREATE TABLE IF NOT EXISTS audit.activity_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    session_id UUID REFERENCES ekyc_sessions(id),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create security_events table in audit schema
CREATE TABLE IF NOT EXISTS audit.security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    user_id UUID REFERENCES users(id),
    ip_address INET,
    description TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    resolved BOOLEAN DEFAULT false,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- PART 3: CREATE INDEXES
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '📈 Creating performance indexes...';
END $$;

-- Indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Indexes for ekyc_sessions table
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_user_id ON ekyc_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_status ON ekyc_sessions(status);
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_reference ON ekyc_sessions(session_reference);
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_created_at ON ekyc_sessions(created_at);

-- Indexes for ekyc_assets table
CREATE INDEX IF NOT EXISTS idx_ekyc_assets_session_id ON ekyc_assets(session_id);
CREATE INDEX IF NOT EXISTS idx_ekyc_assets_type ON ekyc_assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_ekyc_assets_processed ON ekyc_assets(processed);

-- Indexes for ekyc_verifications table
CREATE INDEX IF NOT EXISTS idx_ekyc_verifications_session_id ON ekyc_verifications(session_id);
CREATE INDEX IF NOT EXISTS idx_ekyc_verifications_verification_id ON ekyc_verifications(verification_id);

-- Indexes for training schema
CREATE INDEX IF NOT EXISTS idx_face_identities_person_id ON training.face_identities(person_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_identity_id ON training.face_embeddings(identity_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_version ON training.face_embeddings(embedding_version);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON training.datasets(dataset_name);
CREATE INDEX IF NOT EXISTS idx_training_sessions_training_id ON training.training_sessions(training_id);
CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training.training_sessions(status);

-- Indexes for audit schema
CREATE INDEX IF NOT EXISTS idx_activity_logs_user_id ON audit.activity_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_action ON audit.activity_logs(action);
CREATE INDEX IF NOT EXISTS idx_activity_logs_timestamp ON audit.activity_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_type ON audit.security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON audit.security_events(timestamp);

-- =====================================================
-- PART 4: CREATE TRIGGERS
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '⚡ Creating database triggers...';
END $$;

-- Function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for timestamp updates
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ekyc_sessions_updated_at 
    BEFORE UPDATE ON ekyc_sessions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_face_identities_updated_at 
    BEFORE UPDATE ON training.face_identities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- PART 5: SET UP PERMISSIONS
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '🔑 Setting up table permissions...';
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

-- =====================================================
-- PART 6: CONFIGURATION AND INITIALIZATION DATA
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '⚙️ Setting up database configuration...';
END $$;

-- Create database configuration table
CREATE TABLE IF NOT EXISTS database_config (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert configuration values
INSERT INTO database_config (config_key, config_value, description) VALUES
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
CREATE TABLE IF NOT EXISTS initialization_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    script_name VARCHAR(100) NOT NULL,
    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'completed',
    details TEXT,
    database_version VARCHAR(20)
);

-- Log this initialization
INSERT INTO initialization_log (script_name, details, database_version) VALUES
('pgadmin_complete_setup.sql', 'Complete database setup via pgAdmin with all tables and configuration', '1.0.0');

-- =====================================================
-- PART 7: SAMPLE DATA (OPTIONAL)
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '📝 Loading sample data for testing...';
END $$;

-- Insert sample users
INSERT INTO users (id, username, email, hashed_password, full_name, phone_number, is_active, is_verified, metadata) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'admin', 'admin@ekyc.local', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu', 'System Administrator', '+84901234567', true, true, '{"role": "admin", "permissions": ["all"]}'),
('550e8400-e29b-41d4-a716-446655440002', 'test_user_1', 'user1@test.com', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu', 'Nguyen Van Test', '+84901234568', true, true, '{"role": "user", "test_account": true}'),
('550e8400-e29b-41d4-a716-446655440003', 'test_user_2', 'user2@test.com', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu', 'Tran Thi Demo', '+84901234569', true, true, '{"role": "user", "test_account": true}')
ON CONFLICT (id) DO NOTHING;

-- Insert sample EKYC session
INSERT INTO ekyc_sessions (id, user_id, session_reference, status, id_card_data, face_match_score, liveness_score, overall_confidence, final_decision, processing_stages, verification_results, created_at, updated_at) VALUES
('660e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440002', 'EKYC-2025-0917-001', 'completed', 
 '{"id_number": "123456789", "full_name": "NGUYEN VAN TEST", "date_of_birth": "1990-01-15", "place_of_birth": "Ha Noi", "address": "123 Test Street, Ha Noi"}',
 0.85, 0.78, 0.82, 'APPROVED',
 '{"id_card_processing": "completed", "face_detection": "completed", "liveness_check": "completed", "final_decision": "completed"}',
 '{"verification_id": "VER-2025-0917-001", "status": "PASSED", "confidence": 0.82}',
 CURRENT_TIMESTAMP - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '1 hour')
ON CONFLICT (id) DO NOTHING;

-- Insert sample face identity
INSERT INTO training.face_identities (id, person_id, full_name, metadata, created_at) VALUES
('990e8400-e29b-41d4-a716-446655440001', 'PERSON_001', 'Nguyen Van Test', '{"verified": true, "source": "ekyc_verification", "documents": ["CCCD123456789"]}', CURRENT_TIMESTAMP - INTERVAL '1 day')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- PART 8: BASIC UTILITY FUNCTIONS
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '🛠️ Creating basic utility functions...';
END $$;

-- Function to calculate face similarity between two embeddings
CREATE OR REPLACE FUNCTION calculate_face_similarity(
    embedding1 FLOAT[],
    embedding2 FLOAT[]
) RETURNS FLOAT AS $$
DECLARE
    dot_product FLOAT := 0;
    norm1 FLOAT := 0;
    norm2 FLOAT := 0;
    i INTEGER;
BEGIN
    -- Check if embeddings have same dimensions
    IF array_length(embedding1, 1) != array_length(embedding2, 1) THEN
        RAISE EXCEPTION 'Embeddings must have the same dimensions';
    END IF;
    
    -- Calculate dot product and norms
    FOR i IN 1..array_length(embedding1, 1) LOOP
        dot_product := dot_product + (embedding1[i] * embedding2[i]);
        norm1 := norm1 + (embedding1[i] * embedding1[i]);
        norm2 := norm2 + (embedding2[i] * embedding2[i]);
    END LOOP;
    
    -- Calculate cosine similarity
    IF norm1 > 0 AND norm2 > 0 THEN
        RETURN dot_product / (sqrt(norm1) * sqrt(norm2));
    ELSE
        RETURN 0;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE(
    schema_name TEXT,
    table_name TEXT,
    row_count BIGINT,
    size_mb NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname::TEXT,
        tablename::TEXT,
        COALESCE(n_tup_ins - n_tup_del, 0) AS row_count,
        ROUND(pg_total_relation_size(schemaname||'.'||tablename)::NUMERIC / 1024 / 1024, 2) AS size_mb
    FROM pg_stat_user_tables 
    WHERE schemaname IN ('ekyc', 'training', 'audit')
    ORDER BY size_mb DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions on utility functions
GRANT EXECUTE ON FUNCTION calculate_face_similarity TO ekyc_user;
GRANT EXECUTE ON FUNCTION get_database_stats TO ekyc_admin, ekyc_user;

-- =====================================================
-- PART 9: FINAL VALIDATION AND COMPLETION
-- =====================================================

DO $$
DECLARE
    table_count INTEGER;
    view_count INTEGER;
    function_count INTEGER;
    user_count INTEGER;
    config_count INTEGER;
BEGIN
    -- Count tables
    SELECT COUNT(*) INTO table_count 
    FROM information_schema.tables 
    WHERE table_schema IN ('ekyc', 'training', 'audit');
    
    -- Count views  
    SELECT COUNT(*) INTO view_count 
    FROM information_schema.views 
    WHERE table_schema IN ('ekyc', 'training', 'audit');
    
    -- Count functions
    SELECT COUNT(*) INTO function_count 
    FROM information_schema.routines 
    WHERE routine_schema IN ('ekyc', 'training', 'audit');
    
    -- Count sample users
    SELECT COUNT(*) INTO user_count FROM users;
    
    -- Count configuration entries
    SELECT COUNT(*) INTO config_count FROM database_config;
    
    RAISE NOTICE '====================================================';
    RAISE NOTICE '🎉 GAEA EKYC v2 Database Setup Complete!';
    RAISE NOTICE '====================================================';
    RAISE NOTICE '📊 Database Statistics:';
    RAISE NOTICE '   • Tables Created: %', table_count;
    RAISE NOTICE '   • Views Created: %', view_count;
    RAISE NOTICE '   • Functions Created: %', function_count;
    RAISE NOTICE '   • Sample Users: %', user_count;
    RAISE NOTICE '   • Configuration Entries: %', config_count;
    RAISE NOTICE '';
    RAISE NOTICE '🔐 Database Roles Created:';
    RAISE NOTICE '   • ekyc_admin (Password: ekyc_admin_pass_2025)';
    RAISE NOTICE '   • ekyc_user (Password: ekyc_user_pass_2025)'; 
    RAISE NOTICE '   • ekyc_readonly (Password: ekyc_readonly_pass_2025)';
    RAISE NOTICE '';
    RAISE NOTICE '🗂️ Database Schemas:';
    RAISE NOTICE '   • ekyc (% tables) - main EKYC operations', 
        (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'ekyc');
    RAISE NOTICE '   • training (% tables) - ML training data',
        (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'training');
    RAISE NOTICE '   • audit (% tables) - logging and monitoring',
        (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'audit');
    RAISE NOTICE '';
    RAISE NOTICE '🔑 Test Accounts Created:';
    RAISE NOTICE '   • admin@ekyc.local (admin user)';
    RAISE NOTICE '   • user1@test.com (test user 1)';
    RAISE NOTICE '   • user2@test.com (test user 2)';
    RAISE NOTICE '   • Password for all test accounts: test123';
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  SECURITY REMINDERS:';
    RAISE NOTICE '   • Change all default passwords in production!';
    RAISE NOTICE '   • Update database configuration as needed';
    RAISE NOTICE '   • Review and adjust permissions for your environment';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 Database is ready for EKYC operations!';
    RAISE NOTICE '   • Connection URL: postgresql://ekyc_user:ekyc_user_pass_2025@your_host:5432/your_database';
    RAISE NOTICE '   • For admin operations use: ekyc_admin role';
    RAISE NOTICE '   • For application use: ekyc_user role';
    RAISE NOTICE '   • For read-only access use: ekyc_readonly role';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Update your .env file with the connection details';
    RAISE NOTICE '2. Test the connection from your application';
    RAISE NOTICE '3. Run your application tests';
    RAISE NOTICE '4. Change default passwords for production use';
    RAISE NOTICE '====================================================';
    
    -- Validation checks
    IF table_count < 10 THEN
        RAISE WARNING 'Expected at least 10 tables, only found %', table_count;
    END IF;
    
    IF user_count = 0 THEN
        RAISE WARNING 'No sample users found - sample data may not have loaded correctly';
    ELSE
        RAISE NOTICE '✅ Sample data loaded successfully with % users', user_count;
    END IF;
    
    IF config_count = 0 THEN
        RAISE WARNING 'No configuration entries found - configuration may not have loaded correctly';
    ELSE
        RAISE NOTICE '✅ Configuration loaded successfully with % entries', config_count;
    END IF;
END $$;
