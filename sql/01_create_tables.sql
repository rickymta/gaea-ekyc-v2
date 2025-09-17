-- =====================================================
-- GAEA EKYC v2 PostgreSQL Database Schema
-- Created: September 17, 2025
-- Description: Complete database schema for EKYC system
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS ekyc;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set default schema
SET search_path TO ekyc, public;

-- =====================================================
-- CORE EKYC TABLES
-- =====================================================

-- Users table (if not using external authentication)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    phone_number VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- EKYC Sessions table
CREATE TABLE IF NOT EXISTS ekyc_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_reference VARCHAR(100) UNIQUE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    id_card_data JSONB,
    face_match_score FLOAT,
    liveness_score FLOAT,
    overall_confidence FLOAT,
    final_decision VARCHAR(50),
    error_message TEXT,
    processing_stages JSONB DEFAULT '{}'::jsonb,
    verification_results JSONB DEFAULT '{}'::jsonb,
    webhook_status VARCHAR(50) DEFAULT 'pending',
    webhook_attempts INTEGER DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- EKYC Assets table
CREATE TABLE IF NOT EXISTS ekyc_assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES ekyc_sessions(id) ON DELETE CASCADE,
    asset_type VARCHAR(50) NOT NULL, -- 'id_front', 'id_back', 'selfie', 'document'
    file_path VARCHAR(500) NOT NULL,
    original_filename VARCHAR(255),
    file_size BIGINT,
    mime_type VARCHAR(100),
    file_hash VARCHAR(64), -- SHA-256 hash for integrity
    processed BOOLEAN DEFAULT false,
    processing_result JSONB,
    face_embedding FLOAT[] -- Store face embeddings for quick comparison
    quality_score FLOAT,
    confidence_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- EKYC Verifications table (detailed verification results)
CREATE TABLE IF NOT EXISTS ekyc_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES ekyc_sessions(id) ON DELETE CASCADE,
    verification_id VARCHAR(100) UNIQUE NOT NULL,
    verification_type VARCHAR(50) NOT NULL, -- 'complete', 'face_only', 'liveness_only'
    id_card_analysis JSONB,
    selfie_analysis JSONB,
    face_match_result JSONB,
    liveness_result JSONB,
    overall_result JSONB,
    confidence_scores JSONB,
    recommendations TEXT[],
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- FACE RECOGNITION & TRAINING TABLES
-- =====================================================

-- Face Identities table (for storing known faces)
CREATE TABLE IF NOT EXISTS training.face_identities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    metadata JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Face Embeddings table (store face embeddings for recognition)
CREATE TABLE IF NOT EXISTS training.face_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identity_id UUID NOT NULL REFERENCES training.face_identities(id) ON DELETE CASCADE,
    embedding FLOAT[] NOT NULL, -- 512-dimensional embedding vector
    embedding_version VARCHAR(50) NOT NULL, -- model version used
    source_type VARCHAR(50) NOT NULL, -- 'training', 'verification', 'manual'
    source_file_path VARCHAR(500),
    quality_score FLOAT,
    confidence_score FLOAT,
    face_bbox FLOAT[4], -- [x1, y1, x2, y2]
    landmarks FLOAT[][2], -- facial landmarks
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Training Datasets table
CREATE TABLE IF NOT EXISTS training.datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(100) NOT NULL,
    description TEXT,
    dataset_type VARCHAR(50) NOT NULL, -- 'training', 'validation', 'test'
    file_path VARCHAR(500) NOT NULL,
    total_images INTEGER,
    total_persons INTEGER,
    statistics JSONB DEFAULT '{}'::jsonb,
    quality_metrics JSONB DEFAULT '{}'::jsonb,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Training Sessions table
CREATE TABLE IF NOT EXISTS training.training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_id VARCHAR(100) UNIQUE NOT NULL,
    dataset_id UUID REFERENCES training.datasets(id),
    training_config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    metrics JSONB,
    model_info JSONB,
    threshold_optimization JSONB,
    error_message TEXT,
    created_by VARCHAR(255),
    model_output_path VARCHAR(500)
);

-- Model Evaluations table
CREATE TABLE IF NOT EXISTS training.model_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_id VARCHAR(100) UNIQUE NOT NULL,
    training_session_id UUID REFERENCES training.training_sessions(id),
    test_dataset_id UUID REFERENCES training.datasets(id),
    model_config JSONB NOT NULL,
    evaluation_metrics JSONB NOT NULL,
    confusion_matrix JSONB,
    error_analysis JSONB,
    recommendations TEXT[],
    evaluation_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

-- Model Performance History table
CREATE TABLE IF NOT EXISTS training.performance_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    threshold_used FLOAT,
    test_dataset VARCHAR(100),
    measurement_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- User Feedback table (for continuous learning)
CREATE TABLE IF NOT EXISTS training.user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_id VARCHAR(100) UNIQUE NOT NULL,
    verification_id VARCHAR(100) REFERENCES ekyc_verifications(verification_id),
    user_id VARCHAR(255),
    actual_result BOOLEAN NOT NULL,
    system_result BOOLEAN NOT NULL,
    confidence_score FLOAT,
    feedback_type VARCHAR(50) DEFAULT 'accuracy', -- 'accuracy', 'quality', 'user_experience'
    user_comments TEXT,
    is_processed BOOLEAN DEFAULT false,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- SYSTEM MANAGEMENT TABLES
-- =====================================================

-- Webhooks table
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES ekyc_sessions(id),
    webhook_url VARCHAR(500) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'delivered', 'failed'
    response_code INTEGER,
    response_body TEXT,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP WITH TIME ZONE
);

-- API Keys table (for external integrations)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(100) NOT NULL,
    api_key_hash VARCHAR(255) NOT NULL UNIQUE,
    permissions TEXT[] DEFAULT ARRAY['read'],
    rate_limit_per_minute INTEGER DEFAULT 60,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System Configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_encrypted BOOLEAN DEFAULT false,
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Task Queue table (for Celery task tracking)
CREATE TABLE IF NOT EXISTS task_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    task_name VARCHAR(100) NOT NULL,
    task_args JSONB,
    task_kwargs JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    result JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    queue_name VARCHAR(100) DEFAULT 'default',
    priority INTEGER DEFAULT 5,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- AUDIT TABLES
-- =====================================================

-- Audit Log table
CREATE TABLE IF NOT EXISTS audit.activity_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    session_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    details JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Security Events table
CREATE TABLE IF NOT EXISTS audit.security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL, -- 'login_failure', 'suspicious_activity', 'rate_limit_exceeded'
    severity VARCHAR(20) NOT NULL DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    user_id VARCHAR(255),
    ip_address INET,
    details JSONB DEFAULT '{}'::jsonb,
    is_resolved BOOLEAN DEFAULT false,
    resolved_by VARCHAR(255),
    resolved_at TIMESTAMP WITH TIME ZONE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data Retention table
CREATE TABLE IF NOT EXISTS audit.data_retention (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    retention_policy VARCHAR(100) NOT NULL,
    scheduled_deletion TIMESTAMP WITH TIME ZONE NOT NULL,
    is_deleted BOOLEAN DEFAULT false,
    deleted_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- EKYC Sessions indexes
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_user_id ON ekyc_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_status ON ekyc_sessions(status);
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_created_at ON ekyc_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_session_reference ON ekyc_sessions(session_reference);

-- EKYC Assets indexes
CREATE INDEX IF NOT EXISTS idx_ekyc_assets_session_id ON ekyc_assets(session_id);
CREATE INDEX IF NOT EXISTS idx_ekyc_assets_type ON ekyc_assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_ekyc_assets_processed ON ekyc_assets(processed);

-- EKYC Verifications indexes
CREATE INDEX IF NOT EXISTS idx_ekyc_verifications_session_id ON ekyc_verifications(session_id);
CREATE INDEX IF NOT EXISTS idx_ekyc_verifications_verification_id ON ekyc_verifications(verification_id);
CREATE INDEX IF NOT EXISTS idx_ekyc_verifications_type ON ekyc_verifications(verification_type);

-- Face Embeddings indexes
CREATE INDEX IF NOT EXISTS idx_face_embeddings_identity_id ON training.face_embeddings(identity_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_version ON training.face_embeddings(embedding_version);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_source_type ON training.face_embeddings(source_type);

-- Training Sessions indexes
CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training.training_sessions(status);
CREATE INDEX IF NOT EXISTS idx_training_sessions_dataset_id ON training.training_sessions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_training_sessions_start_time ON training.training_sessions(start_time);

-- User Feedback indexes
CREATE INDEX IF NOT EXISTS idx_user_feedback_verification_id ON training.user_feedback(verification_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_processed ON training.user_feedback(is_processed);
CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON training.user_feedback(created_at);

-- Webhook Deliveries indexes
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_session_id ON webhook_deliveries(session_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status ON webhook_deliveries(status);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_next_retry ON webhook_deliveries(next_retry_at) WHERE status = 'failed';

-- Task Queue indexes
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_queue_task_name ON task_queue(task_name);
CREATE INDEX IF NOT EXISTS idx_task_queue_priority ON task_queue(priority);
CREATE INDEX IF NOT EXISTS idx_task_queue_created_at ON task_queue(created_at);

-- Audit indexes
CREATE INDEX IF NOT EXISTS idx_activity_logs_user_id ON audit.activity_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_action ON audit.activity_logs(action);
CREATE INDEX IF NOT EXISTS idx_activity_logs_timestamp ON audit.activity_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON audit.security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON audit.security_events(timestamp);

-- GIN indexes for JSONB columns (for fast JSON queries)
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_processing_stages ON ekyc_sessions USING GIN(processing_stages);
CREATE INDEX IF NOT EXISTS idx_ekyc_sessions_verification_results ON ekyc_sessions USING GIN(verification_results);
CREATE INDEX IF NOT EXISTS idx_ekyc_assets_processing_result ON ekyc_assets USING GIN(processing_result);
CREATE INDEX IF NOT EXISTS idx_system_config_value ON system_config USING GIN(config_value);

-- =====================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMPS
-- =====================================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ekyc_sessions_updated_at 
    BEFORE UPDATE ON ekyc_sessions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_face_identities_updated_at 
    BEFORE UPDATE ON training.face_identities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at 
    BEFORE UPDATE ON system_config 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- =====================================================

-- Function to get session statistics
CREATE OR REPLACE FUNCTION get_session_statistics(
    start_date TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    end_date TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS TABLE(
    total_sessions BIGINT,
    successful_sessions BIGINT,
    failed_sessions BIGINT,
    pending_sessions BIGINT,
    avg_processing_time INTERVAL,
    success_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_sessions,
        COUNT(*) FILTER (WHERE final_decision = 'APPROVED') as successful_sessions,
        COUNT(*) FILTER (WHERE final_decision = 'REJECTED') as failed_sessions,
        COUNT(*) FILTER (WHERE status = 'pending') as pending_sessions,
        AVG(completed_at - created_at) as avg_processing_time,
        ROUND(
            (COUNT(*) FILTER (WHERE final_decision = 'APPROVED') * 100.0 / NULLIF(COUNT(*), 0))::NUMERIC, 
            2
        ) as success_rate
    FROM ekyc_sessions
    WHERE (start_date IS NULL OR created_at >= start_date)
      AND (end_date IS NULL OR created_at <= end_date);
END;
$$ LANGUAGE plpgsql;

-- Function to clean old completed sessions
CREATE OR REPLACE FUNCTION cleanup_old_sessions(
    retention_days INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ekyc_sessions 
    WHERE status IN ('completed', 'expired') 
      AND completed_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * retention_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get face embedding similarity
CREATE OR REPLACE FUNCTION calculate_embedding_similarity(
    embedding1 FLOAT[],
    embedding2 FLOAT[]
)
RETURNS FLOAT AS $$
DECLARE
    dot_product FLOAT := 0;
    norm1 FLOAT := 0;
    norm2 FLOAT := 0;
    i INTEGER;
BEGIN
    -- Check if embeddings have same dimension
    IF array_length(embedding1, 1) != array_length(embedding2, 1) THEN
        RETURN 0;
    END IF;
    
    -- Calculate dot product and norms
    FOR i IN 1..array_length(embedding1, 1) LOOP
        dot_product := dot_product + (embedding1[i] * embedding2[i]);
        norm1 := norm1 + (embedding1[i] * embedding1[i]);
        norm2 := norm2 + (embedding2[i] * embedding2[i]);
    END LOOP;
    
    -- Calculate cosine similarity
    IF norm1 = 0 OR norm2 = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN dot_product / (sqrt(norm1) * sqrt(norm2));
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for session summary with asset counts
CREATE OR REPLACE VIEW session_summary AS
SELECT 
    s.id,
    s.user_id,
    s.session_reference,
    s.status,
    s.final_decision,
    s.face_match_score,
    s.liveness_score,
    s.overall_confidence,
    s.created_at,
    s.updated_at,
    s.completed_at,
    COUNT(a.id) as total_assets,
    COUNT(a.id) FILTER (WHERE a.processed = true) as processed_assets,
    COUNT(a.id) FILTER (WHERE a.asset_type = 'id_front') as id_front_count,
    COUNT(a.id) FILTER (WHERE a.asset_type = 'id_back') as id_back_count,
    COUNT(a.id) FILTER (WHERE a.asset_type = 'selfie') as selfie_count
FROM ekyc_sessions s
LEFT JOIN ekyc_assets a ON s.id = a.session_id
GROUP BY s.id, s.user_id, s.session_reference, s.status, s.final_decision, 
         s.face_match_score, s.liveness_score, s.overall_confidence, 
         s.created_at, s.updated_at, s.completed_at;

-- View for training progress
CREATE OR REPLACE VIEW training_progress AS
SELECT 
    ts.id,
    ts.training_id,
    ts.status,
    ts.start_time,
    ts.end_time,
    ts.duration_seconds,
    d.dataset_name,
    d.total_images,
    d.total_persons,
    ts.metrics->>'accuracy' as accuracy,
    ts.metrics->>'precision' as precision,
    ts.metrics->>'recall' as recall,
    ts.metrics->>'f1_score' as f1_score
FROM training.training_sessions ts
LEFT JOIN training.datasets d ON ts.dataset_id = d.id
ORDER BY ts.start_time DESC;

-- View for webhook delivery status
CREATE OR REPLACE VIEW webhook_status AS
SELECT 
    wd.id,
    wd.session_id,
    s.session_reference,
    wd.event_type,
    wd.status,
    wd.attempts,
    wd.max_attempts,
    wd.response_code,
    wd.created_at,
    wd.delivered_at,
    wd.next_retry_at
FROM webhook_deliveries wd
LEFT JOIN ekyc_sessions s ON wd.session_id = s.id
ORDER BY wd.created_at DESC;

-- =====================================================
-- INITIAL DATA INSERTS
-- =====================================================

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('face_match_threshold', '0.6', 'Default threshold for face matching'),
('liveness_threshold', '0.5', 'Default threshold for liveness detection'),
('session_expiry_hours', '24', 'Hours after which sessions expire'),
('max_file_size_mb', '10', 'Maximum file size for uploads in MB'),
('webhook_retry_attempts', '3', 'Maximum webhook delivery retry attempts'),
('model_version', '1.0.0', 'Current face recognition model version')
ON CONFLICT (config_key) DO NOTHING;

-- =====================================================
-- GRANTS AND PERMISSIONS
-- =====================================================

-- Grant permissions to application user (adjust username as needed)
-- GRANT USAGE ON SCHEMA ekyc TO ekyc_user;
-- GRANT USAGE ON SCHEMA training TO ekyc_user;
-- GRANT USAGE ON SCHEMA audit TO ekyc_user;

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ekyc TO ekyc_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA training TO ekyc_user;
-- GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO ekyc_user;

-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA ekyc TO ekyc_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA training TO ekyc_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA audit TO ekyc_user;

-- =====================================================
-- COMPLETION MESSAGE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'GAEA EKYC v2 Database Schema Created Successfully!';
    RAISE NOTICE 'Schemas: ekyc, training, audit';
    RAISE NOTICE 'Tables: % total tables created', (
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema IN ('ekyc', 'training', 'audit', 'public')
        AND table_name LIKE '%ekyc%' OR table_name IN (
            'users', 'webhook_deliveries', 'api_keys', 'system_config', 
            'task_queue', 'activity_logs', 'security_events', 'data_retention'
        )
    );
    RAISE NOTICE 'Database ready for EKYC operations!';
END $$;
