-- =====================================================
-- GAEA EKYC v2 Database Management Scripts
-- Description: Utility functions and maintenance scripts
-- =====================================================

-- Set search path
SET search_path TO ekyc, training, audit, public;

-- =====================================================
-- UTILITY FUNCTIONS
-- =====================================================

-- Function to clean up old sessions (older than specified days)
CREATE OR REPLACE FUNCTION cleanup_old_sessions(days_to_keep INTEGER DEFAULT 30)
RETURNS TABLE(
    deleted_sessions INTEGER,
    deleted_assets INTEGER,
    deleted_verifications INTEGER,
    cleaned_storage_mb NUMERIC
) AS $$
DECLARE
    cutoff_date TIMESTAMP;
    session_ids UUID[];
    storage_size BIGINT := 0;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    -- Get session IDs to delete
    SELECT ARRAY_AGG(id) INTO session_ids 
    FROM ekyc_sessions 
    WHERE created_at < cutoff_date;
    
    -- Calculate storage size of assets to be deleted
    SELECT COALESCE(SUM(file_size), 0) INTO storage_size
    FROM ekyc_assets 
    WHERE session_id = ANY(session_ids);
    
    -- Delete verifications first (foreign key constraint)
    DELETE FROM ekyc_verifications WHERE session_id = ANY(session_ids);
    GET DIAGNOSTICS deleted_verifications = ROW_COUNT;
    
    -- Delete assets
    DELETE FROM ekyc_assets WHERE session_id = ANY(session_ids);
    GET DIAGNOSTICS deleted_assets = ROW_COUNT;
    
    -- Delete sessions
    DELETE FROM ekyc_sessions WHERE id = ANY(session_ids);
    GET DIAGNOSTICS deleted_sessions = ROW_COUNT;
    
    cleaned_storage_mb := ROUND(storage_size / 1024.0 / 1024.0, 2);
    
    RETURN QUERY SELECT deleted_sessions, deleted_assets, deleted_verifications, cleaned_storage_mb;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE(
    schema_name TEXT,
    table_name TEXT,
    row_count BIGINT,
    size_mb NUMERIC,
    last_analyzed TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname::TEXT,
        tablename::TEXT,
        COALESCE(n_tup_ins - n_tup_del, 0) AS row_count,
        ROUND(pg_total_relation_size(schemaname||'.'||tablename)::NUMERIC / 1024 / 1024, 2) AS size_mb,
        last_analyze
    FROM pg_stat_user_tables 
    WHERE schemaname IN ('ekyc', 'training', 'audit')
    ORDER BY size_mb DESC;
END;
$$ LANGUAGE plpgsql;

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

-- Function to find similar faces in database
CREATE OR REPLACE FUNCTION find_similar_faces(
    query_embedding FLOAT[],
    similarity_threshold FLOAT DEFAULT 0.6,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    identity_id UUID,
    person_id TEXT,
    full_name TEXT,
    similarity_score FLOAT,
    embedding_id UUID
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fi.id,
        fi.person_id,
        fi.full_name,
        calculate_face_similarity(query_embedding, fe.embedding) AS similarity,
        fe.id
    FROM training.face_identities fi
    JOIN training.face_embeddings fe ON fi.id = fe.identity_id
    WHERE calculate_face_similarity(query_embedding, fe.embedding) >= similarity_threshold
    ORDER BY similarity DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to archive completed sessions
CREATE OR REPLACE FUNCTION archive_completed_sessions(days_old INTEGER DEFAULT 7)
RETURNS TABLE(
    archived_sessions INTEGER,
    archive_table_name TEXT
) AS $$
DECLARE
    archive_table TEXT;
    cutoff_date TIMESTAMP;
    session_count INTEGER;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - INTERVAL '1 day' * days_old;
    archive_table := 'ekyc_sessions_archive_' || to_char(CURRENT_DATE, 'YYYY_MM');
    
    -- Create archive table if not exists
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I (
            LIKE ekyc_sessions INCLUDING ALL
        )', archive_table);
    
    -- Move completed sessions to archive
    EXECUTE format('
        INSERT INTO %I 
        SELECT * FROM ekyc_sessions 
        WHERE status = ''completed'' 
        AND updated_at < $1
    ', archive_table) USING cutoff_date;
    
    GET DIAGNOSTICS session_count = ROW_COUNT;
    
    -- Delete archived sessions from main table
    DELETE FROM ekyc_sessions 
    WHERE status = 'completed' 
    AND updated_at < cutoff_date;
    
    RETURN QUERY SELECT session_count, archive_table;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- MONITORING VIEWS
-- =====================================================

-- View for session processing performance
CREATE OR REPLACE VIEW session_performance_stats AS
SELECT 
    DATE(created_at) as processing_date,
    COUNT(*) as total_sessions,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_sessions,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_sessions,
    COUNT(*) FILTER (WHERE status = 'processing') as processing_sessions,
    ROUND(AVG(EXTRACT(EPOCH FROM (updated_at - created_at))), 2) as avg_processing_time_seconds,
    ROUND(AVG(overall_confidence), 3) as avg_confidence,
    COUNT(*) FILTER (WHERE final_decision = 'APPROVED') as approved_count,
    COUNT(*) FILTER (WHERE final_decision = 'REJECTED') as rejected_count
FROM ekyc_sessions 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY processing_date DESC;

-- View for training model performance
CREATE OR REPLACE VIEW training_performance_stats AS
SELECT 
    t.training_id,
    t.status,
    t.start_time,
    t.duration_seconds,
    d.dataset_name,
    d.total_images,
    d.total_persons,
    (t.metrics->>'accuracy')::FLOAT as accuracy,
    (t.metrics->>'precision')::FLOAT as precision,
    (t.metrics->>'recall')::FLOAT as recall,
    (t.metrics->>'f1_score')::FLOAT as f1_score,
    (t.model_info->>'optimal_threshold')::FLOAT as optimal_threshold
FROM training.training_sessions t
JOIN training.datasets d ON t.dataset_id = d.id
ORDER BY t.start_time DESC;

-- View for asset storage statistics
CREATE OR REPLACE VIEW asset_storage_stats AS
SELECT 
    asset_type,
    COUNT(*) as total_files,
    ROUND(SUM(file_size) / 1024.0 / 1024.0, 2) as total_size_mb,
    ROUND(AVG(file_size) / 1024.0, 2) as avg_size_kb,
    ROUND(AVG(quality_score), 3) as avg_quality_score,
    ROUND(AVG(confidence_score), 3) as avg_confidence_score
FROM ekyc_assets 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY asset_type
ORDER BY total_size_mb DESC;

-- View for user feedback analysis
CREATE OR REPLACE VIEW user_feedback_analysis AS
SELECT 
    DATE(uf.created_at) as feedback_date,
    COUNT(*) as total_feedback,
    COUNT(*) FILTER (WHERE uf.actual_result = uf.system_result) as correct_predictions,
    COUNT(*) FILTER (WHERE uf.actual_result != uf.system_result) as incorrect_predictions,
    ROUND(
        COUNT(*) FILTER (WHERE uf.actual_result = uf.system_result)::FLOAT / COUNT(*)::FLOAT * 100, 
        2
    ) as accuracy_percentage,
    ROUND(AVG(uf.confidence_score), 3) as avg_confidence_score,
    COUNT(*) FILTER (WHERE uf.feedback_type = 'false_positive') as false_positives,
    COUNT(*) FILTER (WHERE uf.feedback_type = 'false_negative') as false_negatives
FROM training.user_feedback uf
WHERE uf.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(uf.created_at)
ORDER BY feedback_date DESC;

-- View for system health monitoring
CREATE OR REPLACE VIEW system_health_monitor AS
SELECT 
    'Sessions Today' as metric,
    COUNT(*)::TEXT as value,
    'sessions' as unit,
    CASE WHEN COUNT(*) > 100 THEN 'healthy' ELSE 'warning' END as status
FROM ekyc_sessions 
WHERE DATE(created_at) = CURRENT_DATE

UNION ALL

SELECT 
    'Average Processing Time' as metric,
    ROUND(AVG(EXTRACT(EPOCH FROM (updated_at - created_at))))::TEXT as value,
    'seconds' as unit,
    CASE 
        WHEN AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) < 60 THEN 'healthy'
        WHEN AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) < 120 THEN 'warning'
        ELSE 'critical'
    END as status
FROM ekyc_sessions 
WHERE status = 'completed' 
AND DATE(created_at) = CURRENT_DATE

UNION ALL

SELECT 
    'Success Rate Today' as metric,
    ROUND(
        COUNT(*) FILTER (WHERE final_decision = 'APPROVED')::FLOAT / 
        COUNT(*)::FLOAT * 100, 1
    )::TEXT as value,
    '%' as unit,
    CASE 
        WHEN ROUND(COUNT(*) FILTER (WHERE final_decision = 'APPROVED')::FLOAT / COUNT(*)::FLOAT * 100, 1) > 80 THEN 'healthy'
        WHEN ROUND(COUNT(*) FILTER (WHERE final_decision = 'APPROVED')::FLOAT / COUNT(*)::FLOAT * 100, 1) > 60 THEN 'warning'
        ELSE 'critical'
    END as status
FROM ekyc_sessions 
WHERE status = 'completed' 
AND DATE(created_at) = CURRENT_DATE

UNION ALL

SELECT 
    'Storage Used' as metric,
    ROUND(SUM(file_size) / 1024.0 / 1024.0 / 1024.0, 2)::TEXT as value,
    'GB' as unit,
    CASE 
        WHEN SUM(file_size) / 1024.0 / 1024.0 / 1024.0 < 10 THEN 'healthy'
        WHEN SUM(file_size) / 1024.0 / 1024.0 / 1024.0 < 50 THEN 'warning'
        ELSE 'critical'
    END as status
FROM ekyc_assets;

-- =====================================================
-- MAINTENANCE PROCEDURES
-- =====================================================

-- Procedure to rebuild indexes
CREATE OR REPLACE FUNCTION rebuild_all_indexes()
RETURNS TABLE(schema_name TEXT, index_name TEXT, status TEXT) AS $$
DECLARE
    index_record RECORD;
BEGIN
    FOR index_record IN 
        SELECT schemaname, indexname 
        FROM pg_indexes 
        WHERE schemaname IN ('ekyc', 'training', 'audit')
    LOOP
        BEGIN
            EXECUTE format('REINDEX INDEX %I.%I', index_record.schemaname, index_record.indexname);
            RETURN QUERY SELECT index_record.schemaname, index_record.indexname, 'SUCCESS';
        EXCEPTION WHEN others THEN
            RETURN QUERY SELECT index_record.schemaname, index_record.indexname, 'FAILED: ' || SQLERRM;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Procedure to update table statistics
CREATE OR REPLACE FUNCTION update_table_statistics()
RETURNS TABLE(schema_name TEXT, table_name TEXT, status TEXT) AS $$
DECLARE
    table_record RECORD;
BEGIN
    FOR table_record IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE schemaname IN ('ekyc', 'training', 'audit')
    LOOP
        BEGIN
            EXECUTE format('ANALYZE %I.%I', table_record.schemaname, table_record.tablename);
            RETURN QUERY SELECT table_record.schemaname, table_record.tablename, 'ANALYZED';
        EXCEPTION WHEN others THEN
            RETURN QUERY SELECT table_record.schemaname, table_record.tablename, 'FAILED: ' || SQLERRM;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- BACKUP AND RESTORE UTILITIES
-- =====================================================

-- Function to create a data export for a specific user
CREATE OR REPLACE FUNCTION export_user_data(user_uuid UUID)
RETURNS TABLE(
    table_name TEXT,
    record_count BIGINT,
    export_status TEXT
) AS $$
DECLARE
    export_timestamp TEXT;
    table_record RECORD;
    query TEXT;
    record_count BIGINT;
BEGIN
    export_timestamp := to_char(CURRENT_TIMESTAMP, 'YYYY_MM_DD_HH24_MI_SS');
    
    -- Export user sessions
    query := format('COPY (SELECT * FROM ekyc_sessions WHERE user_id = ''%s'') TO ''/tmp/user_sessions_%s_%s.csv'' CSV HEADER', 
                   user_uuid, user_uuid, export_timestamp);
    EXECUTE query;
    
    SELECT COUNT(*) INTO record_count FROM ekyc_sessions WHERE user_id = user_uuid;
    RETURN QUERY SELECT 'ekyc_sessions'::TEXT, record_count, 'EXPORTED'::TEXT;
    
    -- Export user activity logs
    query := format('COPY (SELECT * FROM audit.activity_logs WHERE user_id = ''%s'') TO ''/tmp/user_activity_%s_%s.csv'' CSV HEADER', 
                   user_uuid, user_uuid, export_timestamp);
    EXECUTE query;
    
    SELECT COUNT(*) INTO record_count FROM audit.activity_logs WHERE user_id = user_uuid;
    RETURN QUERY SELECT 'activity_logs'::TEXT, record_count, 'EXPORTED'::TEXT;
    
EXCEPTION WHEN others THEN
    RETURN QUERY SELECT 'export_error'::TEXT, 0::BIGINT, ('FAILED: ' || SQLERRM)::TEXT;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PERFORMANCE OPTIMIZATION
-- =====================================================

-- Function to identify slow queries and suggest optimizations
CREATE OR REPLACE FUNCTION analyze_query_performance()
RETURNS TABLE(
    query_text TEXT,
    total_calls BIGINT,
    mean_time_ms NUMERIC,
    suggestion TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        LEFT(query, 100) as query_text,
        calls as total_calls,
        ROUND(mean_time, 2) as mean_time_ms,
        CASE 
            WHEN mean_time > 1000 THEN 'Consider adding indexes or optimizing WHERE clauses'
            WHEN calls > 10000 THEN 'High frequency query - consider caching'
            WHEN mean_time > 500 THEN 'Moderate optimization needed'
            ELSE 'Query performance acceptable'
        END as suggestion
    FROM pg_stat_statements
    WHERE query LIKE '%ekyc%' OR query LIKE '%training%' OR query LIKE '%audit%'
    ORDER BY mean_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SECURITY FUNCTIONS
-- =====================================================

-- Function to audit potentially suspicious activities
CREATE OR REPLACE FUNCTION audit_suspicious_activities(days_back INTEGER DEFAULT 7)
RETURNS TABLE(
    activity_type TEXT,
    user_id UUID,
    session_count BIGINT,
    risk_level TEXT,
    details TEXT
) AS $$
BEGIN
    -- Multiple failed sessions from same user
    RETURN QUERY
    SELECT 
        'multiple_failures'::TEXT,
        es.user_id,
        COUNT(*) as session_count,
        CASE 
            WHEN COUNT(*) > 10 THEN 'HIGH'
            WHEN COUNT(*) > 5 THEN 'MEDIUM'
            ELSE 'LOW'
        END as risk_level,
        'User has ' || COUNT(*) || ' failed sessions in last ' || days_back || ' days' as details
    FROM ekyc_sessions es
    WHERE es.final_decision = 'REJECTED'
    AND es.created_at >= CURRENT_DATE - INTERVAL '1 day' * days_back
    GROUP BY es.user_id
    HAVING COUNT(*) > 3;
    
    -- Unusual processing patterns
    RETURN QUERY
    SELECT 
        'unusual_timing'::TEXT,
        es.user_id,
        COUNT(*) as session_count,
        'MEDIUM'::TEXT as risk_level,
        'User active during unusual hours (midnight to 6am)' as details
    FROM ekyc_sessions es
    WHERE EXTRACT(HOUR FROM es.created_at) BETWEEN 0 AND 6
    AND es.created_at >= CURRENT_DATE - INTERVAL '1 day' * days_back
    GROUP BY es.user_id
    HAVING COUNT(*) > 2;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- GRANT PERMISSIONS TO UTILITY FUNCTIONS
-- =====================================================

-- Grant execute permissions on utility functions
GRANT EXECUTE ON FUNCTION cleanup_old_sessions TO ekyc_admin;
GRANT EXECUTE ON FUNCTION get_database_stats TO ekyc_admin, ekyc_user;
GRANT EXECUTE ON FUNCTION calculate_face_similarity TO ekyc_user;
GRANT EXECUTE ON FUNCTION find_similar_faces TO ekyc_user;
GRANT EXECUTE ON FUNCTION archive_completed_sessions TO ekyc_admin;
GRANT EXECUTE ON FUNCTION rebuild_all_indexes TO ekyc_admin;
GRANT EXECUTE ON FUNCTION update_table_statistics TO ekyc_admin;
GRANT EXECUTE ON FUNCTION export_user_data TO ekyc_admin;
GRANT EXECUTE ON FUNCTION analyze_query_performance TO ekyc_admin;
GRANT EXECUTE ON FUNCTION audit_suspicious_activities TO ekyc_admin;

-- Grant select permissions on monitoring views
GRANT SELECT ON session_performance_stats TO ekyc_admin, ekyc_user;
GRANT SELECT ON training_performance_stats TO ekyc_admin, ekyc_user;
GRANT SELECT ON asset_storage_stats TO ekyc_admin, ekyc_user;
GRANT SELECT ON user_feedback_analysis TO ekyc_admin, ekyc_user;
GRANT SELECT ON system_health_monitor TO ekyc_admin, ekyc_user;

-- =====================================================
-- COMPLETION MESSAGE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '🛠️ Database management utilities created successfully!';
    RAISE NOTICE '📊 Available monitoring views:';
    RAISE NOTICE '  - session_performance_stats';
    RAISE NOTICE '  - training_performance_stats';
    RAISE NOTICE '  - asset_storage_stats';
    RAISE NOTICE '  - user_feedback_analysis';
    RAISE NOTICE '  - system_health_monitor';
    RAISE NOTICE '🔧 Available utility functions:';
    RAISE NOTICE '  - cleanup_old_sessions(days)';
    RAISE NOTICE '  - get_database_stats()';
    RAISE NOTICE '  - find_similar_faces(embedding, threshold, limit)';
    RAISE NOTICE '  - archive_completed_sessions(days)';
    RAISE NOTICE '  - audit_suspicious_activities(days)';
    RAISE NOTICE '⚡ Performance and maintenance tools ready!';
END $$;
