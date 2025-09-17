-- =====================================================
-- GAEA EKYC v2 Sample Data & Test Records
-- Description: Insert sample data for testing and development
-- =====================================================

-- Set search path
SET search_path TO ekyc, training, audit, public;

-- =====================================================
-- SAMPLE USERS
-- =====================================================

INSERT INTO users (id, username, email, hashed_password, full_name, phone_number, is_active, is_verified, metadata) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'admin', 'admin@ekyc.local', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu', 'System Administrator', '+84901234567', true, true, '{"role": "admin", "permissions": ["all"]}'),
('550e8400-e29b-41d4-a716-446655440002', 'test_user_1', 'user1@test.com', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu', 'Nguyen Van Test', '+84901234568', true, true, '{"role": "user", "test_account": true}'),
('550e8400-e29b-41d4-a716-446655440003', 'test_user_2', 'user2@test.com', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu', 'Tran Thi Demo', '+84901234569', true, true, '{"role": "user", "test_account": true}')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE EKYC SESSIONS
-- =====================================================

INSERT INTO ekyc_sessions (id, user_id, session_reference, status, id_card_data, face_match_score, liveness_score, overall_confidence, final_decision, processing_stages, verification_results, created_at, updated_at) VALUES
-- Successful session
('660e8400-e29b-41d4-a716-446655440001', 'test_user_1', 'EKYC-2025-0917-001', 'completed', 
 '{"id_number": "123456789", "full_name": "NGUYEN VAN TEST", "date_of_birth": "1990-01-15", "place_of_birth": "Ha Noi", "address": "123 Test Street, Ha Noi"}',
 0.85, 0.78, 0.82, 'APPROVED',
 '{"id_card_processing": "completed", "face_detection": "completed", "liveness_check": "completed", "final_decision": "completed"}',
 '{"verification_id": "VER-2025-0917-001", "status": "PASSED", "confidence": 0.82}',
 CURRENT_TIMESTAMP - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '1 hour'),

-- Failed session  
('660e8400-e29b-41d4-a716-446655440002', 'test_user_2', 'EKYC-2025-0917-002', 'completed',
 '{"id_number": "987654321", "full_name": "TRAN THI DEMO", "date_of_birth": "1985-05-20", "place_of_birth": "Ho Chi Minh", "address": "456 Demo Avenue, Ho Chi Minh"}',
 0.45, 0.30, 0.38, 'REJECTED',
 '{"id_card_processing": "completed", "face_detection": "completed", "liveness_check": "failed", "final_decision": "completed"}',
 '{"verification_id": "VER-2025-0917-002", "status": "FAILED", "confidence": 0.38, "reason": "Liveness detection failed"}',
 CURRENT_TIMESTAMP - INTERVAL '3 hours', CURRENT_TIMESTAMP - INTERVAL '2 hours'),

-- Pending session
('660e8400-e29b-41d4-a716-446655440003', 'test_user_1', 'EKYC-2025-0917-003', 'processing',
 NULL, NULL, NULL, NULL, NULL,
 '{"id_card_processing": "completed", "face_detection": "in_progress"}',
 '{}',
 CURRENT_TIMESTAMP - INTERVAL '30 minutes', CURRENT_TIMESTAMP - INTERVAL '30 minutes')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE EKYC ASSETS
-- =====================================================

INSERT INTO ekyc_assets (id, session_id, asset_type, file_path, original_filename, file_size, mime_type, file_hash, processed, processing_result, quality_score, confidence_score, created_at) VALUES
-- Assets for successful session
('770e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440001', 'id_front', '/storage/sessions/660e8400-e29b-41d4-a716-446655440001/id_front.jpg', 'id_card_front.jpg', 2458320, 'image/jpeg', 'a1b2c3d4e5f6789...', true, 
 '{"faces_detected": 1, "text_detected": true, "quality_score": 0.87, "confidence": 0.92}', 0.87, 0.92, CURRENT_TIMESTAMP - INTERVAL '2 hours'),
('770e8400-e29b-41d4-a716-446655440002', '660e8400-e29b-41d4-a716-446655440001', 'id_back', '/storage/sessions/660e8400-e29b-41d4-a716-446655440001/id_back.jpg', 'id_card_back.jpg', 2234567, 'image/jpeg', 'b2c3d4e5f6789a1...', true,
 '{"text_detected": true, "quality_score": 0.82, "mrz_detected": true}', 0.82, 0.88, CURRENT_TIMESTAMP - INTERVAL '2 hours'),
('770e8400-e29b-41d4-a716-446655440003', '660e8400-e29b-41d4-a716-446655440001', 'selfie', '/storage/sessions/660e8400-e29b-41d4-a716-446655440001/selfie.jpg', 'selfie_photo.jpg', 1876543, 'image/jpeg', 'c3d4e5f6789a1b2...', true,
 '{"faces_detected": 1, "liveness_score": 0.78, "quality_score": 0.81, "face_match_score": 0.85}', 0.81, 0.85, CURRENT_TIMESTAMP - INTERVAL '2 hours'),

-- Assets for failed session
('770e8400-e29b-41d4-a716-446655440004', '660e8400-e29b-41d4-a716-446655440002', 'id_front', '/storage/sessions/660e8400-e29b-41d4-a716-446655440002/id_front.jpg', 'blurry_id.jpg', 1234567, 'image/jpeg', 'd4e5f6789a1b2c3...', true,
 '{"faces_detected": 1, "text_detected": true, "quality_score": 0.45, "confidence": 0.52}', 0.45, 0.52, CURRENT_TIMESTAMP - INTERVAL '3 hours'),
('770e8400-e29b-41d4-a716-446655440005', '660e8400-e29b-41d4-a716-446655440002', 'selfie', '/storage/sessions/660e8400-e29b-41d4-a716-446655440002/selfie.jpg', 'fake_selfie.jpg', 987654, 'image/jpeg', 'e5f6789a1b2c3d4...', true,
 '{"faces_detected": 1, "liveness_score": 0.30, "quality_score": 0.65, "face_match_score": 0.45, "liveness_failed": true}', 0.65, 0.45, CURRENT_TIMESTAMP - INTERVAL '3 hours'),

-- Assets for pending session
('770e8400-e29b-41d4-a716-446655440006', '660e8400-e29b-41d4-a716-446655440003', 'id_front', '/storage/sessions/660e8400-e29b-41d4-a716-446655440003/id_front.jpg', 'new_id.jpg', 2100000, 'image/jpeg', 'f6789a1b2c3d4e5...', true,
 '{"faces_detected": 1, "text_detected": true, "quality_score": 0.75, "confidence": 0.80}', 0.75, 0.80, CURRENT_TIMESTAMP - INTERVAL '30 minutes'),
('770e8400-e29b-41d4-a716-446655440007', '660e8400-e29b-41d4-a716-446655440003', 'selfie', '/storage/sessions/660e8400-e29b-41d4-a716-446655440003/selfie.jpg', 'new_selfie.jpg', 1800000, 'image/jpeg', '789a1b2c3d4e5f6...', false,
 '{}', NULL, NULL, CURRENT_TIMESTAMP - INTERVAL '30 minutes')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE VERIFICATIONS
-- =====================================================

INSERT INTO ekyc_verifications (id, session_id, verification_id, verification_type, id_card_analysis, selfie_analysis, face_match_result, liveness_result, overall_result, confidence_scores, recommendations, processing_time_ms, model_version, created_at) VALUES
('880e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440001', 'VER-2025-0917-001', 'complete',
 '{"faces_detected": 1, "primary_face": {"bbox": [120, 80, 200, 180], "confidence": 0.92}, "face_quality": {"overall_score": 0.87}, "embedding_extracted": true}',
 '{"faces_detected": 1, "primary_face": {"bbox": [100, 60, 220, 200], "confidence": 0.89}, "face_quality": {"overall_score": 0.81}, "embedding_extracted": true}',
 '{"is_match": true, "similarity_score": 0.85, "threshold_used": 0.6, "confidence_level": "HIGH"}',
 '{"is_live": true, "score": 0.78, "checks": {"face_confidence": 0.89, "face_size_adequate": true, "face_quality": 0.81, "multiple_faces": true}}',
 '{"status": "PASSED", "confidence": 0.82, "decision_factors": []}',
 '{"id_card_quality": 0.87, "selfie_quality": 0.81, "liveness_score": 0.78, "face_match_score": 0.85}',
 ARRAY['Verification successful', 'High confidence match'],
 1250, 'InsightFace-R100-v1.0', CURRENT_TIMESTAMP - INTERVAL '2 hours'),

('880e8400-e29b-41d4-a716-446655440002', '660e8400-e29b-41d4-a716-446655440002', 'VER-2025-0917-002', 'complete',
 '{"faces_detected": 1, "primary_face": {"bbox": [115, 75, 195, 175], "confidence": 0.52}, "face_quality": {"overall_score": 0.45}, "embedding_extracted": true}',
 '{"faces_detected": 1, "primary_face": {"bbox": [95, 55, 215, 195], "confidence": 0.68}, "face_quality": {"overall_score": 0.65}, "embedding_extracted": true}',
 '{"is_match": false, "similarity_score": 0.45, "threshold_used": 0.6, "confidence_level": "LOW"}',
 '{"is_live": false, "score": 0.30, "reason": "Liveness detection failed", "checks": {"face_confidence": 0.68, "face_size_adequate": true, "face_quality": 0.65, "multiple_faces": true}}',
 '{"status": "FAILED", "confidence": 0.38, "decision_factors": ["Face does not match ID document", "Liveness detection failed", "Face similarity below threshold"]}',
 '{"id_card_quality": 0.45, "selfie_quality": 0.65, "liveness_score": 0.30, "face_match_score": 0.45}',
 ARRAY['Improve ID document image quality - ensure good lighting and clear focus', 'Take a live selfie - avoid using photos of photos', 'Ensure the person in selfie matches the ID document'],
 1850, 'InsightFace-R100-v1.0', CURRENT_TIMESTAMP - INTERVAL '3 hours')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE FACE IDENTITIES & EMBEDDINGS
-- =====================================================

INSERT INTO training.face_identities (id, person_id, full_name, metadata, created_at) VALUES
('990e8400-e29b-41d4-a716-446655440001', 'PERSON_001', 'Nguyen Van Test', '{"verified": true, "source": "ekyc_verification", "documents": ["CCCD123456789"]}', CURRENT_TIMESTAMP - INTERVAL '1 day'),
('990e8400-e29b-41d4-a716-446655440002', 'PERSON_002', 'Tran Thi Demo', '{"verified": false, "source": "training_data", "notes": "Test identity for development"}', CURRENT_TIMESTAMP - INTERVAL '2 days'),
('990e8400-e29b-41d4-a716-446655440003', 'PERSON_003', 'Le Van Sample', '{"verified": true, "source": "manual_entry", "documents": ["CCCD987654321"]}', CURRENT_TIMESTAMP - INTERVAL '3 days')
ON CONFLICT (id) DO NOTHING;

-- Generate sample face embeddings (512-dimensional vectors with random values for testing)
INSERT INTO training.face_embeddings (id, identity_id, embedding, embedding_version, source_type, source_file_path, quality_score, confidence_score, face_bbox, created_at) VALUES
('aa0e8400-e29b-41d4-a716-446655440001', '990e8400-e29b-41d4-a716-446655440001', 
 -- Sample embedding (in real scenario, this would be actual face embedding from model)
 ARRAY[0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678, -0.901, 0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678, 0.901, -0.123]::FLOAT[], 
 'InsightFace-R100-v1.0', 'verification', '/storage/sessions/660e8400-e29b-41d4-a716-446655440001/id_front.jpg', 
 0.87, 0.92, ARRAY[120, 80, 200, 180], CURRENT_TIMESTAMP - INTERVAL '2 hours')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE TRAINING DATA
-- =====================================================

INSERT INTO training.datasets (id, dataset_name, description, dataset_type, file_path, total_images, total_persons, statistics, quality_metrics, created_by, created_at) VALUES
('bb0e8400-e29b-41d4-a716-446655440001', 'ekyc_production_dataset_v1', 'Production EKYC verification dataset for training', 'training', '/storage/datasets/ekyc_prod_v1.pkl', 1250, 83, 
 '{"avg_images_per_person": 15.06, "image_formats": {"jpg": 1100, "png": 150}, "resolution_stats": {"min": "224x224", "max": "1920x1080", "avg": "640x480"}}',
 '{"overall_score": 0.78, "balance_score": 0.65, "embedding_quality": 0.82}', 'admin', CURRENT_TIMESTAMP - INTERVAL '1 week'),
 
('bb0e8400-e29b-41d4-a716-446655440002', 'validation_dataset_v1', 'Validation dataset for model evaluation', 'validation', '/storage/datasets/validation_v1.pkl', 320, 20,
 '{"avg_images_per_person": 16.0, "image_formats": {"jpg": 300, "png": 20}, "resolution_stats": {"min": "256x256", "max": "1024x768", "avg": "512x384"}}',
 '{"overall_score": 0.85, "balance_score": 0.80, "embedding_quality": 0.87}', 'admin', CURRENT_TIMESTAMP - INTERVAL '1 week')
ON CONFLICT (id) DO NOTHING;

INSERT INTO training.training_sessions (id, training_id, dataset_id, training_config, status, start_time, end_time, duration_seconds, metrics, model_info, created_by) VALUES
('cc0e8400-e29b-41d4-a716-446655440001', 'TRAIN-2025-0917-001', 'bb0e8400-e29b-41d4-a716-446655440001',
 '{"validation_split": 0.2, "epochs": 10, "batch_size": 32, "learning_rate": 0.001, "augmentation": {"enabled": true}}',
 'completed', CURRENT_TIMESTAMP - INTERVAL '5 days', CURRENT_TIMESTAMP - INTERVAL '5 days' + INTERVAL '2 hours', 7200,
 '{"accuracy": 0.94, "precision": 0.92, "recall": 0.96, "f1_score": 0.94, "avg_confidence": 0.87}',
 '{"model_type": "InsightFace_R100", "embedding_size": 512, "optimal_threshold": 0.65, "training_samples": 1000, "validation_samples": 250}',
 'admin')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE USER FEEDBACK
-- =====================================================

INSERT INTO training.user_feedback (id, feedback_id, verification_id, user_id, actual_result, system_result, confidence_score, feedback_type, user_comments, created_at) VALUES
('dd0e8400-e29b-41d4-a716-446655440001', 'FB-2025-0917-001', 'VER-2025-0917-001', 'test_user_1', true, true, 0.82, 'accuracy', 'Correct verification result', CURRENT_TIMESTAMP - INTERVAL '1 hour'),
('dd0e8400-e29b-41d4-a716-446655440002', 'FB-2025-0917-002', 'VER-2025-0917-002', 'test_user_2', false, false, 0.38, 'accuracy', 'System correctly rejected fake selfie', CURRENT_TIMESTAMP - INTERVAL '2 hours')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE WEBHOOK DELIVERIES
-- =====================================================

INSERT INTO webhook_deliveries (id, session_id, webhook_url, event_type, payload, status, response_code, attempts, created_at, delivered_at) VALUES
('ee0e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440001', 'https://api.example.com/webhooks/ekyc', 'session.completed',
 '{"session_id": "660e8400-e29b-41d4-a716-446655440001", "status": "completed", "decision": "APPROVED", "confidence": 0.82, "timestamp": "2025-09-17T10:00:00Z"}',
 'delivered', 200, 1, CURRENT_TIMESTAMP - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '1 hour'),
 
('ee0e8400-e29b-41d4-a716-446655440002', '660e8400-e29b-41d4-a716-446655440002', 'https://api.example.com/webhooks/ekyc', 'session.completed',
 '{"session_id": "660e8400-e29b-41d4-a716-446655440002", "status": "completed", "decision": "REJECTED", "confidence": 0.38, "timestamp": "2025-09-17T09:00:00Z"}',
 'delivered', 200, 1, CURRENT_TIMESTAMP - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '2 hours')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE TASK QUEUE ENTRIES
-- =====================================================

INSERT INTO task_queue (id, task_id, task_name, task_args, task_kwargs, status, result, started_at, completed_at, created_at) VALUES
('ff0e8400-e29b-41d4-a716-446655440001', 'process_ekyc_session_660e8400-e29b-41d4-a716-446655440001', 'process_ekyc_session',
 '["660e8400-e29b-41d4-a716-446655440001"]', '{}', 'completed',
 '{"success": true, "face_match_score": 0.85, "liveness_score": 0.78, "decision": "APPROVED"}',
 CURRENT_TIMESTAMP - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '1 hour 30 minutes', CURRENT_TIMESTAMP - INTERVAL '2 hours'),
 
('ff0e8400-e29b-41d4-a716-446655440002', 'send_webhook_ee0e8400-e29b-41d4-a716-446655440001', 'send_webhook_notification',
 '["ee0e8400-e29b-41d4-a716-446655440001"]', '{}', 'completed',
 '{"delivered": true, "response_code": 200}',
 CURRENT_TIMESTAMP - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '1 hour')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE AUDIT LOGS
-- =====================================================

INSERT INTO audit.activity_logs (id, user_id, session_id, action, resource_type, resource_id, details, ip_address, success, timestamp) VALUES
('110e8400-e29b-41d4-a716-446655440001', 'test_user_1', '660e8400-e29b-41d4-a716-446655440001', 'session.create', 'ekyc_session', '660e8400-e29b-41d4-a716-446655440001',
 '{"user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}', '192.168.1.100', true, CURRENT_TIMESTAMP - INTERVAL '2 hours'),
 
('110e8400-e29b-41d4-a716-446655440002', 'test_user_1', '660e8400-e29b-41d4-a716-446655440001', 'asset.upload', 'ekyc_asset', '770e8400-e29b-41d4-a716-446655440001',
 '{"asset_type": "id_front", "file_size": 2458320, "mime_type": "image/jpeg"}', '192.168.1.100', true, CURRENT_TIMESTAMP - INTERVAL '2 hours'),
 
('110e8400-e29b-41d4-a716-446655440003', 'test_user_1', '660e8400-e29b-41d4-a716-446655440001', 'verification.complete', 'ekyc_verification', '880e8400-e29b-41d4-a716-446655440001',
 '{"decision": "APPROVED", "confidence": 0.82, "processing_time_ms": 1250}', '192.168.1.100', true, CURRENT_TIMESTAMP - INTERVAL '1 hour 30 minutes')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE API KEYS
-- =====================================================

INSERT INTO api_keys (id, key_name, api_key_hash, permissions, rate_limit_per_minute, is_active, created_by, created_at) VALUES
('220e8400-e29b-41d4-a716-446655440001', 'Development API Key', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu', 
 ARRAY['read', 'write'], 100, true, 'admin', CURRENT_TIMESTAMP - INTERVAL '1 week'),
 
('220e8400-e29b-41d4-a716-446655440002', 'Production API Key', '$2b$12$LQv3c1yqBFVyE6sNT8emCO0MCUuL2lINnoy6PH.y3TxSQw8.5qiFu',
 ARRAY['read', 'write', 'admin'], 200, true, 'admin', CURRENT_TIMESTAMP - INTERVAL '2 weeks')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================

DO $$
DECLARE
    session_count INT;
    asset_count INT;
    verification_count INT;
    identity_count INT;
BEGIN
    SELECT COUNT(*) INTO session_count FROM ekyc_sessions;
    SELECT COUNT(*) INTO asset_count FROM ekyc_assets;
    SELECT COUNT(*) INTO verification_count FROM ekyc_verifications;
    SELECT COUNT(*) INTO identity_count FROM training.face_identities;
    
    RAISE NOTICE '✅ Sample data inserted successfully!';
    RAISE NOTICE '📊 EKYC Sessions: %', session_count;
    RAISE NOTICE '📁 Assets: %', asset_count;
    RAISE NOTICE '✔️ Verifications: %', verification_count;
    RAISE NOTICE '👤 Face Identities: %', identity_count;
    RAISE NOTICE '🎯 Database ready for testing and development!';
END $$;
