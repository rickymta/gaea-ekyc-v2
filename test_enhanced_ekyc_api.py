"""
Test script for Enhanced EKYC API
Script để test các endpoint của Enhanced EKYC API
"""

import requests
import json
import time
from pathlib import Path
import base64

# API Base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing Health Check...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/ekyc/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def create_test_image():
    """Create a simple test image using PIL"""
    try:
        from PIL import Image, ImageDraw
        import io
        
        # Create a simple 300x300 RGB image
        img = Image.new('RGB', (300, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face-like shape
        draw.ellipse([100, 100, 200, 200], fill='lightpink', outline='black')  # Face
        draw.ellipse([130, 130, 140, 140], fill='black')  # Left eye
        draw.ellipse([160, 130, 170, 140], fill='black')  # Right eye
        draw.arc([140, 160, 160, 170], 0, 180, fill='black')  # Smile
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
        
    except ImportError:
        print("⚠️ PIL not available, using placeholder")
        return b"fake_image_data"


def test_image_quality_assessment():
    """Test image quality assessment endpoint"""
    print("\n📷 Testing Image Quality Assessment...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        files = {
            'image': ('test_selfie.jpg', test_image, 'image/jpeg')
        }
        data = {
            'image_type': 'selfie',
            'session_id': 'test-session-quality'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/assess-image-quality",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Image quality test failed: {e}")
        return False


def test_liveness_detection():
    """Test liveness detection endpoint"""
    print("\n👤 Testing Liveness Detection...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        files = {
            'image_or_video': ('test_selfie.jpg', test_image, 'image/jpeg')
        }
        data = {
            'session_id': 'test-session-liveness'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/verify-liveness-only",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Liveness detection test failed: {e}")
        return False


def test_ocr_extraction():
    """Test OCR ID card extraction endpoint"""
    print("\n📄 Testing OCR ID Card Extraction...")
    
    try:
        # Create test images for ID card
        test_image_front = create_test_image()
        test_image_back = create_test_image()
        
        files = {
            'id_front_image': ('id_front.jpg', test_image_front, 'image/jpeg'),
            'id_back_image': ('id_back.jpg', test_image_back, 'image/jpeg')
        }
        data = {
            'session_id': 'test-session-ocr'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/extract-id-info",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OCR extraction test failed: {e}")
        return False


def test_face_matching():
    """Test face matching endpoint"""
    print("\n👥 Testing Face Matching...")
    
    try:
        # Create test images
        selfie_image = create_test_image()
        id_image = create_test_image()
        
        files = {
            'selfie_image': ('selfie.jpg', selfie_image, 'image/jpeg'),
            'id_image': ('id_card.jpg', id_image, 'image/jpeg')
        }
        data = {
            'session_id': 'test-session-face-match'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/face-match",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Face matching test failed: {e}")
        return False


def test_simple_face_matching():
    """Test simple face matching endpoint (NEW API)"""
    print("\n🆕 Testing Simple Face Matching (No Liveness)...")
    
    try:
        # Create test images
        selfie_image = create_test_image()
        id_image = create_test_image()
        
        files = {
            'selfie_image': ('selfie.jpg', selfie_image, 'image/jpeg'),
            'id_image': ('id_card.jpg', id_image, 'image/jpeg')
        }
        data = {
            'session_id': 'test-session-simple-face-match'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/face-match-simple",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Verify this API does not include liveness detection
            data_section = result.get('data', {})
            if 'liveness_detection' not in data_section:
                print("✅ Confirmed: No liveness detection in simple API")
            else:
                print("⚠️ Warning: Liveness detection found in simple API")
                
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Simple face matching test failed: {e}")
        return False


def test_simple_face_matching_detailed():
    """Test simple face matching with detailed analysis"""
    print("\n🔍 Testing Simple Face Matching (Detailed Analysis)...")
    
    try:
        # Create test images
        selfie_image = create_test_image()
        id_image = create_test_image()
        
        files = {
            'selfie_image': ('selfie.jpg', selfie_image, 'image/jpeg'),
            'id_image': ('id_card.jpg', id_image, 'image/jpeg')
        }
        data = {
            'session_id': 'test-session-simple-detailed',
            'return_detailed_analysis': 'true',
            'skip_quality_check': 'false'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/face-match-simple",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Check for detailed analysis
            data_section = result.get('data', {})
            if data_section.get('detailed_analysis'):
                print("✅ Detailed analysis included")
            if data_section.get('quality_assessment'):
                print("✅ Quality assessment included")
                
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Detailed simple face matching test failed: {e}")
        return False


def test_simple_face_matching_fast():
    """Test simple face matching with quality checks skipped"""
    print("\n⚡ Testing Simple Face Matching (Fast Mode - No Quality Check)...")
    
    try:
        # Create test images
        selfie_image = create_test_image()
        id_image = create_test_image()
        
        files = {
            'selfie_image': ('selfie.jpg', selfie_image, 'image/jpeg'),
            'id_image': ('id_card.jpg', id_image, 'image/jpeg')
        }
        data = {
            'session_id': 'test-session-simple-fast',
            'skip_quality_check': 'true',
            'return_detailed_analysis': 'false'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/face-match-simple",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Check that quality assessment was skipped
            data_section = result.get('data', {})
            quality_assessment = data_section.get('quality_assessment')
            if quality_assessment is None:
                print("✅ Quality assessment skipped as requested")
            elif isinstance(quality_assessment, dict) and quality_assessment.get('error'):
                print("✅ Quality assessment marked as unavailable")
                
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Fast simple face matching test failed: {e}")
        return False


def test_complete_verification():
    """Test complete EKYC verification endpoint"""
    print("\n🎯 Testing Complete EKYC Verification...")
    
    try:
        # Create test images
        selfie_image = create_test_image()
        id_front_image = create_test_image()
        id_back_image = create_test_image()
        
        files = {
            'selfie_image': ('selfie.jpg', selfie_image, 'image/jpeg'),
            'id_front_image': ('id_front.jpg', id_front_image, 'image/jpeg'),
            'id_back_image': ('id_back.jpg', id_back_image, 'image/jpeg')
        }
        data = {
            'session_id': 'test-session-complete'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/ekyc/verify-complete",
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Check verification status
            session_id = result.get('data', {}).get('session_id')
            if session_id:
                test_verification_status(session_id)
            
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Complete verification test failed: {e}")
        return False


def test_verification_status(session_id: str):
    """Test verification status endpoint"""
    print(f"\n📊 Testing Verification Status for session: {session_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/ekyc/verification-status/{session_id}")
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Verification status test failed: {e}")
        return False


def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting Enhanced EKYC API Tests...")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Image Quality Assessment", test_image_quality_assessment),
        ("Liveness Detection", test_liveness_detection),
        ("OCR ID Card Extraction", test_ocr_extraction),
        ("Face Matching (Complete)", test_face_matching),
        ("Simple Face Matching (Basic)", test_simple_face_matching),
        ("Simple Face Matching (Detailed)", test_simple_face_matching_detailed),
        ("Simple Face Matching (Fast Mode)", test_simple_face_matching_fast),
        ("Complete EKYC Verification", test_complete_verification),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📋 TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed successfully!")
    else:
        print(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    print("Enhanced EKYC API Test Suite")
    print("============================")
    print(f"Base URL: {BASE_URL}")
    print(f"Testing time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            success = run_all_tests()
            exit(0 if success else 1)
        else:
            print("❌ Server responded but with error")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ Server is not running or not reachable")
        print("💡 Please start the server first:")
        print("   cd app && python main.py")
        exit(1)
