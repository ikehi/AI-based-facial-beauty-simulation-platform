"""
Comprehensive Test Script for Real AI Beauty Platform
Tests all real AI modules and provides performance analysis
"""

import os
import sys
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.append('src')

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_real_ai.log')
        ]
    )
    return logging.getLogger(__name__)

def create_test_face_image(width: int = 400, height: int = 500) -> np.ndarray:
    """Create a realistic test face image that works with AI face detection"""
    # Create base image with skin tone
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Create a more realistic face shape (larger, more prominent)
    center = (width // 2, height // 2)
    face_axes = (width // 2, height // 2)  # Larger face area
    
    # Draw face outline with skin tone
    skin_color = (220, 200, 180)
    cv2.ellipse(image, center, face_axes, 0, 0, 360, skin_color, -1)
    
    # Add more prominent eyes (larger and more realistic)
    left_eye = (center[0] - width // 5, center[1] - height // 8)
    right_eye = (center[0] + width // 5, center[1] - height // 8)
    eye_size = (width // 8, height // 12)  # Larger eyes
    
    cv2.ellipse(image, left_eye, eye_size, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, right_eye, eye_size, 0, 0, 360, (255, 255, 255), -1)
    
    # Add pupils
    pupil_size = (width // 16, height // 24)
    cv2.ellipse(image, left_eye, pupil_size, 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(image, right_eye, pupil_size, 0, 0, 360, (20, 20, 20), -1)
    
    # Add eyebrows
    left_brow = (center[0] - width // 5, center[1] - height // 6)
    right_brow = (center[0] + width // 5, center[1] - height // 6)
    brow_size = (width // 6, height // 20)
    cv2.ellipse(image, left_brow, brow_size, 0, 0, 360, (80, 50, 30), -1)
    cv2.ellipse(image, right_brow, brow_size, 0, 0, 360, (80, 50, 30), -1)
    
    # Add nose (more prominent)
    nose_center = (center[0], center[1] + height // 10)
    nose_size = (width // 15, height // 12)
    cv2.ellipse(image, nose_center, nose_size, 0, 0, 360, (200, 180, 160), -1)
    
    # Add mouth (more prominent)
    mouth_center = (center[0], center[1] + height // 3)
    mouth_size = (width // 6, height // 15)
    cv2.ellipse(image, mouth_center, mouth_size, 0, 0, 360, (180, 120, 100), -1)
    
    # Add hair (more prominent and realistic)
    hair_center = (center[0], center[1] - height // 1.8)
    hair_size = (width // 1.5, height // 2.5)
    cv2.ellipse(image, hair_center, hair_size, 0, 0, 360, (80, 50, 30), -1)
    
    # Add ears
    left_ear = (center[0] - width // 2, center[1])
    right_ear = (center[0] + width // 2, center[1])
    ear_size = (width // 20, height // 8)
    cv2.ellipse(image, left_ear, ear_size, 0, 0, 360, skin_color, -1)
    cv2.ellipse(image, right_ear, ear_size, 0, 0, 360, skin_color, -1)
    
    # Add realistic skin texture and shadows
    for i in range(0, height, 5):  # More frequent noise
        for j in range(0, width, 5):
            noise = np.random.normal(0, 20)
            image[i, j] = np.clip(image[i, j] + noise, 0, 255)
    
    # Add subtle shadows around face features
    cv2.circle(image, left_eye, eye_size[0] + 5, (200, 180, 160), 2)
    cv2.circle(image, right_eye, eye_size[0] + 5, (200, 180, 160), 2)
    
    return image

def test_robust_face_detection():
    """Test the robust face detection system"""
    print("\n" + "="*60)
    print("TESTING ROBUST FACE DETECTION")
    print("="*60)
    
    try:
        from face_recognition.robust_face_detector import RobustFaceDetector
        
        # Try to use existing test image first, fallback to synthetic
        test_image_path = 'test_image.jpg'
        if os.path.exists(test_image_path):
            test_image = cv2.imread(test_image_path)
            print(f"✓ Loaded existing test image: {test_image_path}")
        else:
            # Create synthetic test image
            test_image = create_test_face_image()
            cv2.imwrite('test_face_input.jpg', test_image)
            print("✓ Created synthetic test face image")
        
        # Initialize detector
        detector = RobustFaceDetector()
        print("✓ Robust face detector initialized")
        
        # Test face detection
        start_time = time.time()
        detections = detector.detect_faces(test_image)
        processing_time = time.time() - start_time
        
        print(f"✓ Face detection completed in {processing_time:.3f}s")
        print(f"✓ Detected {len(detections)} face(s)")
        
        # Display detection details
        for i, detection in enumerate(detections):
            print(f"  Face {i+1}:")
            print(f"    Bbox: {detection.bbox}")
            print(f"    Confidence: {detection.confidence:.3f}")
            print(f"    Method: {detection.method}")
            print(f"    Quality Score: {detection.quality_score:.3f}")
        
        # Test statistics
        stats = detector.get_detection_stats()
        print(f"✓ Detection statistics:")
        print(f"    Methods available: {stats.get('methods_available', {})}")
        print(f"    Total detections: {stats.get('total_detections', 0)}")
        print(f"    Average processing time: {stats.get('average_processing_time', 0):.3f}s")
        
        # Test face cropping
        if detections:
            largest_face = detector.get_largest_face(detections)
            if largest_face:
                cropped_face = detector.crop_face(test_image, largest_face)
                cv2.imwrite('test_face_cropped.jpg', cropped_face)
                print("✓ Face cropping test passed")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"✗ Robust face detection test failed: {e}")
        logging.error(f"Robust face detection test failed: {e}")
        return False

def test_real_makeup_transfer():
    """Test the real makeup transfer system"""
    print("\n" + "="*60)
    print("TESTING REAL MAKEUP TRANSFER")
    print("="*60)
    
    try:
        from makeup_ai.real_makeup_transfer import RealMakeupTransfer
        
        # Use existing test image or create synthetic one
        test_image_path = 'test_image.jpg'
        if os.path.exists(test_image_path):
            test_image = cv2.imread(test_image_path)
            print(f"✓ Using existing test image: {test_image_path}")
        else:
            test_image = create_test_face_image()
            print("✓ Created synthetic test image")
        
        # Initialize makeup transfer
        makeup_transfer = RealMakeupTransfer()
        print("✓ Real makeup transfer initialized")
        
        # Get available styles
        styles = makeup_transfer.get_available_styles()
        print(f"✓ Available makeup styles: {styles}")
        
        # Test each style
        results = {}
        for style_name in styles[:3]:  # Test first 3 styles
            print(f"  Testing style: {style_name}")
            
            start_time = time.time()
            result_image = makeup_transfer.apply_makeup_style(test_image, style_name)
            processing_time = time.time() - start_time
            
            results[style_name] = {
                'image': result_image,
                'time': processing_time
            }
            
            print(f"    ✓ Applied in {processing_time:.3f}s")
            
            # Save result
            cv2.imwrite(f'test_makeup_{style_name}.jpg', result_image)
        
        # Test statistics
        stats = makeup_transfer.get_processing_stats()
        print(f"✓ Makeup transfer statistics:")
        print(f"    Total applications: {stats.get('total_applications', 0)}")
        print(f"    Average processing time: {stats.get('average_processing_time', 0):.3f}s")
        print(f"    AI models available: {stats.get('ai_models_available', {})}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"✗ Real makeup transfer test failed: {e}")
        logging.error(f"Real makeup transfer test failed: {e}")
        return False

def test_real_hair_transformation():
    """Test the real hair transformation system"""
    print("\n" + "="*60)
    print("TESTING REAL HAIR TRANSFORMATION")
    print("="*60)
    
    try:
        from hair_ai.real_hair_transformer import RealHairTransformer
        
        # Use existing test image or create synthetic one
        test_image_path = 'test_image.jpg'
        if os.path.exists(test_image_path):
            test_image = cv2.imread(test_image_path)
            print(f"✓ Using existing test image: {test_image_path}")
        else:
            test_image = create_test_face_image()
            print("✓ Created synthetic test image")
        
        # Initialize hair transformer
        hair_transformer = RealHairTransformer()
        print("✓ Real hair transformer initialized")
        
        # Get available styles and colors
        styles = hair_transformer.get_available_styles()
        colors = hair_transformer.get_available_colors()
        print(f"✓ Available hair styles: {styles}")
        print(f"✓ Available hair colors: {colors}")
        
        # Test hair styling
        style_results = {}
        for style_name in styles[:3]:  # Test first 3 styles
            print(f"  Testing hair style: {style_name}")
            
            start_time = time.time()
            result_image = hair_transformer.transform_hair_style(test_image, style_name)
            processing_time = time.time() - start_time
            
            style_results[style_name] = {
                'image': result_image,
                'time': processing_time
            }
            
            print(f"    ✓ Applied in {processing_time:.3f}s")
            
            # Save result
            cv2.imwrite(f'test_hair_style_{style_name}.jpg', result_image)
        
        # Test hair color change
        color_results = {}
        for color_name in colors[:3]:  # Test first 3 colors
            print(f"  Testing hair color: {color_name}")
            
            start_time = time.time()
            result_image = hair_transformer.change_hair_color(test_image, color_name)
            processing_time = time.time() - start_time
            
            color_results[color_name] = {
                'image': result_image,
                'time': processing_time
            }
            
            print(f"    ✓ Applied in {processing_time:.3f}s")
            
            # Save result
            cv2.imwrite(f'test_hair_color_{color_name}.jpg', result_image)
        
        # Test statistics
        stats = hair_transformer.get_processing_stats()
        print(f"✓ Hair transformation statistics:")
        print(f"    Total transformations: {stats.get('total_transformations', 0)}")
        print(f"    Average processing time: {stats.get('average_processing_time', 0):.3f}s")
        print(f"    AI models available: {stats.get('ai_models_available', {})}")
        
        return len(style_results) > 0 and len(color_results) > 0
        
    except Exception as e:
        print(f"✗ Real hair transformation test failed: {e}")
        logging.error(f"Real hair transformation test failed: {e}")
        return False

def test_real_ai_api():
    """Test the real AI API"""
    print("\n" + "="*60)
    print("TESTING REAL AI API")
    print("="*60)
    
    try:
        from api.real_ai_app import RealAIBeautyAPI
        
        # Initialize API
        api = RealAIBeautyAPI()
        print("✓ Real AI API initialized")
        
        # Test health endpoint
        with api.app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Health endpoint working")
                health_data = response.get_json()
                print(f"    AI modules: {health_data.get('ai_modules', {})}")
                print(f"    Real AI available: {health_data.get('real_ai_available', False)}")
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
        
        # Test makeup styles endpoint
        with api.app.test_client() as client:
            response = client.get('/api/makeup/styles')
            if response.status_code == 200:
                print("✓ Makeup styles endpoint working")
                styles_data = response.get_json()
                print(f"    Available styles: {list(styles_data.get('styles', {}).keys())}")
            else:
                print(f"✗ Makeup styles endpoint failed: {response.status_code}")
        
        # Test hair styles endpoint
        with api.app.test_client() as client:
            response = client.get('/api/hair/styles')
            if response.status_code == 200:
                print("✓ Hair styles endpoint working")
                hair_data = response.get_json()
                print(f"    Available styles: {list(hair_data.get('styles', {}).keys())}")
                print(f"    Available colors: {list(hair_data.get('colors', {}).keys())}")
            else:
                print(f"✗ Hair styles endpoint failed: {response.status_code}")
        
        # Test system info endpoint
        with api.app.test_client() as client:
            response = client.get('/api/system/info')
            if response.status_code == 200:
                print("✓ System info endpoint working")
                system_data = response.get_json()
                print(f"    API stats: {system_data.get('system_info', {}).get('api_stats', {})}")
            else:
                print(f"✗ System info endpoint failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"✗ Real AI API test failed: {e}")
        logging.error(f"Real AI API test failed: {e}")
        return False

def performance_comparison():
    """Compare performance between real AI and fallback systems"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    try:
        # Use existing test image or create synthetic one
        test_image_path = 'test_image.jpg'
        if os.path.exists(test_image_path):
            test_image = cv2.imread(test_image_path)
            print(f"✓ Using existing test image: {test_image_path}")
        else:
            test_image = create_test_face_image()
            print("✓ Created synthetic test image")
        
        # Test robust face detection
        print("Testing Robust Face Detection:")
        start_time = time.time()
        from face_recognition.robust_face_detector import RobustFaceDetector
        robust_detector = RobustFaceDetector()
        robust_detections = robust_detector.detect_faces(test_image)
        robust_time = time.time() - start_time
        print(f"  ✓ Robust: {len(robust_detections)} faces in {robust_time:.3f}s")
        
        # Test enhanced face detection (fallback)
        print("Testing Enhanced Face Detection (Fallback):")
        start_time = time.time()
        from face_recognition.enhanced_face_detector import EnhancedFaceDetector
        enhanced_detector = EnhancedFaceDetector()
        enhanced_detections = enhanced_detector.detect_faces(test_image)
        enhanced_time = time.time() - start_time
        print(f"  ✓ Enhanced: {len(enhanced_detections)} faces in {enhanced_time:.3f}s")
        
        # Performance comparison
        if robust_time > 0 and enhanced_time > 0:
            speedup = enhanced_time / robust_time
            print(f"\nPerformance Analysis:")
            print(f"  Robust detection: {robust_time:.3f}s")
            print(f"  Enhanced detection: {enhanced_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            
            if speedup > 1:
                print(f"  ✓ Robust detection is {speedup:.2f}x faster")
            else:
                print(f"  ⚠ Enhanced detection is {1/speedup:.2f}x faster")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        logging.error(f"Performance comparison failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 REAL AI BEAUTY PLATFORM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Setup logging
    logger = setup_logging()
    
    # Test results
    test_results = {}
    
    # Run all tests
    test_results['robust_face_detection'] = test_robust_face_detection()
    test_results['real_makeup_transfer'] = test_real_makeup_transfer()
    test_results['real_hair_transformation'] = test_real_hair_transformation()
    test_results['real_ai_api'] = test_real_ai_api()
    test_results['performance_comparison'] = performance_comparison()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Real AI system is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check logs for details.")
    
    # Save test results
    with open('test_real_ai_results.txt', 'w') as f:
        f.write("Real AI Beauty Platform - Test Results\n")
        f.write("="*50 + "\n\n")
        for test_name, result in test_results.items():
            f.write(f"{test_name}: {'PASS' if result else 'FAIL'}\n")
        f.write(f"\nOverall: {passed}/{total} tests passed\n")
    
    print(f"\nTest results saved to: test_real_ai_results.txt")
    print(f"Log file: test_real_ai.log")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
