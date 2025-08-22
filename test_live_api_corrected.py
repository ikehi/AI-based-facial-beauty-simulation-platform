"""
Corrected Live API Testing Script
Tests the running API server with proper form data and file uploads
"""

import requests
import json
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
import os

class CorrectedLiveAPITester:
    """Test the live API server with proper form data"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint"""
        print("🏥 Testing Health Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   AI Modules: {data['ai_modules']}")
                print(f"   Real AI Available: {data['real_ai_available']}")
                self.test_results['health'] = True
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                self.test_results['health'] = False
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            self.test_results['health'] = False
            return False
    
    def test_makeup_styles(self) -> bool:
        """Test makeup styles endpoint"""
        print("\n💄 Testing Makeup Styles...")
        try:
            response = self.session.get(f"{self.base_url}/api/makeup/styles")
            if response.status_code == 200:
                data = response.json()
                styles = data['styles']
                print(f"✅ Makeup styles retrieved: {len(styles)} styles available")
                for style_name, style_info in styles.items():
                    print(f"   - {style_name}: {style_info['description']}")
                self.test_results['makeup_styles'] = True
                return True
            else:
                print(f"❌ Makeup styles failed: {response.status_code}")
                self.test_results['makeup_styles'] = False
                return False
        except Exception as e:
            print(f"❌ Makeup styles error: {e}")
            self.test_results['makeup_styles'] = False
            return False
    
    def test_hair_styles(self) -> bool:
        """Test hair styles endpoint"""
        print("\n💇 Testing Hair Styles...")
        try:
            response = self.session.get(f"{self.base_url}/api/hair/styles")
            if response.status_code == 200:
                data = response.json()
                colors = data['colors']
                styles = data['styles']
                print(f"✅ Hair styles retrieved: {len(colors)} colors, {len(styles)} styles")
                print("   Colors:", ", ".join(list(colors.keys())[:5]) + "...")
                print("   Styles:", ", ".join(list(styles.keys())[:5]) + "...")
                self.test_results['hair_styles'] = True
                return True
            else:
                print(f"❌ Hair styles failed: {response.status_code}")
                self.test_results['hair_styles'] = False
                return False
        except Exception as e:
            print(f"❌ Hair styles error: {e}")
            self.test_results['hair_styles'] = False
            return False
    
    def create_test_image(self, width: int = 400, height: int = 500) -> str:
        """Create a test image and return the file path"""
        # Create a simple test image
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add a face-like shape
        center = (width // 2, height // 2)
        face_axes = (width // 3, height // 2)
        cv2.ellipse(image, center, face_axes, 0, 0, 360, (220, 200, 180), -1)
        
        # Add eyes
        left_eye = (center[0] - width // 6, center[1] - height // 6)
        right_eye = (center[0] + width // 6, center[1] - height // 6)
        eye_size = (width // 12, height // 20)
        cv2.ellipse(image, left_eye, eye_size, 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, right_eye, eye_size, 0, 0, 360, (255, 255, 255), -1)
        
        # Add pupils
        pupil_size = (width // 24, height // 40)
        cv2.ellipse(image, left_eye, pupil_size, 0, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(image, right_eye, pupil_size, 0, 0, 360, (20, 20, 20), -1)
        
        # Add mouth
        mouth_center = (center[0], center[1] + height // 4)
        mouth_size = (width // 8, height // 20)
        cv2.ellipse(image, mouth_center, mouth_size, 0, 0, 360, (180, 120, 100), -1)
        
        # Save image
        filename = 'test_api_image.jpg'
        cv2.imwrite(filename, image)
        print(f"   📸 Created test image: {filename}")
        return filename
    
    def test_face_detection(self) -> bool:
        """Test face detection endpoint"""
        print("\n👤 Testing Face Detection...")
        try:
            # Create test image
            image_path = self.create_test_image()
            
            # Test face detection with form data
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                response = self.session.post(
                    f"{self.base_url}/api/face/detect",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()
                faces_detected = data.get('faces_detected', 0)
                print(f"✅ Face detection successful: {faces_detected} face(s) detected")
                if faces_detected > 0:
                    print(f"   Processing time: {data.get('processing_time', 'N/A')}s")
                self.test_results['face_detection'] = True
                return True
            else:
                print(f"❌ Face detection failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['face_detection'] = False
                return False
                
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            self.test_results['face_detection'] = False
            return False
    
    def test_makeup_transfer(self) -> bool:
        """Test makeup transfer endpoint"""
        print("\n💄 Testing Makeup Transfer...")
        try:
            # Create test image
            image_path = self.create_test_image()
            
            # Test makeup transfer with form data
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'style': 'casual',
                    'intensity': '0.7'
                }
                response = self.session.post(
                    f"{self.base_url}/api/makeup/apply",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                print("✅ Makeup transfer successful")
                print(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                print(f"   Content-Length: {response.headers.get('Content-Length', 'N/A')}")
                
                # Save result
                if response.content:
                    with open('makeup_result_api.jpg', 'wb') as f:
                        f.write(response.content)
                    print("   💾 Saved result: makeup_result_api.jpg")
                
                self.test_results['makeup_transfer'] = True
                return True
            else:
                print(f"❌ Makeup transfer failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['makeup_transfer'] = False
                return False
                
        except Exception as e:
            print(f"❌ Makeup transfer error: {e}")
            self.test_results['makeup_transfer'] = False
            return False
    
    def test_hair_transformation(self) -> bool:
        """Test hair transformation endpoint"""
        print("\n💇 Testing Hair Transformation...")
        try:
            # Create test image
            image_path = self.create_test_image()
            
            # Test hair styling with form data
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'style': 'wavy',
                    'intensity': '0.8'
                }
                response = self.session.post(
                    f"{self.base_url}/api/hair/style",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                print("✅ Hair styling successful")
                print(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                print(f"   Content-Length: {response.headers.get('Content-Length', 'N/A')}")
                
                # Save result
                if response.content:
                    with open('hair_style_result_api.jpg', 'wb') as f:
                        f.write(response.content)
                    print("   💾 Saved result: hair_style_result_api.jpg")
                
                self.test_results['hair_styling'] = True
                return True
            else:
                print(f"❌ Hair styling failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['hair_styling'] = False
                return False
                
        except Exception as e:
            print(f"❌ Hair styling error: {e}")
            self.test_results['hair_styling'] = False
            return False
    
    def test_hair_color_change(self) -> bool:
        """Test hair color change endpoint"""
        print("\n🎨 Testing Hair Color Change...")
        try:
            # Create test image
            image_path = self.create_test_image()
            
            # Test hair color change with form data
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'color': 'blonde',
                    'intensity': '0.8'
                }
                response = self.session.post(
                    f"{self.base_url}/api/hair/color",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                print("✅ Hair color change successful")
                print(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                print(f"   Content-Length: {response.headers.get('Content-Length', 'N/A')}")
                
                # Save result
                if response.content:
                    with open('hair_color_result_api.jpg', 'wb') as f:
                        f.write(response.content)
                    print("   💾 Saved result: hair_color_result_api.jpg")
                
                self.test_results['hair_color_change'] = True
                return True
            else:
                print(f"❌ Hair color change failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['hair_color_change'] = False
                return False
                
        except Exception as e:
            print(f"❌ Hair color change error: {e}")
            self.test_results['hair_color_change'] = False
            return False
    
    def test_full_beauty_transformation(self) -> bool:
        """Test full beauty transformation endpoint"""
        print("\n✨ Testing Full Beauty Transformation...")
        try:
            # Create test image
            image_path = self.create_test_image()
            
            # Test full beauty transformation with form data
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'makeup_style': 'evening',
                    'makeup_intensity': '0.8',
                    'hair_style': 'curly',
                    'hair_intensity': '0.9',
                    'hair_color': 'red',
                    'color_intensity': '0.8'
                }
                response = self.session.post(
                    f"{self.base_url}/api/beauty/full",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                print("✅ Full beauty transformation successful")
                print(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                print(f"   Content-Length: {response.headers.get('Content-Length', 'N/A')}")
                
                # Save result
                if response.content:
                    with open('full_beauty_result_api.jpg', 'wb') as f:
                        f.write(response.content)
                    print("   💾 Saved result: full_beauty_result_api.jpg")
                
                self.test_results['full_beauty_transformation'] = True
                return True
            else:
                print(f"❌ Full beauty transformation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['full_beauty_transformation'] = False
                return False
                
        except Exception as e:
            print(f"❌ Full beauty transformation error: {e}")
            self.test_results['full_beauty_transformation'] = False
            return False
    
    def test_system_info(self) -> bool:
        """Test system info endpoint"""
        print("\n🔧 Testing System Info...")
        try:
            response = self.session.get(f"{self.base_url}/api/system/info")
            if response.status_code == 200:
                data = response.json()
                print("✅ System info retrieved successfully")
                print(f"   Real AI Available: {data.get('system_info', {}).get('real_ai_available', 'N/A')}")
                print(f"   Fallback Available: {data.get('system_info', {}).get('fallback_available', 'N/A')}")
                self.test_results['system_info'] = True
                return True
            else:
                print(f"❌ System info failed: {response.status_code}")
                self.test_results['system_info'] = False
                return False
        except Exception as e:
            print(f"❌ System info error: {e}")
            self.test_results['system_info'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests"""
        print("🚀 CORRECTED LIVE API TESTING SUITE")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        print("Using proper form data and file uploads")
        print()
        
        # Run all tests
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Makeup Styles", self.test_makeup_styles),
            ("Hair Styles", self.test_hair_styles),
            ("Face Detection", self.test_face_detection),
            ("Makeup Transfer", self.test_makeup_transfer),
            ("Hair Styling", self.test_hair_transformation),
            ("Hair Color Change", self.test_hair_color_change),
            ("Full Beauty Transformation", self.test_full_beauty_transformation),
            ("System Info", self.test_system_info)
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"❌ {test_name} test failed with exception: {e}")
                test_key = test_name.lower().replace(' ', '_')
                self.test_results[test_key] = False
        
        return self.test_results
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! The API is working perfectly!")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Check the logs above for details.")


def main():
    """Main testing function"""
    # Check if API server is running
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding. Please start the server first.")
            print("   Run: python src/api/real_ai_app.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API server. Please start the server first.")
        print("   Run: python src/api/real_ai_app.py")
        return
    
    # Run tests
    tester = CorrectedLiveAPITester()
    results = tester.run_all_tests()
    tester.print_summary()


if __name__ == "__main__":
    main()
