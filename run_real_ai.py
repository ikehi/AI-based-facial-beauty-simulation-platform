#!/usr/bin/env python3
"""
Real AI Beauty Platform Startup Script
Handles dependencies and provides user-friendly interface
"""

import os
import sys
import subprocess
import importlib
import logging
from typing import List, Dict, Optional

def setup_logging():
    """Setup logging for the startup script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available"""
    logger = logging.getLogger(__name__)
    
    dependencies = {
        'opencv': False,
        'numpy': False,
        'torch': False,
        'mediapipe': False,
        'ultralytics': False,
        'flask': False
    }
    
    for dep in dependencies.keys():
        try:
            importlib.import_module(dep)
            dependencies[dep] = True
            logger.info(f"✓ {dep} is available")
        except ImportError:
            logger.warning(f"✗ {dep} is not available")
    
    return dependencies

def install_dependencies():
    """Install required dependencies"""
    print("\n🔧 Installing dependencies...")
    
    try:
        # Install core dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'models/face_recognition',
        'models/makeup_ai',
        'models/hair_ai',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Directories created")

def download_models():
    """Download pre-trained models (placeholder)"""
    print("\n📥 Model Download Status:")
    print("  Note: This is a placeholder for model downloads")
    print("  In production, you would download:")
    print("    - YOLO face detection models")
    print("    - Face parsing models")
    print("    - Makeup transfer GANs")
    print("    - Hair transformation models")

def test_system():
    """Test the system components"""
    print("\n🧪 Testing system components...")
    
    try:
        # Test imports
        from src.face_recognition.robust_face_detector import RobustFaceDetector
        print("✓ Robust face detector imported")
        
        from src.makeup_ai.real_makeup_transfer import RealMakeupTransfer
        print("✓ Real makeup transfer imported")
        
        from src.hair_ai.real_hair_transformer import RealHairTransformer
        print("✓ Real hair transformer imported")
        
        from src.api.real_ai_app import RealAIBeautyAPI
        print("✓ Real AI API imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def start_api():
    """Start the Real AI Beauty API"""
    print("\n🚀 Starting Real AI Beauty API...")
    
    try:
        from src.api.real_ai_app import RealAIBeautyAPI
        
        # Initialize API
        api = RealAIBeautyAPI()
        
        print("✓ API initialized successfully")
        print("✓ Starting server on http://localhost:5000")
        print("✓ Press Ctrl+C to stop")
        
        # Start the API
        api.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return False

def run_tests():
    """Run the comprehensive test suite"""
    print("\n🧪 Running comprehensive test suite...")
    
    try:
        result = subprocess.run([sys.executable, "test_real_ai_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def show_menu():
    """Show the main menu"""
    print("\n" + "="*60)
    print("🎨 REAL AI BEAUTY PLATFORM - STARTUP MENU")
    print("="*60)
    print("1. Check system status")
    print("2. Install dependencies")
    print("3. Run system tests")
    print("4. Start API server")
    print("5. Run comprehensive tests")
    print("6. Exit")
    print("="*60)

def main():
    """Main startup function"""
    print("🎨 Welcome to the Real AI Beauty Platform!")
    print("This platform provides advanced AI-powered beauty transformations.")
    
    # Setup logging
    logger = setup_logging()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    while True:
        show_menu()
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            print("\n📊 System Status:")
            dependencies = check_dependencies()
            print(f"\nDependencies: {sum(dependencies.values())}/{len(dependencies)} available")
            
            if all(dependencies.values()):
                print("✓ All dependencies are available")
            else:
                print("⚠ Some dependencies are missing")
                print("Run option 2 to install dependencies")
        
        elif choice == '2':
            if install_dependencies():
                print("✓ Dependencies installed successfully")
                print("Run option 1 to check status again")
            else:
                print("❌ Failed to install dependencies")
        
        elif choice == '3':
            if test_system():
                print("✓ System test passed")
            else:
                print("❌ System test failed")
        
        elif choice == '4':
            start_api()
            break  # Exit after starting API
        
        elif choice == '5':
            if run_tests():
                print("✓ All tests passed")
            else:
                print("❌ Some tests failed")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
