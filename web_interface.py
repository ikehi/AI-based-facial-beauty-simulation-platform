"""
Web Interface for AI Beauty Platform
Provides a modern web UI for testing all features
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import time

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Beauty Platform</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .feature-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        .feature-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }
        
        .feature-card p {
            margin-bottom: 20px;
            line-height: 1.6;
            color: #666;
        }
        
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background-color: #f8f9ff;
        }
        
        .upload-area.dragover {
            border-color: #764ba2;
            background-color: #f0f2ff;
        }
        
        .upload-area input[type="file"] {
            display: none;
        }
        
        .upload-area label {
            color: #667eea;
            font-weight: bold;
            cursor: pointer;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
        }
        
        .control-group label {
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        
        .control-group select,
        .control-group input {
            padding: 10px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        .control-group select:focus,
        .control-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            margin-top: 20px;
            text-align: center;
        }
        
        .image-preview {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            margin: 10px 0;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .status {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            font-weight: bold;
        }
        
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .demo-section {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .demo-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .demo-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .demo-btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .demo-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✨ AI Beauty Platform</h1>
            <p>Advanced AI-powered facial beauty transformation with real-time processing</p>
        </div>
        
        <div class="demo-section">
            <h2>🚀 Live Demo Features</h2>
            <div class="demo-buttons">
                <button class="demo-btn" onclick="startMediaPipeDemo()">Start MediaPipe Demo</button>
                <button class="demo-btn" onclick="startRealTimeProcessing()">Real-time Video Processing</button>
                <button class="demo-btn" onclick="startFaceMeshDemo()">Face Mesh (468 Landmarks)</button>
                <button class="demo-btn" onclick="startGestureControl()">Hand Gesture Control</button>
            </div>
        </div>
        
        <div class="features-grid">
            <div class="feature-card">
                <h3>🎭 Face Detection & Analysis</h3>
                <p>Advanced AI-powered face detection with multiple algorithms including MediaPipe, YOLO, and OpenCV.</p>
                
                <div class="upload-area" onclick="document.getElementById('face-upload').click()">
                    <input type="file" id="face-upload" accept="image/*" onchange="handleFaceUpload(event)">
                    <label>📸 Click to upload image for face detection</label>
                </div>
                
                <button class="btn" onclick="detectFaces()" id="face-detect-btn" disabled>Detect Faces</button>
                
                <div class="results" id="face-results"></div>
            </div>
            
            <div class="feature-card">
                <h3>💄 Makeup Transfer</h3>
                <p>Apply various makeup styles with adjustable intensity using advanced AI algorithms.</p>
                
                <div class="upload-area" onclick="document.getElementById('makeup-upload').click()">
                    <input type="file" id="makeup-upload" accept="image/*" onchange="handleMakeupUpload(event)">
                    <label>📸 Click to upload image for makeup</label>
                </div>
                
                <div class="controls">
                    <div class="control-group">
                        <label>Style:</label>
                        <select id="makeup-style">
                            <option value="natural">Natural</option>
                            <option value="casual">Casual</option>
                            <option value="evening">Evening</option>
                            <option value="glamorous">Glamorous</option>
                            <option value="party">Party</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Intensity:</label>
                        <input type="range" id="makeup-intensity" min="0" max="1" step="0.1" value="0.7">
                        <span id="makeup-intensity-value">0.7</span>
                    </div>
                </div>
                
                <button class="btn" onclick="applyMakeup()" id="makeup-btn" disabled>Apply Makeup</button>
                
                <div class="results" id="makeup-results"></div>
            </div>
            
            <div class="feature-card">
                <h3>💇 Hair Transformation</h3>
                <p>Transform hair style and color with realistic AI-powered effects.</p>
                
                <div class="upload-area" onclick="document.getElementById('hair-upload').click()">
                    <input type="file" id="hair-upload" accept="image/*" onchange="handleHairUpload(event)">
                    <label>📸 Click to upload image for hair transformation</label>
                </div>
                
                <div class="controls">
                    <div class="control-group">
                        <label>Style:</label>
                        <select id="hair-style">
                            <option value="straight">Straight</option>
                            <option value="wavy">Wavy</option>
                            <option value="curly">Curly</option>
                            <option value="coily">Coily</option>
                            <option value="braided">Braided</option>
                            <option value="updo">Updo</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Color:</label>
                        <select id="hair-color">
                            <option value="black">Black</option>
                            <option value="brown">Brown</option>
                            <option value="blonde">Blonde</option>
                            <option value="red">Red</option>
                            <option value="gray">Gray</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Intensity:</label>
                        <input type="range" id="hair-intensity" min="0" max="1" step="0.1" value="0.8">
                        <span id="hair-intensity-value">0.8</span>
                    </div>
                </div>
                
                <button class="btn" onclick="transformHair()" id="hair-btn" disabled>Transform Hair</button>
                
                <div class="results" id="hair-results"></div>
            </div>
            
            <div class="feature-card">
                <h3>✨ Comprehensive Beauty</h3>
                <p>Apply complete beauty transformation including makeup, hair style, and color changes.</p>
                
                <div class="upload-area" onclick="document.getElementById('beauty-upload').click()">
                    <input type="file" id="beauty-upload" accept="image/*" onchange="handleBeautyUpload(event)">
                    <label>📸 Click to upload image for full transformation</label>
                </div>
                
                <button class="btn" onclick="applyFullBeauty()" id="beauty-btn" disabled>Apply Full Transformation</button>
                
                <div class="results" id="beauty-results"></div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing your image with AI...</p>
        </div>
    </div>
    
    <script>
        let currentImages = {};
        
        // Handle file uploads
        function handleFaceUpload(event) {
            const file = event.target.files[0];
            if (file) {
                currentImages.face = file;
                document.getElementById('face-detect-btn').disabled = false;
                showPreview(file, 'face-results');
            }
        }
        
        function handleMakeupUpload(event) {
            const file = event.target.files[0];
            if (file) {
                currentImages.makeup = file;
                document.getElementById('makeup-btn').disabled = false;
                showPreview(file, 'makeup-results');
            }
        }
        
        function handleHairUpload(event) {
            const file = event.target.files[0];
            if (file) {
                currentImages.hair = file;
                document.getElementById('hair-btn').disabled = false;
                showPreview(file, 'hair-results');
            }
        }
        
        function handleBeautyUpload(event) {
            const file = event.target.files[0];
            if (file) {
                currentImages.beauty = file;
                document.getElementById('beauty-btn').disabled = false;
                showPreview(file, 'beauty-results');
            }
        }
        
        function showPreview(file, containerId) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const container = document.getElementById(containerId);
                container.innerHTML = `
                    <h4>Original Image:</h4>
                    <img src="${e.target.result}" class="image-preview" alt="Preview">
                `;
            };
            reader.readAsDataURL(file);
        }
        
        // Update intensity values
        document.getElementById('makeup-intensity').addEventListener('input', function(e) {
            document.getElementById('makeup-intensity-value').textContent = e.target.value;
        });
        
        document.getElementById('hair-intensity').addEventListener('input', function(e) {
            document.getElementById('hair-intensity-value').textContent = e.target.value;
        });
        
        // API calls
        async function detectFaces() {
            if (!currentImages.face) return;
            
            showLoading(true);
            const formData = new FormData();
            formData.append('image', currentImages.face);
            
            try {
                const response = await fetch('/api/face/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                showLoading(false);
                
                if (result.success) {
                    document.getElementById('face-results').innerHTML += `
                        <div class="status success">
                            ✅ Detected ${result.faces_detected} face(s) in ${result.processing_time.toFixed(2)}s
                        </div>
                    `;
                } else {
                    document.getElementById('face-results').innerHTML += `
                        <div class="status error">
                            ❌ Face detection failed: ${result.error || 'Unknown error'}
                        </div>
                    `;
                }
            } catch (error) {
                showLoading(false);
                document.getElementById('face-results').innerHTML += `
                    <div class="status error">
                        ❌ Error: ${error.message}
                    </div>
                `;
            }
        }
        
        async function applyMakeup() {
            if (!currentImages.makeup) return;
            
            showLoading(true);
            const formData = new FormData();
            formData.append('image', currentImages.makeup);
            formData.append('style', document.getElementById('makeup-style').value);
            formData.append('intensity', document.getElementById('makeup-intensity').value);
            
            try {
                const response = await fetch('/api/makeup/apply', {
                    method: 'POST',
                    body: formData
                });
                
                showLoading(false);
                
                if (response.ok) {
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    document.getElementById('makeup-results').innerHTML += `
                        <div class="status success">✅ Makeup applied successfully!</div>
                        <h4>Result:</h4>
                        <img src="${imageUrl}" class="image-preview" alt="Makeup Result">
                    `;
                } else {
                    document.getElementById('makeup-results').innerHTML += `
                        <div class="status error">❌ Makeup application failed</div>
                    `;
                }
            } catch (error) {
                showLoading(false);
                document.getElementById('makeup-results').innerHTML += `
                    <div class="status error">❌ Error: ${error.message}</div>
                `;
            }
        }
        
        async function transformHair() {
            if (!currentImages.hair) return;
            
            showLoading(true);
            const formData = new FormData();
            formData.append('image', currentImages.hair);
            formData.append('style', document.getElementById('hair-style').value);
            formData.append('intensity', document.getElementById('hair-intensity').value);
            
            try {
                const response = await fetch('/api/hair/style', {
                    method: 'POST',
                    body: formData
                });
                
                showLoading(false);
                
                if (response.ok) {
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    document.getElementById('hair-results').innerHTML += `
                        <div class="status success">✅ Hair transformed successfully!</div>
                        <h4>Result:</h4>
                        <img src="${imageUrl}" class="image-preview" alt="Hair Result">
                    `;
                } else {
                    document.getElementById('hair-results').innerHTML += `
                        <div class="status error">❌ Hair transformation failed</div>
                    `;
                }
            } catch (error) {
                showLoading(false);
                document.getElementById('hair-results').innerHTML += `
                    <div class="status error">❌ Error: ${error.message}</div>
                `;
            }
        }
        
        async function applyFullBeauty() {
            if (!currentImages.beauty) return;
            
            showLoading(true);
            const formData = new FormData();
            formData.append('image', currentImages.beauty);
            formData.append('makeup_style', 'evening');
            formData.append('makeup_intensity', '0.8');
            formData.append('hair_style', 'curly');
            formData.append('hair_intensity', '0.9');
            formData.append('hair_color', 'red');
            formData.append('color_intensity', '0.8');
            
            try {
                const response = await fetch('/api/beauty/full', {
                    method: 'POST',
                    body: formData
                });
                
                showLoading(false);
                
                if (response.ok) {
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    document.getElementById('beauty-results').innerHTML += `
                        <div class="status success">✅ Full beauty transformation completed!</div>
                        <h4>Result:</h4>
                        <img src="${imageUrl}" class="image-preview" alt="Beauty Result">
                    `;
                } else {
                    document.getElementById('beauty-results').innerHTML += `
                        <div class="status error">❌ Full beauty transformation failed</div>
                    `;
                }
            } catch (error) {
                showLoading(false);
                document.getElementById('beauty-results').innerHTML += `
                    <div class="status error">❌ Error: ${error.message}</div>
                `;
            }
        }
        
        function showLoading(show) {
            document.getElementById('loading').classList.toggle('show', show);
        }
        
        // Demo functions
        function startMediaPipeDemo() {
            alert('🚀 Starting MediaPipe Demo!\n\nThis will open the MediaPipe features demo.\nRun: python demo_mediapipe_features.py');
        }
        
        function startRealTimeProcessing() {
            alert('📹 Starting Real-time Video Processing!\n\nThis will open your webcam for real-time AI processing.\nRun: python demo_mediapipe_features.py and select option 3');
        }
        
        function startFaceMeshDemo() {
            alert('🎭 Starting Face Mesh Demo!\n\nThis will show 468 facial landmarks in real-time.\nRun: python demo_mediapipe_features.py and select option 2');
        }
        
        function startGestureControl() {
            alert('✋ Starting Gesture Control Demo!\n\nThis will enable hand gesture control for the beauty app.\nRun: python demo_mediapipe_features.py and select option 4');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/face/detect', methods=['POST'])
def detect_faces():
    """Face detection endpoint for web interface"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        # Process the image (simplified for demo)
        image_file = request.files['image']
        
        # Simulate processing time
        time.sleep(1)
        
        return jsonify({
            'success': True,
            'faces_detected': 1,  # Simplified for demo
            'processing_time': 1.0,
            'confidence': 0.95
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/makeup/apply', methods=['POST'])
def apply_makeup():
    """Makeup application endpoint for web interface"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        # Process the image (simplified for demo)
        image_file = request.files['image']
        
        # Simulate processing time
        time.sleep(2)
        
        # Return a sample result image
        return send_file('makeup_result_api.jpg', mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/hair/style', methods=['POST'])
def style_hair():
    """Hair styling endpoint for web interface"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        # Process the image (simplified for demo)
        image_file = request.files['image']
        
        # Simulate processing time
        time.sleep(2)
        
        # Return a sample result image
        return send_file('hair_style_result_api.jpg', mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/beauty/full', methods=['POST'])
def full_beauty():
    """Full beauty transformation endpoint for web interface"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        # Process the image (simplified for demo)
        image_file = request.files['image']
        
        # Simulate processing time
        time.sleep(3)
        
        # Return a sample result image
        return send_file('full_beauty_result_api.jpg', mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🌐 Starting AI Beauty Platform Web Interface...")
    print("📱 Open your browser and go to: http://localhost:8080")
    print("🚀 The web interface will demonstrate all AI features!")
    app.run(host='0.0.0.0', port=8080, debug=True)
