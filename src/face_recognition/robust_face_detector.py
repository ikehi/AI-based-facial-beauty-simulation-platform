"""
Robust Face Detection Module
Implements multiple detection methods for maximum reliability
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import time

# Try to import MediaPipe for more reliable face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available, using fallback methods")

# Try to import YOLO for advanced detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available, using fallback methods")

@dataclass
class FaceDetection:
    """Face detection result data class"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None
    method: str = "unknown"
    quality_score: float = 0.0

class RobustFaceDetector:
    """
    Robust face detector using multiple detection methods
    Falls back gracefully when models are unavailable
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the robust face detector"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Detection methods
        self.mediapipe_detector = None
        self.yolo_detector = None
        self.opencv_cascade = None
        self.dnn_detector = None
        
        # Initialize detection methods
        self._init_detection_methods()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'method_success': {},
            'average_confidence': 0.0,
            'processing_times': []
        }
    
    def _init_detection_methods(self):
        """Initialize all available detection methods"""
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 0 for short-range, 1 for full-range
                    min_detection_confidence=0.5
                )
                self.logger.info("MediaPipe face detection initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MediaPipe: {e}")
                self.mediapipe_detector = None
        
        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                # Use YOLOv8n-face model for face detection
                self.yolo_detector = YOLO('yolov8n-face.pt')
                self.logger.info("YOLO face detection initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize YOLO: {e}")
                self.yolo_detector = None
        
        # Initialize OpenCV Haar Cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.opencv_cascade = cv2.CascadeClassifier(cascade_path)
            if self.opencv_cascade.empty():
                raise Exception("Failed to load Haar cascade")
            self.logger.info("OpenCV Haar Cascade initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenCV Haar Cascade: {e}")
            self.opencv_cascade = None
        
        # Initialize OpenCV DNN
        try:
            # Use OpenCV's face detection model
            model_path = "models/face_recognition/opencv_face_detector_uint8.pb"
            config_path = "models/face_recognition/opencv_face_detector.pbtxt"
            
            # Try to load the model
            if self._file_exists(model_path) and self._file_exists(config_path):
                self.dnn_detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                self.logger.info("OpenCV DNN face detection initialized successfully")
            else:
                self.logger.warning("OpenCV DNN model files not found")
                self.dnn_detector = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenCV DNN: {e}")
            self.dnn_detector = None
    
    def _file_exists(self, file_path: str) -> bool:
        """Check if a file exists"""
        import os
        return os.path.exists(file_path)
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using multiple methods and combine results
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        start_time = time.time()
        all_detections = []
        
        # Method 1: MediaPipe (most reliable)
        if self.mediapipe_detector:
            try:
                mp_detections = self._detect_mediapipe(image)
                all_detections.extend(mp_detections)
                self.detection_stats['method_success']['mediapipe'] = len(mp_detections)
            except Exception as e:
                self.logger.error(f"MediaPipe detection failed: {e}")
        
        # Method 2: YOLO
        if self.yolo_detector:
            try:
                yolo_detections = self._detect_yolo(image)
                all_detections.extend(yolo_detections)
                self.detection_stats['method_success']['yolo'] = len(yolo_detections)
            except Exception as e:
                self.logger.error(f"YOLO detection failed: {e}")
        
        # Method 3: OpenCV DNN
        if self.dnn_detector:
            try:
                dnn_detections = self._detect_opencv_dnn(image)
                all_detections.extend(dnn_detections)
                self.detection_stats['method_success']['opencv_dnn'] = len(dnn_detections)
            except Exception as e:
                self.logger.error(f"OpenCV DNN detection failed: {e}")
        
        # Method 4: OpenCV Haar Cascade (fallback)
        if self.opencv_cascade:
            try:
                cascade_detections = self._detect_opencv_cascade(image)
                all_detections.extend(cascade_detections)
                self.detection_stats['method_success']['opencv_cascade'] = len(cascade_detections)
            except Exception as e:
                self.logger.error(f"OpenCV Haar Cascade detection failed: {e}")
        
        # Combine and filter detections
        final_detections = self._combine_detections(all_detections, image)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.detection_stats['processing_times'].append(processing_time)
        self.detection_stats['total_detections'] += len(final_detections)
        
        if final_detections:
            avg_confidence = sum(d.confidence for d in final_detections) / len(final_detections)
            self.detection_stats['average_confidence'] = avg_confidence
        
        self.logger.info(f"Detected {len(final_detections)} face(s) in {processing_time:.3f}s")
        return final_detections
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MediaPipe"""
        if not self.mediapipe_detector:
            return []
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mediapipe_detector.process(rgb_image)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                confidence = detection.score[0]
                
                # Extract landmarks if available
                landmarks = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        landmarks.append((kp_x, kp_y))
                
                detections.append(FaceDetection(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    landmarks=landmarks,
                    method="mediapipe",
                    quality_score=self._calculate_face_quality(image, (x, y, width, height))
                ))
        
        return detections
    
    def _detect_yolo(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using YOLO"""
        if not self.yolo_detector:
            return []
        
        try:
            results = self.yolo_detector(image, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if box.cls == 0:  # Assuming face class is 0
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            x = int(x1)
                            y = int(y1)
                            width = int(x2 - x1)
                            height = int(y2 - y1)
                            
                            detections.append(FaceDetection(
                                bbox=(x, y, width, height),
                                confidence=confidence,
                                method="yolo",
                                quality_score=self._calculate_face_quality(image, (x, y, width, height))
                            ))
            
            return detections
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []
    
    def _detect_opencv_dnn(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV DNN"""
        if not self.dnn_detector:
            return []
        
        try:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
            self.dnn_detector.setInput(blob)
            detections = self.dnn_detector.forward()
            
            results = []
            h, w = image.shape[:2]
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    
                    x = x1
                    y = y1
                    width = x2 - x1
                    height = y2 - y1
                    
                    results.append(FaceDetection(
                        bbox=(x, y, width, height),
                        confidence=float(confidence),
                        method="opencv_dnn",
                        quality_score=self._calculate_face_quality(image, (x, y, width, height))
                    ))
            
            return results
        except Exception as e:
            self.logger.error(f"OpenCV DNN detection error: {e}")
            return []
    
    def _detect_opencv_cascade(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV Haar Cascade"""
        if not self.opencv_cascade:
            return []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.opencv_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            detections = []
            for (x, y, width, height) in faces:
                # Estimate confidence based on face size and position
                confidence = min(0.9, 0.5 + (width * height) / (image.shape[0] * image.shape[1]) * 0.4)
                
                detections.append(FaceDetection(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    method="opencv_cascade",
                    quality_score=self._calculate_face_quality(image, (x, y, width, height))
                ))
            
            return detections
        except Exception as e:
            self.logger.error(f"OpenCV Haar Cascade detection error: {e}")
            return []
    
    def _combine_detections(self, detections: List[FaceDetection], image: np.ndarray) -> List[FaceDetection]:
        """Combine detections from multiple methods and remove duplicates"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply non-maximum suppression
        final_detections = []
        for detection in detections:
            is_duplicate = False
            for final_detection in final_detections:
                if self._calculate_iou(detection.bbox, final_detection.bbox) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        return final_detections
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_face_quality(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate face quality score based on size, position, and clarity"""
        x, y, width, height = bbox
        h, w = image.shape[:2]
        
        # Size score (prefer larger faces)
        size_score = min(1.0, (width * height) / (w * h) * 10)
        
        # Position score (prefer centered faces)
        center_x = x + width / 2
        center_y = y + height / 2
        distance_from_center = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        position_score = 1.0 - (distance_from_center / max_distance)
        
        # Clarity score (using Laplacian variance)
        try:
            face_roi = image[y:y+height, x:x+width]
            if face_roi.size > 0:
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                clarity_score = min(1.0, cv2.Laplacian(gray_roi, cv2.CV_64F).var() / 1000)
            else:
                clarity_score = 0.0
        except:
            clarity_score = 0.0
        
        # Combined score
        quality_score = (size_score * 0.4 + position_score * 0.3 + clarity_score * 0.3)
        return min(1.0, quality_score)
    
    def get_largest_face(self, detections: List[FaceDetection]) -> Optional[FaceDetection]:
        """Get the largest detected face"""
        if not detections:
            return None
        
        return max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
    
    def crop_face(self, image: np.ndarray, detection: FaceDetection, 
                  margin: float = 0.2) -> np.ndarray:
        """Crop face from image with optional margin"""
        x, y, width, height = detection.bbox
        h, w = image.shape[:2]
        
        # Add margin
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Calculate crop coordinates
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + width + margin_x)
        y2 = min(h, y + height + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        stats = self.detection_stats.copy()
        
        # Calculate average processing time
        if stats['processing_times']:
            stats['average_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
        else:
            stats['average_processing_time'] = 0.0
        
        # Add method availability
        stats['methods_available'] = {
            'mediapipe': MEDIAPIPE_AVAILABLE and self.mediapipe_detector is not None,
            'yolo': YOLO_AVAILABLE and self.yolo_detector is not None,
            'opencv_dnn': self.dnn_detector is not None,
            'opencv_cascade': self.opencv_cascade is not None
        }
        
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        if self.mediapipe_detector:
            self.mediapipe_detector.close()
