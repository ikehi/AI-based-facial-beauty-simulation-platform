"""
Real-time Video Processing with MediaPipe
Integrates advanced MediaPipe features for live beauty transformation
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
import logging

# Setup MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

logger = logging.getLogger(__name__)

class RealTimeMediaPipeProcessor:
    """
    Real-time video processing using MediaPipe for advanced AI beauty features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the real-time processor"""
        self.config = config or {}
        self.logger = logger
        
        # Initialize MediaPipe solutions
        self.face_detection = None
        self.face_mesh = None
        self.hands = None
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'hands_detected': 0,
            'processing_times': [],
            'fps': 0.0
        }
        
        # Callbacks for real-time processing
        self.face_detection_callback = None
        self.face_mesh_callback = None
        self.hand_tracking_callback = None
        
        self._init_mediapipe_solutions()
        self.logger.info("Real-time MediaPipe processor initialized successfully")
    
    def _init_mediapipe_solutions(self):
        """Initialize MediaPipe solutions with optimal settings"""
        try:
            # Face Detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            
            # Face Mesh (468 landmarks)
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Hand Tracking
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.logger.info("All MediaPipe solutions initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe solutions: {e}")
            raise
    
    def set_face_detection_callback(self, callback: Callable):
        """Set callback for face detection results"""
        self.face_detection_callback = callback
    
    def set_face_mesh_callback(self, callback: Callable):
        """Set callback for face mesh results"""
        self.face_mesh_callback = callback
    
    def set_hand_tracking_callback(self, callback: Callable):
        """Set callback for hand tracking results"""
        self.hand_tracking_callback = callback
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with all MediaPipe features"""
        start_time = time.time()
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face detection
        face_results = self.face_detection.process(frame_rgb)
        
        # Process with face mesh
        face_mesh_results = self.face_mesh.process(frame_rgb)
        
        # Process with hand tracking
        hand_results = self.hands.process(frame_rgb)
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Apply results to frame
        frame_bgr = self._apply_face_detection(frame_bgr, face_results)
        frame_bgr = self._apply_face_mesh(frame_bgr, face_mesh_results)
        frame_bgr = self._apply_hand_tracking(frame_bgr, hand_results)
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time, face_results, hand_results)
        
        return frame_bgr
    
    def _apply_face_detection(self, frame: np.ndarray, results) -> np.ndarray:
        """Apply face detection results to frame"""
        if results.detections:
            self.stats['faces_detected'] = len(results.detections)
            
            for detection in results.detections:
                # Draw detection box
                mp_drawing.draw_detection(frame, detection)
                
                # Extract key points
                keypoints = []
                for keypoint in detection.location_data.relative_keypoints:
                    h, w, _ = frame.shape
                    x = int(keypoint.x * w)
                    y = int(keypoint.y * h)
                    keypoints.append((x, y))
                
                # Call callback if set
                if self.face_detection_callback:
                    self.face_detection_callback(detection, keypoints)
        
        return frame
    
    def _apply_face_mesh(self, frame: np.ndarray, results) -> np.ndarray:
        """Apply face mesh results to frame"""
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw contours
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Call callback if set
                if self.face_mesh_callback:
                    self.face_mesh_callback(face_landmarks)
        
        return frame
    
    def _apply_hand_tracking(self, frame: np.ndarray, results) -> np.ndarray:
        """Apply hand tracking results to frame"""
        if results.multi_hand_landmarks:
            self.stats['hands_detected'] = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Call callback if set
                if self.hand_tracking_callback:
                    self.hand_tracking_callback(hand_landmarks)
        
        return frame
    
    def _update_stats(self, processing_time: float, face_results, hand_results):
        """Update processing statistics"""
        self.stats['frames_processed'] += 1
        self.stats['processing_times'].append(processing_time)
        
        # Calculate FPS
        if len(self.stats['processing_times']) > 1:
            avg_time = np.mean(self.stats['processing_times'][-30:])  # Last 30 frames
            self.stats['fps'] = 1.0 / avg_time if avg_time > 0 else 0.0
        
        # Keep only last 100 processing times
        if len(self.stats['processing_times']) > 100:
            self.stats['processing_times'] = self.stats['processing_times'][-100:]
    
    def get_stats(self) -> Dict:
        """Get current processing statistics"""
        return self.stats.copy()
    
    def start_webcam_processing(self, camera_index: int = 0, window_name: str = "MediaPipe Beauty"):
        """Start real-time webcam processing"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_index}")
            return
        
        self.logger.info(f"Starting webcam processing on camera {camera_index}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add FPS and stats overlay
                self._add_stats_overlay(processed_frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Check for exit key (ESC or 'q')
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Webcam processing interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Webcam processing stopped")
    
    def _add_stats_overlay(self, frame: np.ndarray):
        """Add statistics overlay to frame"""
        # FPS counter
        fps_text = f"FPS: {self.stats['fps']:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Face count
        face_text = f"Faces: {self.stats['faces_detected']}"
        cv2.putText(frame, face_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Hand count
        hand_text = f"Hands: {self.stats['hands_detected']}"
        cv2.putText(frame, hand_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Frame count
        frame_text = f"Frames: {self.stats['frames_processed']}"
        cv2.putText(frame, frame_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if self.face_detection:
            self.face_detection.close()
        if self.face_mesh:
            self.face_mesh.close()
        if self.hands:
            self.hands.close()
        self.logger.info("MediaPipe resources cleaned up")


class BeautyEnhancementProcessor(RealTimeMediaPipeProcessor):
    """
    Specialized processor for beauty enhancement features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.beauty_effects = {
            'smooth_skin': True,
            'enhance_eyes': True,
            'lip_enhancement': True,
            'contour_highlighting': True
        }
    
    def _apply_beauty_effects(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Apply beauty enhancement effects based on face landmarks"""
        if not face_landmarks:
            return frame
        
        # Extract key facial regions
        # This is a simplified version - in production you'd use more sophisticated algorithms
        
        # Smooth skin effect
        if self.beauty_effects['smooth_skin']:
            frame = self._apply_skin_smoothing(frame, face_landmarks)
        
        # Enhance eyes
        if self.beauty_effects['enhance_eyes']:
            frame = self._enhance_eyes(frame, face_landmarks)
        
        # Lip enhancement
        if self.beauty_effects['lip_enhancement']:
            frame = self._enhance_lips(frame, face_landmarks)
        
        return frame
    
    def _apply_skin_smoothing(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Apply skin smoothing effect to detected face region"""
        # Simplified skin smoothing - in production use advanced algorithms
        h, w = frame.shape[:2]
        
        # Create a mask for the face region
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Extract face contour points (simplified)
        face_points = []
        for i in range(0, 27):  # Face contour landmarks
            if i < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[i]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append([x, y])
        
        if len(face_points) > 3:
            # Create face mask
            face_points = np.array(face_points, dtype=np.int32)
            cv2.fillPoly(mask, [face_points], 255)
            
            # Apply bilateral filter for skin smoothing
            smoothed = cv2.bilateralFilter(frame, 15, 80, 80)
            
            # Blend original and smoothed
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = frame * (1 - mask_3d) + smoothed * mask_3d
        
        return frame.astype(np.uint8)
    
    def _enhance_eyes(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Enhance eyes based on facial landmarks"""
        # Simplified eye enhancement
        h, w = frame.shape[:2]
        
        # Eye region landmarks (simplified)
        left_eye_center = None
        right_eye_center = None
        
        # Find eye centers (landmarks 159 and 386 are approximate eye centers)
        if len(face_landmarks.landmark) > 386:
            left_eye_center = face_landmarks.landmark[159]
            right_eye_center = face_landmarks.landmark[386]
        
        if left_eye_center and right_eye_center:
            # Apply subtle brightness enhancement to eye regions
            for eye_center in [left_eye_center, right_eye_center]:
                x, y = int(eye_center.x * w), int(eye_center.y * h)
                
                # Create circular mask around eye
                eye_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(eye_mask, (x, y), 20, 255, -1)
                
                # Enhance brightness in eye region
                eye_mask_3d = cv2.cvtColor(eye_mask, cv2.COLOR_GRAY2BGR) / 255.0
                enhanced = cv2.addWeighted(frame, 1.0, frame, 0.2, 10)
                
                frame = frame * (1 - eye_mask_3d) + enhanced * eye_mask_3d
        
        return frame.astype(np.uint8)
    
    def _enhance_lips(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Enhance lips based on facial landmarks"""
        # Simplified lip enhancement
        h, w = frame.shape[:2]
        
        # Lip landmarks (simplified - in production use more precise lip landmarks)
        if len(face_landmarks.landmark) > 13:
            # Approximate lip center
            lip_center = face_landmarks.landmark[13]
            x, y = int(lip_center.x * w), int(lip_center.y * h)
            
            # Create lip mask
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(lip_mask, (x, y), 25, 255, -1)
            
            # Apply subtle color enhancement
            lip_mask_3d = cv2.cvtColor(lip_mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # Enhance lip color (subtle red tint)
            enhanced = frame.copy()
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.1, 0, 255)  # Increase red channel
            
            frame = frame * (1 - lip_mask_3d) + enhanced * lip_mask_3d
        
        return frame.astype(np.uint8)


if __name__ == "__main__":
    # Example usage
    processor = BeautyEnhancementProcessor()
    
    # Set up callbacks
    def on_face_detected(detection, keypoints):
        print(f"Face detected with confidence: {detection.score[0]:.3f}")
    
    def on_face_mesh(landmarks):
        print(f"Face mesh with {len(landmarks.landmark)} landmarks")
    
    def on_hand_tracked(landmarks):
        print(f"Hand tracked with {len(landmarks.landmark)} landmarks")
    
    processor.set_face_detection_callback(on_face_detected)
    processor.set_face_mesh_callback(on_face_mesh)
    processor.set_hand_tracking_callback(on_hand_tracked)
    
    # Start webcam processing
    processor.start_webcam_processing()
    
    # Cleanup
    processor.cleanup()
