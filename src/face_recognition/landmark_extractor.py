"""
Facial Landmark Extraction using Dlib
"""

import dlib
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class LandmarkExtractor:
    """
    Extract facial landmarks using Dlib's 68-point facial landmark predictor
    """
    
    def __init__(self, predictor_path: str = 'models/shape_predictor_68_face_landmarks.dat'):
        """
        Initialize landmark extractor
        
        Args:
            predictor_path: Path to the facial landmark predictor model
        """
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            self.landmark_model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load landmark predictor: {e}")
            self.landmark_model_loaded = False
        
        # Define landmark regions
        self.landmark_regions = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'mouth': list(range(48, 68))
        }
    
    def extract_landmarks(self, image: np.ndarray, face_rect: dlib.rectangle) -> Optional[np.ndarray]:
        """
        Extract 68 facial landmarks from detected face
        
        Args:
            image: Input image
            face_rect: Face rectangle from dlib detector
            
        Returns:
            Array of 68 landmark points (x, y) or None if extraction fails
        """
        if not self.landmark_model_loaded:
            logger.error("Landmark predictor model not loaded")
            return None
        
        try:
            landmarks = self.predictor(image, face_rect)
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            return points
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def extract_landmarks_from_box(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract landmarks from face bounding box
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, width, height)
            
        Returns:
            Array of 68 landmark points (x, y) or None if extraction fails
        """
        x, y, w, h = face_box
        face_rect = dlib.rectangle(x, y, x + w, y + h)
        return self.extract_landmarks(image, face_rect)
    
    def get_landmark_region(self, landmarks: np.ndarray, region: str) -> np.ndarray:
        """
        Get landmarks for a specific facial region
        
        Args:
            landmarks: Full 68-point landmark array
            region: Region name ('jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye', 'left_eye', 'mouth')
            
        Returns:
            Landmarks for the specified region
        """
        if region not in self.landmark_regions:
            raise ValueError(f"Invalid region: {region}")
        
        indices = self.landmark_regions[region]
        return landmarks[indices]
    
    def get_eye_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get landmarks for both eyes
        
        Args:
            landmarks: Full 68-point landmark array
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' landmarks
        """
        return {
            'left_eye': self.get_landmark_region(landmarks, 'left_eye'),
            'right_eye': self.get_landmark_region(landmarks, 'right_eye')
        }
    
    def get_mouth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Get mouth landmarks
        
        Args:
            landmarks: Full 68-point landmark array
            
        Returns:
            Mouth landmarks
        """
        return self.get_landmark_region(landmarks, 'mouth')
    
    def get_eyebrow_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get landmarks for both eyebrows
        
        Args:
            landmarks: Full 68-point landmark array
            
        Returns:
            Dictionary with 'left_eyebrow' and 'right_eyebrow' landmarks
        """
        return {
            'left_eyebrow': self.get_landmark_region(landmarks, 'left_eyebrow'),
            'right_eyebrow': self.get_landmark_region(landmarks, 'right_eyebrow')
        }
    
    def calculate_face_angle(self, landmarks: np.ndarray) -> float:
        """
        Calculate face angle (yaw) using eye landmarks
        
        Args:
            landmarks: Full 68-point landmark array
            
        Returns:
            Face angle in degrees
        """
        left_eye = self.get_landmark_region(landmarks, 'left_eye')
        right_eye = self.get_landmark_region(landmarks, 'right_eye')
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Calculate angle
        eye_vector = right_eye_center - left_eye_center
        angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
        
        return angle
    
    def get_face_rect_from_landmarks(self, landmarks: np.ndarray, padding: float = 0.1) -> Tuple[int, int, int, int]:
        """
        Get face bounding rectangle from landmarks
        
        Args:
            landmarks: Full 68-point landmark array
            padding: Padding ratio around the face
            
        Returns:
            Face bounding box (x, y, width, height)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Add padding
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        return (x_min - pad_x, y_min - pad_y, width + 2 * pad_x, height + 2 * pad_y)
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      thickness: int = 2) -> np.ndarray:
        """
        Draw landmarks on image
        
        Args:
            image: Input image
            landmarks: 68-point landmark array
            color: BGR color for landmarks
            thickness: Line thickness
            
        Returns:
            Image with landmarks drawn
        """
        result = image.copy()
        
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(result, (x, y), 2, color, thickness)
        
        return result
    
    def draw_landmark_regions(self, image: np.ndarray, landmarks: np.ndarray, 
                            color: Tuple[int, int, int] = (0, 255, 0), 
                            thickness: int = 2) -> np.ndarray:
        """
        Draw landmark regions on image
        
        Args:
            image: Input image
            landmarks: 68-point landmark array
            color: BGR color for regions
            thickness: Line thickness
            
        Returns:
            Image with landmark regions drawn
        """
        result = image.copy()
        
        for region_name, indices in self.landmark_regions.items():
            region_points = landmarks[indices]
            
            # Draw region outline
            for i in range(len(region_points)):
                pt1 = tuple(map(int, region_points[i]))
                pt2 = tuple(map(int, region_points[(i + 1) % len(region_points)]))
                cv2.line(result, pt1, pt2, color, thickness)
        
        return result 