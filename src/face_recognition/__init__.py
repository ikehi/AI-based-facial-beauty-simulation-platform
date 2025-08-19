"""
Face Recognition Module for AI Beauty Platform

This module provides face detection, landmark extraction, and face analysis
capabilities using OpenCV, Dlib, and FaceNet.
"""

from .face_detector import FaceDetector
from .landmark_extractor import LandmarkExtractor
from .face_analyzer import FaceAnalyzer

__all__ = ['FaceDetector', 'LandmarkExtractor', 'FaceAnalyzer'] 