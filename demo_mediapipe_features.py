"""
MediaPipe Features Demo
Showcases real-time video processing, face mesh, and hand tracking
"""

import os
import sys
import cv2
import numpy as np
import time
from typing import Dict, List

# Add src directory to path
sys.path.append('src')

def demo_basic_mediapipe():
    """Demo basic MediaPipe face detection and hand tracking"""
    print("🎬 Basic MediaPipe Features Demo")
    print("=" * 50)
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe solutions
        mp_face_detection = mp.solutions.face_detection
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        # Initialize detectors
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.5
        )
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe solutions initialized")
        
        # Start webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open webcam")
            return
        
        print("📹 Webcam started - Press 'q' to quit, 'f' for face detection, 'h' for hands")
        
        # Processing modes
        show_face_detection = True
        show_hand_tracking = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                if show_face_detection:
                    face_results = face_detection.process(frame_rgb)
                    if face_results.detections:
                        for detection in face_results.detections:
                            mp_drawing.draw_detection(frame, detection)
                
                if show_hand_tracking:
                    hand_results = hands.process(frame_rgb)
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )
                
                # Add instructions overlay
                cv2.putText(frame, "Press 'q' to quit, 'f' for face, 'h' for hands", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('MediaPipe Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    show_face_detection = not show_face_detection
                    print(f"Face detection: {'ON' if show_face_detection else 'OFF'}")
                elif key == ord('h'):
                    show_hand_tracking = not show_hand_tracking
                    print(f"Hand tracking: {'ON' if show_hand_tracking else 'OFF'}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            face_detection.close()
            hands.close()
            print("✅ Demo completed")
            
    except ImportError as e:
        print(f"❌ MediaPipe not available: {e}")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_face_mesh():
    """Demo MediaPipe Face Mesh with 468 landmarks"""
    print("\n🎭 Face Mesh Demo (468 Landmarks)")
    print("=" * 50)
    
    try:
        import mediapipe as mp
        
        # Initialize Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ Face Mesh initialized")
        
        # Start webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open webcam")
            return
        
        print("📹 Webcam started - Press 'q' to quit, 't' to toggle tessellation, 'c' for contours")
        
        # Display modes
        show_tessellation = True
        show_contours = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with Face Mesh
                results = face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw tessellation (mesh)
                        if show_tessellation:
                            mp_drawing.draw_landmarks(
                                frame,
                                face_landmarks,
                                mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )
                        
                        # Draw contours
                        if show_contours:
                            mp_drawing.draw_landmarks(
                                frame,
                                face_landmarks,
                                mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                        
                        # Display landmark count
                        landmark_count = len(face_landmarks.landmark)
                        cv2.putText(frame, f"Landmarks: {landmark_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add instructions overlay
                cv2.putText(frame, "Press 'q' to quit, 't' for tessellation, 'c' for contours", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Face Mesh Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    show_tessellation = not show_tessellation
                    print(f"Tessellation: {'ON' if show_tessellation else 'OFF'}")
                elif key == ord('c'):
                    show_contours = not show_contours
                    print(f"Contours: {'ON' if show_contours else 'OFF'}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            face_mesh.close()
            print("✅ Face Mesh demo completed")
            
    except ImportError as e:
        print(f"❌ MediaPipe not available: {e}")
    except Exception as e:
        print(f"❌ Face Mesh demo failed: {e}")


def demo_beauty_enhancement():
    """Demo beauty enhancement with MediaPipe"""
    print("\n💄 Beauty Enhancement Demo")
    print("=" * 50)
    
    try:
        from mediapipe.real_time_processor import BeautyEnhancementProcessor
        
        # Initialize beauty processor
        processor = BeautyEnhancementProcessor()
        print("✅ Beauty Enhancement processor initialized")
        
        # Set up callbacks
        def on_face_detected(detection, keypoints):
            confidence = detection.score[0]
            if confidence > 0.8:
                print(f"✨ High-confidence face detected: {confidence:.3f}")
        
        def on_face_mesh(landmarks):
            landmark_count = len(landmarks.landmark)
            print(f"🎭 Face mesh detected with {landmark_count} landmarks")
        
        def on_hand_tracked(landmarks):
            landmark_count = len(landmarks.landmark)
            print(f"✋ Hand tracked with {landmark_count} landmarks")
        
        processor.set_face_detection_callback(on_face_detected)
        processor.set_face_mesh_callback(on_face_mesh)
        processor.set_hand_tracking_callback(on_hand_tracked)
        
        print("📹 Starting beauty enhancement webcam - Press 'q' to quit")
        print("🎨 Beauty effects: Skin smoothing, Eye enhancement, Lip enhancement")
        
        # Start webcam processing
        processor.start_webcam_processing(window_name="Beauty Enhancement")
        
        # Cleanup
        processor.cleanup()
        print("✅ Beauty enhancement demo completed")
        
    except ImportError as e:
        print(f"❌ Beauty enhancement not available: {e}")
    except Exception as e:
        print(f"❌ Beauty enhancement demo failed: {e}")


def demo_gesture_control():
    """Demo hand gesture recognition for beauty app control"""
    print("\n✋ Gesture Control Demo")
    print("=" * 50)
    
    try:
        import mediapipe as mp
        
        # Initialize hand tracking
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        print("✅ Hand tracking initialized")
        print("🎯 Gestures:")
        print("   - ✋ Open palm: Increase makeup intensity")
        print("   - ✊ Closed fist: Decrease makeup intensity")
        print("   - 👆 Point up: Next makeup style")
        print("   - 👇 Point down: Previous makeup style")
        
        # Start webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open webcam")
            return
        
        print("📹 Webcam started - Press 'q' to quit")
        
        # Gesture state
        current_intensity = 0.5
        current_style = 0
        styles = ['Natural', 'Glamorous', 'Casual', 'Evening', 'Party']
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with hand tracking
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Analyze gesture
                        gesture = analyze_hand_gesture(hand_landmarks)
                        
                        # Apply gesture actions
                        if gesture == 'open_palm':
                            current_intensity = min(1.0, current_intensity + 0.1)
                            print(f"🎨 Intensity increased to {current_intensity:.1f}")
                        elif gesture == 'closed_fist':
                            current_intensity = max(0.0, current_intensity - 0.1)
                            print(f"🎨 Intensity decreased to {current_intensity:.1f}")
                        elif gesture == 'point_up':
                            current_style = (current_style + 1) % len(styles)
                            print(f"💄 Style changed to {styles[current_style]}")
                        elif gesture == 'point_down':
                            current_style = (current_style - 1) % len(styles)
                            print(f"💄 Style changed to {styles[current_style]}")
                
                # Add status overlay
                cv2.putText(frame, f"Style: {styles[current_style]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Intensity: {current_intensity:.1f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Gesture Control Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            print("✅ Gesture control demo completed")
            
    except ImportError as e:
        print(f"❌ MediaPipe not available: {e}")
    except Exception as e:
        print(f"❌ Gesture control demo failed: {e}")


def analyze_hand_gesture(hand_landmarks):
    """Analyze hand landmarks to determine gesture"""
    try:
        # Get key landmark positions
        landmarks = hand_landmarks.landmark
        
        # Thumb tip (4), index tip (8), middle tip (12), ring tip (16), pinky tip (20)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Thumb base (2), index base (5), middle base (9), ring base (13), pinky base (17)
        thumb_base = landmarks[2]
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        pinky_base = landmarks[17]
        
        # Calculate distances from base to tip for each finger
        def finger_extended(tip, base):
            return tip.y < base.y  # Y increases downward
        
        # Check which fingers are extended
        thumb_extended = finger_extended(thumb_tip, thumb_base)
        index_extended = finger_extended(index_tip, index_base)
        middle_extended = finger_extended(middle_tip, middle_base)
        ring_extended = finger_extended(ring_tip, ring_base)
        pinky_extended = finger_extended(pinky_tip, pinky_base)
        
        # Determine gesture based on finger positions
        if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
            return 'open_palm'  # All fingers extended, thumb down
        elif not any([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
            return 'closed_fist'  # All fingers closed
        elif index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            if thumb_tip.y < thumb_base.y:  # Thumb pointing up
                return 'point_up'
            else:  # Thumb pointing down
                return 'point_down'
        
        return 'unknown'
        
    except Exception as e:
        print(f"Gesture analysis error: {e}")
        return 'unknown'


def main():
    """Main demo function"""
    print("🚀 MEDIAPIPE FEATURES DEMO")
    print("=" * 60)
    print("This demo showcases advanced MediaPipe features for AI Beauty Platform")
    print()
    
    while True:
        print("\n🎬 Available Demos:")
        print("1. Basic MediaPipe Features (Face Detection + Hand Tracking)")
        print("2. Face Mesh (468 Landmarks)")
        print("3. Beauty Enhancement (Real-time)")
        print("4. Gesture Control for Beauty App")
        print("5. Run All Demos")
        print("6. Exit")
        
        choice = input("\nSelect a demo (1-6): ").strip()
        
        if choice == '1':
            demo_basic_mediapipe()
        elif choice == '2':
            demo_face_mesh()
        elif choice == '3':
            demo_beauty_enhancement()
        elif choice == '4':
            demo_gesture_control()
        elif choice == '5':
            print("\n🎬 Running all demos in sequence...")
            demo_basic_mediapipe()
            time.sleep(2)
            demo_face_mesh()
            time.sleep(2)
            demo_beauty_enhancement()
            time.sleep(2)
            demo_gesture_control()
        elif choice == '6':
            print("👋 Thanks for trying the MediaPipe demos!")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
