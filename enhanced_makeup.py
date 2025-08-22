import cv2
import argparse
import os
import numpy as np
from utils import *


# Available features for makeup only
available_features = [
    "LIP_LOWER",
    "LIP_UPPER", 
    "EYEBROW_LEFT",
    "EYEBROW_RIGHT",
    "EYELINER_LEFT",
    "EYELINER_RIGHT",
    "EYESHADOW_LEFT",
    "EYESHADOW_RIGHT",
    "BLUSH_LEFT",
    "BLUSH_RIGHT"
]

# Predefined color palettes for makeup only
color_palettes = {
    "natural_lips": [0, 0, 255],  # Red
    "pink_lips": [147, 20, 255],  # Pink
    "purple_lips": [255, 0, 255],  # Purple
    
    "black_eyeliner": [139, 0, 0],  # Dark Blue
    "brown_eyeliner": [19, 69, 139],  # Dark Brown
    
    "brown_eyeshadow": [19, 69, 139],  # Dark Brown
    "purple_eyeshadow": [255, 0, 255],  # Purple
    "gold_eyeshadow": [0, 215, 255],  # Gold
    
    "brown_eyebrows": [19, 69, 139],  # Dark Brown
    "black_eyebrows": [0, 0, 0],  # Black
    
    "pink_blush": [147, 20, 255],  # Pink
    "peach_blush": [0, 165, 255]  # Peach
}

# Default makeup configuration
default_config = {
    "LIP_UPPER": "natural_lips",
    "LIP_LOWER": "natural_lips", 
    "EYEBROW_LEFT": "brown_eyebrows",
    "EYEBROW_RIGHT": "brown_eyebrows",
    "EYELINER_LEFT": "black_eyeliner",
    "EYELINER_RIGHT": "black_eyeliner",
    "EYESHADOW_LEFT": "brown_eyeshadow",
    "EYESHADOW_RIGHT": "brown_eyeshadow",
    "BLUSH_LEFT": "pink_blush",
    "BLUSH_RIGHT": "pink_blush"
}


def apply_makeup(image_path, config=None, save_output=True, show_result=True):
    """
    Apply makeup to an image
    
    Args:
        image_path: Path to input image
        config: Dictionary with feature:color_mapping
        save_output: Whether to save the result
        show_result: Whether to display the result
    """
    if config is None:
        config = default_config
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Create empty mask
    makeup_mask = np.zeros_like(image)
    
    # Extract facial landmarks
    try:
        face_landmarks = read_landmarks(image=image)
    except Exception as e:
        print(f"Error detecting face landmarks: {e}")
        return None
    
    # Apply makeup features
    for feature, color_name in config.items():
        if feature in face_points and color_name in color_palettes:
            # Get the landmark points for this feature
            feature_points = face_points[feature]
            color = color_palettes[color_name]
            
            # Create points array for this feature
            points = []
            for idx in feature_points:
                if idx in face_landmarks:
                    points.append(face_landmarks[idx])
            
            if len(points) > 2:  # Need at least 3 points for a polygon
                points = np.array(points, dtype=np.int32)
                
                # Fill the feature with color
                cv2.fillPoly(makeup_mask, [points], color)
    
    # Apply smoothing to makeup mask
    makeup_mask = cv2.GaussianBlur(makeup_mask, (7, 7), 4)
    
    # Blend with original image
    output = cv2.addWeighted(image, 1.0, makeup_mask, 0.3, 0)
    
    # Save output
    if save_output:
        input_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(input_filename)
        output_filename = f"{name}_makeup{ext}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, output)
        print(f"Makeup image saved to: {output_path}")
    
    # Show result
    if show_result:
        show_image(output, "Makeup Image")
    
    return output


def print_available_options():
    """Print available features and colors"""
    print("\n=== Available Makeup Features ===")
    for feature in available_features:
        print(f"  - {feature}")
    
    print("\n=== Available Colors ===")
    for color_name in color_palettes.keys():
        print(f"  - {color_name}")


def main():
    parser = argparse.ArgumentParser(description="Virtual Makeup Application")
    parser.add_argument("--img", type=str, required=True, help="Path to the image")
    parser.add_argument("--lip_color", type=str, help="Lip color (e.g., natural_lips, pink_lips, purple_lips)")
    parser.add_argument("--eyeshadow_color", type=str, help="Eyeshadow color (e.g., brown_eyeshadow, purple_eyeshadow)")
    parser.add_argument("--list_options", action="store_true", help="List all available features and colors")
    parser.add_argument("--no_save", action="store_true", help="Don't save the output image")
    parser.add_argument("--no_show", action="store_true", help="Don't show the result window")
    
    args = parser.parse_args()
    
    if args.list_options:
        print_available_options()
        return
    
    # Create custom configuration
    config = default_config.copy()
    
    if args.lip_color:
        if args.lip_color in color_palettes:
            config["LIP_UPPER"] = args.lip_color
            config["LIP_LOWER"] = args.lip_color
        else:
            print(f"Warning: Unknown lip color '{args.lip_color}'. Using default.")
    
    if args.eyeshadow_color:
        if args.eyeshadow_color in color_palettes:
            config["EYESHADOW_LEFT"] = args.eyeshadow_color
            config["EYESHADOW_RIGHT"] = args.eyeshadow_color
        else:
            print(f"Warning: Unknown eyeshadow color '{args.eyeshadow_color}'. Using default.")
    
    # Apply makeup
    apply_makeup(
        args.img, 
        config=config,
        save_output=not args.no_save,
        show_result=not args.no_show
    )


if __name__ == "__main__":
    main() 