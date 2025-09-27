import cv2 as cv
from PIL import Image
import numpy as np

def getlimits(color_bgr):
    """
    Convert BGR color to HSV and return upper and lower limits for color detection
    """
    # Convert single BGR color to HSV
    color = np.uint8([[color_bgr]])   # Wrap into numpy array
    hsv = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    
    # Extract hue value
    hue = hsv[0][0][0]
    
    # Define lower and upper limits with +/- 10 hue range
    # Handle hue wrapping (0-179 in OpenCV)
    lower_hue = max(0, hue - 10)
    upper_hue = min(179, hue + 10)
    
    lower_limit = np.array([lower_hue, 100, 100], dtype=np.uint8)
    upper_limit = np.array([upper_hue, 255, 255], dtype=np.uint8)
    
    return lower_limit, upper_limit

def detect_color_realtime():
    """
    Real-time color detection using webcam
    """
    # Define the color to detect (BGR format)
    blue = [139, 69, 19]  # This is actually more of a brown color
    # For actual blue, try: [255, 0, 0] or [200, 100, 50]
    
    # Initialize video capture
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    print("Press 'c' to change color detection")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Convert frame to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Get color limits
        lower_limits, upper_limits = getlimits(blue)
        
        # Create mask for the specified color
        mask = cv.inRange(hsv_frame, lower_limits, upper_limits)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        # Convert mask to PIL Image to get bounding box
        mask_pil = Image.fromarray(mask)
        bbox = mask_pil.getbbox()
        
        # Draw bounding box if object detected
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle around detected object
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text label
            cv.putText(frame, f'Color Detected', (x1, y1-10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate and display area
            area = (x2 - x1) * (y2 - y1)
            cv.putText(frame, f'Area: {area}', (x1, y2+20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the original frame and mask
        cv.imshow('Color Detection', frame)
        cv.imshow('Mask', mask)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Click on the video window and press a key to select new color:")
            print("'r' for red, 'g' for green, 'b' for blue, 'y' for yellow")
            color_key = cv.waitKey(0) & 0xFF
            if color_key == ord('r'):
                blue = [0, 0, 255]  # Red
                print("Switched to red detection")
            elif color_key == ord('g'):
                blue = [0, 255, 0]  # Green
                print("Switched to green detection")
            elif color_key == ord('b'):
                blue = [255, 0, 0]  # Blue
                print("Switched to blue detection")
            elif color_key == ord('y'):
                blue = [0, 255, 255]  # Yellow
                print("Switched to yellow detection")
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()

def detect_color_in_image(image_path, color_bgr):
    """
    Detect color in a static image
    """
    # Read image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Get color limits
    lower_limits, upper_limits = getlimits(color_bgr)
    
    # Create mask
    mask = cv.inRange(hsv, lower_limits, upper_limits)
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    result = image.copy()
    for contour in contours:
        if cv.contourArea(contour) > 500:  # Filter small objects
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display results
    cv.imshow('Original', image)
    cv.imshow('Mask', mask)
    cv.imshow('Detection Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    print("Color Detection Project")
    print("1. Real-time detection")
    print("2. Image detection")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        detect_color_realtime()
    elif choice == "2":
        image_path = input("Enter image path: ")
        # Example: blue color in BGR
        color = [255, 0, 0]
        detect_color_in_image(image_path, color)
    else:
        print("Invalid choice. Running real-time detection...")
        detect_color_realtime()
