#!/usr/bin/env python3
import logging
import cv2
import requests
import json
import io
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # Load image
    image_path = "demo/animals.jpg"
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image from {image_path}")
        return
    
    # Prepare prompt
    text_prompt = "dog.raccoon.butterfly.salamander.fish"
    
    # Encode image to bytes
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()
    
    # Send request to server
    try:
        # Create files dictionary with the correct parameter name 'file'
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        
        # Send the text_prompt as form data
        data = {"text_prompt": text_prompt}
        
        # Make the request
        response = requests.post(
            "http://localhost:8765/process-image/",  # Note the trailing slash
            files=files,
            data=data
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract and print bounding boxes
        if 'annotations' in result:  # Server returns 'annotations', not 'objects'
            for annotation in result['annotations']:
                if 'bbox' in annotation:
                    bbox = annotation['bbox']
                    label = annotation.get('label', 'Unknown')
                    confidence = annotation.get('confidence', 0)
                    logging.info(f"Object: Label = {label}, Confidence = {confidence:.2f}, Bbox = {bbox}")
                    
                    # Draw bounding box on image (bbox format is [x_center, y_center, width, height])
                    x_center, y_center, width, height = [float(coord) for coord in bbox]
                    
                    # Convert to pixel coordinates
                    img_height, img_width = image.shape[:2]
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    # Ensure coordinates are within image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_width, x2), min(img_height, y2)
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display image with bounding boxes
        cv2.imshow("Detected Objects", image)
        # Wait for 'q' key to quit
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()