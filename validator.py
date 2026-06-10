import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

class ECommerceValidator:
    def __init__(self):
        print("Loading AI Engine...")
        self.model = YOLO('yolov8n.pt') 
        
        # --- STRICT E-COMMERCE RULES ---
        self.MIN_RESOLUTION = (500, 500)
        self.BLUR_THRESHOLD = 100.0      
        # If any of these are detected, the image is instantly rejected
        self.BANNED_BACKGROUND_OBJECTS = ['person', 'dog', 'cat', 'cell phone', 'car'] 

    def analyze_image(self, image_path):
        """Analyzes properties and objects, then makes a strict Pass/Fail decision."""
        if not os.path.exists(image_path):
            return {"status": "ERROR", "reasons": [f"File not found: {image_path}"]}

        img = cv2.imread(image_path)
        height, width, _ = img.shape
        
        rejection_reasons = []

        # 1. Image Property Checks
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            rejection_reasons.append(f"Resolution too low ({width}x{height}). Minimum is {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]}.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.BLUR_THRESHOLD:
            rejection_reasons.append(f"Image is too blurry (Score: {round(blur_score, 1)}).")

        brightness = np.mean(gray)
        if not (50 < brightness <= 255):
            rejection_reasons.append("Lighting is completely incorrect (Too dark).")

        # 2. AI Object Detection Checks
        results = self.model(image_path, verbose=False) # verbose=False hides YOLO's default spam
        detected_items = []
        
        for result in results:
            # Save the visual proof
            annotated_image = result.plot()
            output_name = f"scanned_{os.path.basename(image_path)}"
            cv2.imwrite(output_name, annotated_image)
            
            # Check for banned objects
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                detected_items.append(class_name)
                
                if class_name in self.BANNED_BACKGROUND_OBJECTS:
                    rejection_reasons.append(f"Banned object detected in background: {class_name.upper()}")

        # 3. Final Decision Logic
        if len(rejection_reasons) > 0:
            return {
                "status": "REJECTED ❌",
                "detected": list(set(detected_items)),
                "reasons": rejection_reasons,
                "output_file": output_name
            }
        else:
            return {
                "status": "APPROVED ✅",
                "detected": list(set(detected_items)),
                "reasons": ["Meets all quality standards."],
                "output_file": output_name
            }

# === THE COMMAND LINE INTERFACE ===
def main():
    # This block allows you to run the tool professionally from the terminal
    parser = argparse.ArgumentParser(description="E-Commerce AI Image Quality Gatekeeper")
    parser.add_argument("image", help="The path to the image file you want to test")
    args = parser.parse_args()

    validator = ECommerceValidator()
    
    print(f"\n--- SCANNING: {args.image} ---")
    report = validator.analyze_image(args.image)
    
    # Print the sleek terminal report
    print(f"\nFINAL VERDICT: {report['status']}")
    print(f"Objects Detected: {', '.join(report.get('detected', [])) if report.get('detected') else 'None'}")
    
    print("\nFeedback:")
    for reason in report.get('reasons', []):
        print(f" - {reason}")
        
    if "output_file" in report:
        print(f"\n[INFO] Visual proof saved as: {report['output_file']}")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    main()