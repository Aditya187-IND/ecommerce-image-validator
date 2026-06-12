import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

class ECommerceValidator:
    def __init__(self):
        print("Loading AI Engine...")
        self.model = YOLO('yolov8n.pt') 
        
        self.MIN_RESOLUTION = (500, 500)
        self.BLUR_THRESHOLD = 100.0      
        self.BANNED_BACKGROUND_OBJECTS = ['person', 'dog', 'cat', 'cell phone', 'car'] 

    def extract_dominant_colors(self, image_path, k=5):
        """Uses K-Means clustering to find the dominant hex colors in the image."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to standard Web RGB
        
        # Reshape image to a list of pixels
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)

        # Define criteria and apply K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers back to 8-bit values and then to Hex codes
        centers = np.uint8(centers)
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in centers]
        return hex_colors

    def analyze_image(self, image_path):
        if not os.path.exists(image_path):
            return {"status": "ERROR", "reasons": [f"File not found: {image_path}"]}

        img = cv2.imread(image_path)
        height, width, _ = img.shape
        file_size_mb = round(os.path.getsize(image_path) / (1024 * 1024), 2)
        
        rejection_reasons = []
        
        # --- Checklist Booleans ---
        passed_resolution = True
        passed_focus = True
        passed_lighting = True
        passed_background = True

        # 1. Image Property Checks
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            passed_resolution = False
            rejection_reasons.append(f"Resolution too low ({width}x{height}). Minimum is {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]}.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.BLUR_THRESHOLD:
            passed_focus = False
            rejection_reasons.append(f"Image is too blurry (Score: {round(blur_score, 1)}).")

        brightness = np.mean(gray)
        if not (50 < brightness <= 255):
            passed_lighting = False
            rejection_reasons.append("Lighting is completely incorrect (Too dark).")

        # 2. AI Object Detection Checks
        results = self.model(image_path, verbose=False)
        detected_items = []
        
        for result in results:
            annotated_image = result.plot()
            output_name = f"scanned_{os.path.basename(image_path)}"
            cv2.imwrite(output_name, annotated_image)
            
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                detected_items.append(class_name)
                
                if class_name in self.BANNED_BACKGROUND_OBJECTS:
                    passed_background = False
                    rejection_reasons.append(f"Banned object detected: {class_name.upper()}")

        # 3. Extract Dominant Colors
        dominant_colors = self.extract_dominant_colors(image_path)

        # 4. Compile Data
        report = {
            "status": "REJECTED ❌" if rejection_reasons else "APPROVED ✅",
            "detected": list(set(detected_items)),
            "reasons": rejection_reasons if rejection_reasons else ["Meets all quality standards."],
            "output_file": output_name,
            "metrics": {
                "Resolution": f"{width}x{height} px",
                "Sharpness": round(blur_score, 1),
                "Brightness": round(brightness, 1),
                "File Size": f"{file_size_mb} MB"
            },
            "checklist": {
                "Resolution": passed_resolution,
                "Focus & Sharpness": passed_focus,
                "Lighting & Exposure": passed_lighting,
                "Clean Background": passed_background
            },
            "colors": dominant_colors
        }
        return report

# === THE COMMAND LINE INTERFACE ===
def main():
    parser = argparse.ArgumentParser(description="E-Commerce AI Image Quality Gatekeeper")
    parser.add_argument("image", help="The path to the image file")
    args = parser.parse_args()

    validator = ECommerceValidator()
    print(f"\n--- SCANNING: {args.image} ---")
    report = validator.analyze_image(args.image)
    
    print(f"\nFINAL VERDICT: {report['status']}")
    print(f"Colors Extracted: {', '.join(report['colors'])}")
    print("\nFeedback:")
    for reason in report.get('reasons', []):
        print(f" - {reason}")
        
if __name__ == "__main__":
    main()
