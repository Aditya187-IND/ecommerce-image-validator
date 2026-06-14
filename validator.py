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

    def extract_color_data(self, image_path, k=5):
        """Extracts dominant colors and calculates their exact percentage in the image."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        
        # --- NEW: Calculate Percentages ---
        counts = np.bincount(labels.flatten())
        total_pixels = len(labels)
        
        color_data = []
        for i in range(k):
            hex_color = '#{:02x}{:02x}{:02x}'.format(centers[i][0], centers[i][1], centers[i][2])
            percent = round((counts[i] / total_pixels) * 100, 1)
            # Save RGB temporarily for brightness math
            color_data.append({"hex": hex_color, "percent": percent, "rgb": centers[i]})
            
        # Sort from highest percentage to lowest
        color_data = sorted(color_data, key=lambda x: x['percent'], reverse=True)
        return color_data

    def analyze_image(self, image_path):
        if not os.path.exists(image_path):
            return {"status": "ERROR", "reasons": [f"File not found: {image_path}"]}

        img = cv2.imread(image_path)
        height, width, _ = img.shape
        file_size_mb = round(os.path.getsize(image_path) / (1024 * 1024), 2)
        
        rejection_reasons = []
        suggestions = [] 
        
        passed_resolution = True
        passed_focus = True
        passed_lighting = True
        passed_background = True

        # 1. Standard Property Checks
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            passed_resolution = False
            rejection_reasons.append(f"Resolution too low ({width}x{height}).")
            suggestions.append("📷 **Resolution Fix:** Move your camera closer to the product.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.BLUR_THRESHOLD:
            passed_focus = False
            rejection_reasons.append(f"Image is too blurry (Score: {round(blur_score, 1)}).")
            suggestions.append("🔍 **Focus Fix:** Rest your arms on a table or use a tripod.")

        brightness = np.mean(gray)
        if not (50 < brightness <= 255):
            passed_lighting = False
            rejection_reasons.append("Lighting is completely incorrect (Too dark).")
            suggestions.append("☀️ **Lighting Fix:** Move your setup near a window for natural daylight.")

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
                    suggestions.append(f"🧹 **Background Fix:** Please remove the '{class_name}' from the background.")

        # 3. Extract Color Graph Data & Add Color Suggestions
        color_data = self.extract_color_data(image_path)
        dominant_color = color_data[0] # The #1 most used color (usually the background)
        
        if dominant_color['percent'] < 40.0:
            suggestions.append("🎨 **Color Tip:** Your background is too chaotic! Because no single color makes up more than 40% of the image, the product gets lost. Try putting it on a solid colored desk or paper.")
        else:
            # Check if the main background color is dark
            r, g, b = dominant_color['rgb']
            if (r+g+b) / 3 < 100: 
                suggestions.append(f"🎨 **Color Tip:** Your dominant background color ({dominant_color['hex']}) is very dark. E-commerce products sell 30% better on pure white or light backgrounds.")

        # Clean up data for the frontend
        clean_colors = [{"hex": c["hex"], "percent": c["percent"]} for c in color_data]

        report = {
            "status": "REJECTED ❌" if rejection_reasons else "APPROVED ✅",
            "detected": list(set(detected_items)),
            "reasons": rejection_reasons if rejection_reasons else ["Meets all quality standards."],
            "suggestions": list(set(suggestions)),
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
            "colors": clean_colors # Pass the percentage data to the frontend graph!
        }
        return report

def main():
    pass # Kept short for deployment

if __name__ == "__main__":
    main()
