#make .csv from labeld images
import os
import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Use custom-trained model if available

# Constants
REAL_WORLD_HEIGHTS = {"car": 1.5}  # Approximate real-world car height (meters)
FOCAL_LENGTH = 300  # Assumed focal length in pixels (tunable)

# Function to estimate distance
def estimate_distance(bbox_height, real_height):
    return (FOCAL_LENGTH * real_height) / bbox_height if bbox_height > 0 else 0

# Process images in a given folder
def process_images(folder_path):
    if not os.path.exists(folder_path):
        print("❌ Folder does not exist!")
        return

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))]
    if not images:
        print("⚠️ No images found in the folder!")
        return

    csv_file = os.path.join(folder_path, "s_adas_dataset.csv")
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Image", "Avg_X", "Avg_Y", "Avg_Height", "Avg_Width", "Avg_Distance", "Label"])

        for image_name in images:
            img_path = os.path.join(folder_path, image_name)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"⚠️ Skipping {image_name} (unable to read)")
                continue

            img_height, img_width = frame.shape[:2]
            results = model.predict(source=frame, conf=0.5, device='cpu')

            # Check if results are None
            if results is None or len(results) == 0:
                print(f"🚫 No detections for {image_name}")
                continue

            detections = [det for det in results[0].boxes.data.cpu().numpy() if model.names[int(det[5])] == "car"] # type: ignore

            if not detections:
                print(f"🚫 No cars detected in {image_name}")
                continue

            x_vals, y_vals, width_vals, height_vals, dist_vals = [], [], [], [], []

            for det in detections:
                x1, y1, x2, y2, _, _ = det
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                bbox_width = (x2 - x1)
                bbox_height = (y2 - y1)
                distance = estimate_distance(bbox_height, REAL_WORLD_HEIGHTS["car"])
                if distance > 4.5 :
                    continue
                x_vals.append(x_center)
                y_vals.append(y_center)
                width_vals.append(bbox_width)
                height_vals.append(bbox_height)
                dist_vals.append(distance)

            # Compute averages
            avg_x = sum(x_vals) / (len(x_vals)+1e-10)
            avg_y = sum(y_vals) / (len(y_vals)+1e-10)
            avg_width = sum(width_vals) / (len(width_vals)+1e-10)
            avg_height = sum(height_vals) / (len(height_vals)+1e-10)
            avg_distance = sum(dist_vals) / (len(dist_vals)+1e-10)

            # Append to CSV
            if(avg_distance < 5.5 and avg_distance > 0):
                writer.writerow([image_name, avg_x, avg_y, avg_height, avg_width, avg_distance, 0])
                print(f"✅ Processed: {image_name}")

    print(f"📁 CSV saved at: {csv_file}")

# Get folder input from user
# folder_name = input("Enter the folder name containing images: ").strip()
process_images("a-label")
