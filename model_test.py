import cv2
import numpy as np
from ultralytics import YOLO
import time
import joblib
import keyboard
import pandas as pd
import threading
import torch
import vgamepad as vg

class ObjectDetector:
    def __init__(self):
        # Check for GPU availability and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load paths for models
        self.model_path = r"./svm_model.pkl"
        self.scaler_path = r"./scaler.pkl"
        
        # Constants
        self.REAL_WORLD_HEIGHTS = {"car": 1.5}  # Approximate real-world car height (meters)
        self.FOCAL_LENGTH = 300  # Focal length for the screen size
        
        # Speed control parameters
        self.BASE_SPEED = 0.35  # Base speed value (normalized between 0-1)
        self.MAX_SPEED = 0.50   # Maximum speed value
        self.MIN_SPEED = 0.27  # Minimum speed value
        self.SAFE_DISTANCE = 10.0  # Distance threshold for speed adjustment (meters)
        self.CRITICAL_DISTANCE = 5.0  # Distance threshold for significant speed reduction
        
        # Screen dimensions (GTA 5 at 800x600)
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        
        # Define lane boundaries (adjust these based on your game's view)
        self.LEFT_LANE_BOUNDARY = 175  # Left boundary of your lane
        self.RIGHT_LANE_BOUNDARY = 615  # Right boundary of your lane
        
        # Initialize virtual gamepad
        self.gamepad = vg.VX360Gamepad()
        
        # Load the YOLO model and ensure it's on GPU if available
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)
        
        # Load SVM model and scaler
        self.svm_model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Create a buffer for predictions to reduce jitter
        self.prediction_buffer = []
        self.buffer_size = 3
        
        # Current speed value
        self.current_speed = self.BASE_SPEED
        
        # Closest car distance tracking
        self.closest_car_distance = float('inf')
        
        # Action state
        self.current_action = None
        self.action_mapping = {0: "LEFT", 1: "BRAKE", 2: "RIGHT"}
        
        # Last detection time
        self.last_detection_time = time.time()
        
        # Running flag
        self.running = True

    def calculate_driving_params(self, speed):
        """Calculate driving parameters based on speed"""
        if speed <= 10:
            return 0.4, 0.6
        elif speed <= 100:
            return 0.4, 0.3
        elif speed <= 150:
            return 0.7, 0.8
        elif speed <= 200:
            return 1, 1.5
        else:
            return 0.8, 0.4

    def press_key_duration(self, key, duration):
        """Press a key for the specified duration"""
        keyboard.press(key)
        threading.Timer(duration, lambda: keyboard.release(key)).start()
        time.sleep(1)

    def set_speed(self, value):
        """Set the speed using the virtual gamepad"""
        # Apply the right trigger for acceleration
        self.gamepad.right_trigger(value=int(value * 255))
        self.gamepad.update()

    def estimate_distance(self, bbox_height, real_height):
        """Estimate distance based on bounding box height"""
        return (self.FOCAL_LENGTH * real_height) / bbox_height if bbox_height > 0 else 0
    
    def is_in_my_lane(self, x1, x2, y2):
        """Determine if a car is in my lane based on its position
        
        Parameters:
        x1, x2: The x-coordinates of the left and right sides of the bounding box
        y2: The y-coordinate of the bottom of the bounding box
        
        Returns:
        bool: True if the car is in my lane, False otherwise
        """
        # Calculate center of the car
        center_x = (x1 + x2) / 2
        
        # Check if the car is within lane boundaries
        is_in_boundaries = self.LEFT_LANE_BOUNDARY < center_x < self.RIGHT_LANE_BOUNDARY
        
        # The lower in the screen (higher y value), the more likely it's in my lane
        # This helps filter cars that are far away and might be in other lanes
        y_factor = y2 / self.SCREEN_HEIGHT  # 0 at top, 1 at bottom
        
        # Combine factors - cars lower in the screen have higher priority
        return is_in_boundaries and y_factor > 0.6

    def process_frame(self, frame):
        """Process a frame for object detection"""
        # Perform detection with YOLO
        results = self.model.predict(source=frame, conf=0.5, classes=[2,7,0], verbose=False)
        
        x_vals, y_vals, dist_vals = [], [], []
        detections = results[0].boxes.data.cpu().numpy()
        detected = False
        
        # Original frame for display
        display_frame = frame.copy()
        
        # Draw lane boundaries for visualization
        cv2.line(display_frame, (self.LEFT_LANE_BOUNDARY, 0), (self.LEFT_LANE_BOUNDARY, self.SCREEN_HEIGHT), (255, 0, 0), 2)
        cv2.line(display_frame, (self.RIGHT_LANE_BOUNDARY, 0), (self.RIGHT_LANE_BOUNDARY, self.SCREEN_HEIGHT), (255, 0, 0), 2)
        
        # Reset closest car distance for this frame
        self.closest_car_distance = float('inf')
        
        # Filter detections to only include cars in my lane
        my_lane_detections = []
        
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            if confidence < 0.5:
                continue
            
            # Check if this is a car in my lane
            if self.is_in_my_lane(x1, x2, y2):
                my_lane_detections.append(det)
                bbox_height = y2 - y1
                distance = self.estimate_distance(bbox_height, self.REAL_WORLD_HEIGHTS["car"])
                
                # Update closest car distance
                if distance < self.closest_car_distance:
                    self.closest_car_distance = distance
                
                # Draw green rectangle for cars in my lane
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(display_frame, f"My Lane: {distance:.1f}m", 
                           (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw red rectangle for cars not in my lane (for visualization only)
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(display_frame, "Other Lane", 
                           (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Process only cars in my lane
        if my_lane_detections:
            detected = True
            for det in my_lane_detections:
                x1, y1, x2, y2, confidence, class_id = det
                
                # Calculate center of bounding box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                
                bbox_height = y2 - y1
                distance = self.estimate_distance(bbox_height, self.REAL_WORLD_HEIGHTS["car"])
                
                x_vals.append(x_center)
                y_vals.append(y_center)
                dist_vals.append(distance)
        
        # Adjust speed based on the closest car distance
        if self.closest_car_distance < float('inf'):
            if self.closest_car_distance < self.CRITICAL_DISTANCE:
                # Very close to a car - slow down significantly
                self.current_speed = self.MIN_SPEED
            elif self.closest_car_distance < self.SAFE_DISTANCE:
                # Scale speed based on distance within the safe range
                distance_factor = (self.closest_car_distance - self.CRITICAL_DISTANCE) / (self.SAFE_DISTANCE - self.CRITICAL_DISTANCE)
                self.current_speed = self.MIN_SPEED + distance_factor * (self.BASE_SPEED - self.MIN_SPEED)
            else:
                # No cars nearby - increase speed
                self.current_speed = min(self.current_speed + 0.02, self.MAX_SPEED)
        else:
            # No cars detected - maintain or increase to base speed
            if self.current_speed < self.BASE_SPEED:
                self.current_speed = min(self.current_speed + 0.01, self.BASE_SPEED)
            else:
                self.current_speed = self.BASE_SPEED
        
        # Apply the new speed
        self.set_speed(self.current_speed)
        
        # Process detection results
        self.current_action = None
        if detected and x_vals and y_vals and dist_vals:
            avg_distance = sum(dist_vals) / len(dist_vals)
            x_avg = sum(x_vals) / len(x_vals)
            y_avg = sum(y_vals) / len(y_vals)
            
            # Adjust fine-tuning parameters based on screen size
            fine_tune_x = 83.33 if x_avg < 400 else -83.33  # Reduced offset for more accurate centering
            fine_tune_y = 19.44
            fine_tune_distance = 1.5
            
            if 1 < avg_distance < 6.0:
                features = pd.DataFrame([[x_avg + fine_tune_x, y_avg + fine_tune_y, avg_distance - fine_tune_distance]], 
                                       columns=['Avg_X', 'Avg_Y', 'Avg_Distance'])
                features_scaled = self.scaler.transform(features)
                current_prediction = self.svm_model.predict(features_scaled)[0]
                
                # Add to prediction buffer
                self.prediction_buffer.append(current_prediction)
                if len(self.prediction_buffer) > self.buffer_size:
                    self.prediction_buffer.pop(0)
                
                # Use most common prediction from buffer
                if self.prediction_buffer:
                    from collections import Counter
                    labels = Counter(self.prediction_buffer).most_common(1)[0][0]
                    
                    # Add visual feedback for prediction
                    action_text = {0: "LEFT (A)", 1: "BRAKE (Space)", 2: "RIGHT (D)"}
                    cv2.putText(display_frame, f"Action: {action_text.get(labels, 'Unknown')}", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Store the action
                    self.current_action = self.action_mapping.get(labels)
                    self.last_detection_time = time.time()
        
        # Display speed information
        cv2.putText(display_frame, f"Speed: {self.current_speed:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Closest Car: {self.closest_car_distance:.1f}m", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"My Lane Cars: {len(my_lane_detections)}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame

    def get_current_action(self):
        """Get the current recommended action"""
        # Return None if the last detection is too old (more than 1 second)
        if time.time() - self.last_detection_time > 1.0:
            return None
        return self.current_action
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.gamepad.reset()

