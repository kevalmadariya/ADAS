import cv2
import numpy as np
import keyboard
import time

class LaneDetector:
    def __init__(self):
        # Constants
        self.lane_memory = None
        self.lane_memory_frames = 5  # Number of frames to remember lane position
        self.lane_frames_count = 0
        self.running = True
        self.action_priority = {"LEFT": 0, "RIGHT": 0, "BRAKE": 0}
        self.last_action_time = time.time()
        self.action_cooldown = 0.5  # Cooldown between actions in seconds
        self.lane_confidence = 0.0  # Confidence in lane detection
        
    def preprocess_image(self, image):
        """Preprocess image for lane detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges
    
    def region_of_interest(self, image):
        """Define region of interest for lane detection"""
        height, width = image.shape
        mask = np.zeros_like(image)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width // 2, int(height * 0.6))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(image, mask)
    
    def detect_lines(self, image):
        """Detect lines using Hough Transform"""
        lines = cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        return lines
    
    def draw_lines(self, image, lines):
        """Draw detected lanes on the image"""
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        return cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
    def analyze_lanes(self, lines, width):
        """Analyze lane positions and determine actions"""
        if lines is None:
            self.lane_confidence = 0.0
            return None
        
        left_count, right_count = 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 0
            if slope < 0:  
                left_count += 1
            elif slope > 0:  
                right_count += 1
        
        # Calculate confidence in lane detection
        total_lines = left_count + right_count 
        self.lane_confidence = min(1.0, total_lines / 10.0)  # Normalize to [0, 1]
        
        # Determine action based on lane position
        if left_count > right_count * 1.5:  # More left lanes visible
            return "RIGHT"
        elif right_count > left_count * 1.5:  # More right lanes visible
            return "LEFT"
        else:
            return None  # Stay centered
    
    def process_frame(self, frame):
        """Process a frame for lane detection"""
        # Create a copy of the frame for visualization
        display_frame = frame.copy()
        
        # Preprocess image
        edges = self.preprocess_image(frame)
        
        # Define region of interest
        roi = self.region_of_interest(edges)
        
        # Detect lines
        lines = self.detect_lines(roi)
        
        # Analyze lanes
        lane_action = self.analyze_lanes(lines, frame.shape[1])
        
        # Draw lanes on the frame
        result = self.draw_lines(display_frame, lines)
        
        # Add visualization
        cv2.putText(result, f"Lane Action: {lane_action}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Lane Confidence: {self.lane_confidence:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result, lane_action
    
    def get_lane_confidence(self):
        """Get the current confidence in lane detection"""
        return self.lane_confidence
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False