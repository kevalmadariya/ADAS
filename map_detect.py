import cv2
import numpy as np
import mss

class PathDetector:
    def __init__(self):
        self.minimap_region = [(13, 489), (163, 485), (163, 570), (14, 570)]
        self.pointer_position = (87, 559)
        
        self.path_detection_params = {
            'min_area': 100,
            'angle_threshold': 15, 
            'critical_distance': 20,  
            'turn_angle_threshold': 10  # Lower threshold to detect more subtle turns
        }

    def extract_minimap(self, frame):
        x_min, y_min = self.minimap_region[0]
        x_max, y_max = self.minimap_region[2]
        return frame[y_min:y_max, x_min:x_max]

    def preprocess_image(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_path_direction(self, frame):
        """Detect turns and calculate precise turn angles in degrees."""
        mask = self.preprocess_image(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "No Path Detected", 0, [], mask
        
        main_path = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_path) < self.path_detection_params['min_area']:
            return "No Path Detected", 0, [], mask
        
        # Get pointer position relative to minimap
        pointer_x = self.pointer_position[0] - self.minimap_region[0][0]
        pointer_y = self.pointer_position[1] - self.minimap_region[0][1]
        pointer = np.array([pointer_x, pointer_y])
        
        # Convert contour to points and sort by distance to pointer
        path_points = [point[0] for point in main_path]
        path_points.sort(key=lambda p: np.linalg.norm(np.array(p) - pointer))
        
        # Get immediate points near the pointer
        immediate_points = []
        critical_points = []
        
        for point in path_points:
            dist = np.linalg.norm(np.array(point) - pointer)
            if dist <= self.path_detection_params['critical_distance']:
                immediate_points.append(point)
            elif dist <= self.path_detection_params['critical_distance'] * 3:  # Increased range for better prediction
                critical_points.append(point)
        
        if len(immediate_points) < 2 or len(critical_points) < 2:
            return "Go Straight", 0, path_points, mask
            
        # Calculate vectors for immediate path and upcoming path
        immediate_vector = np.array(immediate_points[-1]) - np.array(immediate_points[0])
        upcoming_vector = np.array(critical_points[-1]) - np.array(critical_points[0])
        
        # Normalize vectors for better angle calculation
        immediate_vector = immediate_vector / np.linalg.norm(immediate_vector)
        upcoming_vector = upcoming_vector / np.linalg.norm(upcoming_vector)
        
        # Calculate angle between vectors
        dot_product = np.dot(immediate_vector, upcoming_vector)
        # Clamp dot product to [-1, 1] to avoid numerical issues
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        angle_deg = np.degrees(angle)
        
        # Use cross product for turn direction
        cross_product = np.cross(immediate_vector, upcoming_vector)
        
        # Calculate distance to turn
        dist_to_critical = min([np.linalg.norm(np.array(p) - pointer) for p in critical_points])
        dist_percentage = min(100, int((self.path_detection_params['critical_distance'] * 2 - dist_to_critical) / 
                                      (self.path_detection_params['critical_distance'] * 2) * 100))
        
        # Only detect turn if angle is significant
        if angle_deg < self.path_detection_params['turn_angle_threshold']:
            return f"Go Straight", 0, path_points, mask
        
        # Determine turn direction and angle
        if cross_product > 0:
            decision = f"Turn Right {angle_deg:.1f}° ({dist_percentage}% close)"
            turn_angle = angle_deg
        else:
            decision = f"Turn Left {angle_deg:.1f}° ({dist_percentage}% close)"
            turn_angle = -angle_deg
            
        return decision, turn_angle, path_points, mask

    def visualize_detection(self, frame, decision, path_points):
        result = frame.copy()
        
        # Draw path points
        for point in path_points:
            cv2.circle(result, tuple(map(int, point)), 2, (0, 255, 0), -1)
        
        # Draw pointer
        pointer_x = self.pointer_position[0] - self.minimap_region[0][0]
        pointer_y = self.pointer_position[1] - self.minimap_region[0][1]
        cv2.circle(result, (pointer_x, pointer_y), 5, (255, 0, 0), -1)
        
        # Add decision text
        cv2.putText(result, decision, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result

def main():
    detector = PathDetector()
    monitor = {"top": 40, "left": 0, "width": 800, "height": 600}
    
    cv2.namedWindow("Path Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Turn Guidance", cv2.WINDOW_NORMAL)
    
    with mss.mss() as sct:
        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            minimap = detector.extract_minimap(frame)
            decision, turn_angle, path_points, mask = detector.get_path_direction(minimap)
            result = detector.visualize_detection(minimap, decision, path_points)
            
            # Create a visual turn indicator
            turn_indicator = np.zeros((300, 300, 3), dtype=np.uint8)
            center = (150, 150)
            cv2.circle(turn_indicator, center, 100, (50, 50, 50), -1)
            
            # Draw the current direction (up)
            cv2.line(turn_indicator, center, (150, 50), (0, 255, 0), 3)
            
            # Draw the suggested turn direction
            if abs(turn_angle) > detector.path_detection_params['turn_angle_threshold']:
                # Calculate the endpoint for the turn angle
                angle_rad = np.radians(turn_angle)
                end_x = int(center[0] - 100 * np.sin(angle_rad))  # Inverted for correct display
                end_y = int(center[1] - 100 * np.cos(angle_rad))
                
                # Draw the turn direction line
                color = (0, 0, 255) if abs(turn_angle) > 30 else (0, 255, 255)
                cv2.line(turn_indicator, center, (end_x, end_y), color, 3)
                
                # Add angle text
                angle_text = f"{abs(turn_angle):.1f}°"
                cv2.putText(turn_indicator, angle_text, (center[0] + 10, center[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add direction text
                direction = "RIGHT" if turn_angle > 0 else "LEFT"
                cv2.putText(turn_indicator, direction, (center[0] - 30, center[1] - 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Path Detection", result)
            cv2.imshow("Turn Guidance", turn_indicator)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()