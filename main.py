import cv2
import numpy as np
import time
import keyboard
from mss import mss
import threading
from model_test import ObjectDetector
from lane_detect import LaneDetector

class DrivingController:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.lane_detector = LaneDetector()
        
        # Screen capture settings
        self.screen_region = {"top": 40, "left": 0, "width": 800, "height": 600}
        self.sct = mss()
        
        # Processing variables
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Action priority weights
        self.object_priority = 0.7  # Object detection has higher priority
        self.lane_priority = 0.3    # Lane detection has lower priority
        
        # Cooldown for actions
        self.last_action_time = time.time()
        self.action_cooldown = 0.3  # seconds
    
    def execute_action(self, action):
        """Execute driving action"""
        if action is None or time.time() - self.last_action_time < self.action_cooldown:
            return
            
        self.last_action_time = time.time()
        
        if action == "LEFT":
            keyboard.press('a')
            threading.Timer(0.2, lambda: keyboard.release('a')).start()
        elif action == "RIGHT":
            keyboard.press('d')
            threading.Timer(0.2, lambda: keyboard.release('d')).start()
        elif action == "BRAKE":
            keyboard.press('space')
            threading.Timer(1.0, lambda: keyboard.release('space')).start()
    
    def determine_action(self, object_action, lane_action, lane_confidence):
        """Determine the best action based on both detectors"""
        # If object detector detects something critical (like potential collision), prioritize it
        if object_action == "BRAKE":
            return "BRAKE"
        
        # If lane confidence is low, give more weight to object detection
        effective_lane_priority = self.lane_priority * lane_confidence
        effective_obj_priority = 1.0 - effective_lane_priority
        
        # If they suggest the same action, just do it
        if object_action == lane_action:
            return object_action
        
        # If only one detector suggests an action, consider the priorities
        if object_action and not lane_action:
            return object_action
        if lane_action and not object_action:
            return lane_action if lane_confidence > 0.3 else None
            
        # If they conflict, decide based on priority
        if effective_obj_priority > effective_lane_priority:
            return object_action
        else:
            return lane_action
    
    def run(self):
        """Main running loop"""
        try:
            while self.running:
                loop_start = time.time()
                
                # Capture screen
                screenshot = np.array(self.sct.grab(self.screen_region))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                
                # Process with object detector
                object_frame = self.object_detector.process_frame(frame)
                object_action = self.object_detector.get_current_action()
                
                # Process with lane detector
                lane_frame, lane_action = self.lane_detector.process_frame(frame)
                lane_confidence = self.lane_detector.get_lane_confidence()
                
                # Determine best action
                final_action = self.determine_action(object_action, lane_action, lane_confidence)
                
                # Execute action
                self.execute_action(final_action)
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count >= 10:
                    end_time = time.time()
                    self.fps = self.frame_count / (end_time - self.start_time)
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Create combined visualization
                object_frame_resized = cv2.resize(object_frame, (400, 300))
                lane_frame_resized = cv2.resize(lane_frame, (400, 300))
                top_row = np.hstack((object_frame_resized, lane_frame_resized))
                
                # Add information panel
                info_panel = np.zeros((100, 800, 3), dtype=np.uint8)
                cv2.putText(info_panel, f"FPS: {self.fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Object Action: {object_action}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Lane Action: {lane_action} (Conf: {lane_confidence:.2f})", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Final Action: {final_action}", (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Combine everything
                combined_display = np.vstack((top_row, info_panel))
                
                # Show the combined frame
                cv2.imshow("Autonomous Driving", combined_display)
                
                # Check for exit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                
                # Add small delay to prevent CPU overload
                process_time = time.time() - loop_start
                if process_time < 0.01:  # Target ~100 FPS max
                    time.sleep(0.01 - process_time)
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.object_detector.cleanup()
        self.lane_detector.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = DrivingController()
    controller.run()