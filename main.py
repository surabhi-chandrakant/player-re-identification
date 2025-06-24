import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO # Make sure you have ultralytics installed

# --- Configuration ---
# Path to your input video file
VIDEO_PATH = "15sec_input_720p.mp4" 
# Path to your downloaded YOLOv11 model
MODEL_PATH = "best.pt" 
# Path to save the processed output video
OUTPUT_VIDEO_PATH = "reidentified_output.mp4" 

# Thresholds for object detection and tracking
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for a detection to be considered valid
IOU_TRACKING_THRESHOLD = 0.3 # IoU threshold for associating a new detection with an existing active player
FEATURE_SIMILARITY_THRESHOLD = 0.5 # Similarity threshold for re-identifying a player who went out of frame
                                   # (For color histograms, 0.5 is a common starting point for correlation)

# How many frames a player can be "lost" (not detected) before being moved to the inactive pool.
# This helps in handling brief occlusions or missed detections.
MAX_LOST_FRAMES = 15 # e.g., at 30 FPS, 15 frames is 0.5 seconds of allowed absence


# --- Global Variables for Tracking State ---
next_player_id = 0 # Counter for assigning unique new player IDs
active_players = {} # Dictionary to store Player objects currently being tracked in the frame
                    # Format: {player_id: Player_object}
inactive_players = {} # Dictionary to store Player objects that went out of frame or were lost for too long,
                      # but whose features are retained for potential re-identification.
                      # Format: {player_id: Player_object}

class Player:
    """
    Represents a single player being tracked, holding their current state and historical features.
    """
    def __init__(self, player_id, bbox, frame_num, features=None):
        self.player_id = player_id
        self.bbox = bbox # Current bounding box [x1, y1, x2, y2]
        self.last_seen_frame = frame_num # Last frame number this player was detected
        self.features = features # Extracted visual features for re-identification
        self.lost_frames_count = 0 # Counter for consecutive frames the player has not been detected

    def update_bbox(self, new_bbox, frame_num):
        """Updates the player's bounding box and resets the lost frame count."""
        self.bbox = new_bbox
        self.last_seen_frame = frame_num
        self.lost_frames_count = 0 # Reset lost count when player is seen

    def __repr__(self):
        """String representation for debugging/logging."""
        return (f"Player(ID:{self.player_id}, Bbox:[{int(self.bbox[0])},{int(self.bbox[1])},"
                f"{int(self.bbox[2])},{int(self.bbox[3])}], LastSeen:{self.last_seen_frame})")

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou

def extract_features(image, bbox):
    """
    Extracts visual features from the cropped player region.
    Currently uses a simple 3D color histogram (HSV).
    This is a basic approach; more robust solutions often involve deep learning embeddings.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure bounding box coordinates are within image dimensions to prevent errors
    h, w, _ = image.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Check for valid crop dimensions (width and height must be positive)
    if x2 <= x1 or y2 <= y1:
        return None 

    cropped_player = image[y1:y2, x1:x2]

    # Check if the cropped image is empty after slicing
    if cropped_player.size == 0 or cropped_player.shape[0] == 0 or cropped_player.shape[1] == 0:
        return None

    # Compute HSV color histogram
    try:
        hsv_cropped = cv2.cvtColor(cropped_player, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_cropped], [0, 1, 2], None, [8, 8, 8], # 8 bins for H, S, V
                            [0, 180, 0, 256, 0, 256]) # H range 0-180, S/V range 0-256
        hist = cv2.normalize(hist, hist).flatten() # Normalize and flatten the histogram
        return hist
    except cv2.error as e:
        # Catch OpenCV errors that might occur if the cropped_player is problematic
        return None

def compare_features(features1, features2):
    """
    Compares two feature vectors (e.g., histograms).
    Currently uses correlation for histogram comparison (1.0 is perfect match).
    """
    if features1 is None or features2 is None or len(features1) != len(features2):
        return 0.0 # Return 0.0 if features are invalid or mismatched

    return cv2.compareHist(features1, features2, cv2.HISTCMP_CORREL)


def main():
    global next_player_id, active_players, inactive_players

    # --- Model Loading and Class ID Discovery ---
    try:
        model = YOLO(MODEL_PATH)
        print("Model Class Names:", model.names) # Print all class names detected by the model
        player_class_id = -1
        found_player_class = False
        
        # Dynamically find the class ID for 'player' (case-insensitive)
        for class_id, class_name in model.names.items():
            if class_name.lower() == 'player': 
                player_class_id = class_id
                found_player_class = True
                break
        
        if not found_player_class:
            print("Error: 'player' class not found in model's names.")
            print("Please ensure your model's training data included a 'player' class and its name is correctly defined.")
            return

        print(f"Detected 'player' class ID: {player_class_id}")

    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure 'ultralytics' is installed and MODEL_PATH is correct.")
        return

    # --- Video Input Setup ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}. Check file path and existence.")
        return

    # Get video properties for output video configuration
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- Video Output Setup ---
    # Define the codec (e.g., 'mp4v' for .mp4, 'XVID' for .avi) and create VideoWriter object
    # If 'mp4v' does not work, try 'XVID' and change OUTPUT_VIDEO_PATH extension to .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {OUTPUT_VIDEO_PATH}.")
        print("Possible issues: Invalid path, lack of write permissions, or missing video codecs (e.g., FFmpeg).")
        return

    # --- Main Video Processing Loop ---
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret: # Break loop if no more frames (end of video)
            break

        frame_num += 1
        # print(f"Processing Frame: {frame_num}") # Uncomment for frame-by-frame progress log

        # --- 1. Object Detection (YOLOv11 Inference) ---
        # This list will hold all raw detections from the model, regardless of class or confidence.
        # Used for debugging visualization.
        all_raw_detections = [] 
        
        # This list will only contain 'player' detections that pass the confidence threshold.
        # These are the detections that will be fed into the tracking/re-identification logic.
        player_detections_for_tracking = [] 

        results = model(frame, verbose=False) # Run inference, suppress verbose output in console
        for r in results: # Iterate over detection results for the current frame
            for *xyxy, conf, cls in r.boxes.data: # Unpack bounding box, confidence, and class ID
                # --- CORRECTED LINE BELOW ---
                bbox = xyxy # xyxy is already a list, no need for .tolist()
                # --- END CORRECTION ---
                confidence = float(conf)
                class_id = int(cls)
                class_name = model.names.get(class_id, "unknown") # Get class name safely

                # Store all raw detections for the debug visualization layer
                all_raw_detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

                # Only consider 'player' detections above the confidence threshold for tracking
                if class_id == player_class_id and confidence > CONFIDENCE_THRESHOLD:
                    player_detections_for_tracking.append(bbox)


        # --- 2. Tracking and Re-identification Logic ---
        current_frame_assigned_ids = [] # List to hold IDs of players successfully matched in the current frame
        matched_detections_indices = set() # Store indices of `player_detections_for_tracking` that have been matched

        # 2a. Short-term Tracking: Try to match new detections with existing active players (IoU-based)
        for i, det_bbox in enumerate(player_detections_for_tracking):
            best_match_player_id = -1
            max_iou = 0.0

            # Iterate over a copy to safely modify `active_players` if needed
            for player_id, player in list(active_players.items()): 
                iou = calculate_iou(player.bbox, det_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_match_player_id = player_id

            if max_iou >= IOU_TRACKING_THRESHOLD: 
                # If a good IoU match is found, update the existing player
                active_players[best_match_player_id].update_bbox(det_bbox, frame_num)
                current_frame_assigned_ids.append(best_match_player_id)
                matched_detections_indices.add(i)
            # Else: This detection is not a direct IoU match with any currently active player.
            # It will be considered for re-identification or assigned a new ID.

        # 2b. Re-identification: Process unmatched detections (appearance-based)
        unmatched_detections = [det_bbox for i, det_bbox in enumerate(player_detections_for_tracking) 
                                if i not in matched_detections_indices]

        for det_bbox in unmatched_detections:
            player_features = extract_features(frame, det_bbox)
            if player_features is None: # Skip if feature extraction failed for this detection
                continue

            best_reid_match_id = -1
            max_similarity = 0.0

            # Compare current detection's features with features of previously inactive players
            for player_id, player in list(inactive_players.items()): # Iterate over copy
                similarity = compare_features(player_features, player.features)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_reid_match_id = player_id

            if max_similarity >= FEATURE_SIMILARITY_THRESHOLD: 
                # If a strong feature match is found, re-identify the player
                reidentified_player = inactive_players.pop(best_reid_match_id) # Move from inactive to active
                reidentified_player.update_bbox(det_bbox, frame_num) # Update bbox and reset lost_frames_count
                reidentified_player.features = player_features # Update features with latest appearance for robustness
                active_players[best_reid_match_id] = reidentified_player
                current_frame_assigned_ids.append(best_reid_match_id)
                # print(f"Re-identified Player {reidentified_player.player_id} at frame {frame_num}") # Detailed log
            else:
                # If no match with active or inactive players, assign a brand new ID
                new_player = Player(next_player_id, det_bbox, frame_num, player_features)
                active_players[next_player_id] = new_player
                current_frame_assigned_ids.append(next_player_id)
                # print(f"Assigned new ID {next_player_id} at frame {frame_num}") # Detailed log
                next_player_id += 1

        # 2c. Update Player Status: Handle players no longer detected
        players_to_deactivate = []
        for player_id, player in list(active_players.items()): # Iterate over copy
            if player_id not in current_frame_assigned_ids:
                player.lost_frames_count += 1 # Increment lost count if player not seen in current frame
                if player.lost_frames_count > MAX_LOST_FRAMES:
                    # If player has been lost for too many frames, move to inactive pool
                    players_to_deactivate.append(player_id)

        for player_id in players_to_deactivate:
            # print(f"Player {player_id} moved to inactive (lost for too many frames) at frame {frame_num}") # Detailed log
            player = active_players.pop(player_id) # Remove from active
            inactive_players[player_id] = player # Add to inactive

        # --- 3. Visualization and Saving Output ---
        display_frame = frame.copy()

        # Debug Visualization Layer (shows ALL raw detections from YOLO)
        # This helps in diagnosing if the model is mistakenly detecting non-player objects.
        # Green: Player (high confidence), Cyan: Ball, Orange: Goalkeeper, Magenta: Referee, Red: Other/low confidence
        for det in all_raw_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']

            color = (0, 0, 255) # Default: Red for unmatched or other classes
            if class_name.lower() == 'player':
                if confidence > CONFIDENCE_THRESHOLD:
                    color = (0, 255, 0) # Green for players detected above threshold
                else: # Player, but below confidence threshold
                    color = (0, 128, 0) # Darker green
            elif class_name.lower() == 'ball':
                color = (255, 255, 0) # Cyan
            elif class_name.lower() == 'goalkeeper':
                color = (0, 165, 255) # Orange
            elif class_name.lower() == 'referee':
                color = (255, 0, 255) # Magenta

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1) # Thin box for raw detections
            cv2.putText(display_frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Final Visualization Layer (shows only the TRACKED players with their consistent IDs)
        # These boxes will be thicker and yellow to stand out, indicating successful tracking.
        for player_id, player in active_players.items():
            x1, y1, x2, y2 = map(int, player.bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3) # Thicker, yellow box for tracked
            cv2.putText(display_frame, f"ID: {player.player_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) # Yellow text for ID

        # Write the processed frame to the output video file
        out.write(display_frame)

        # Display the frame in a window (optional, can be commented out for headless processing)
        cv2.imshow("Player Re-identification", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit early
            break

    # --- Cleanup and Final Report ---
    cap.release() # Release input video capture
    out.release() # Release output video writer (IMPORTANT: saves the file)
    cv2.destroyAllWindows() # Close all OpenCV windows

    print("\nRe-identification process complete.")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
    print("\n--- Final Player Status ---")
    print("Active Players (players present in the last processed frame):")
    for pid, player_obj in active_players.items():
        print(f"  {player_obj}")
    print("\nInactive Players (players who left frame or were lost, but their IDs are retained):")
    for pid, player_obj in inactive_players.items():
        print(f"  {player_obj}")

if __name__ == "__main__":
    main()