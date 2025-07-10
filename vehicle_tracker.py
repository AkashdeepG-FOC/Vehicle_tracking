import cv2
from ultralytics import YOLO
import numpy as np
from collections import OrderedDict
import math
from scipy.optimize import linear_sum_assignment

# Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure you have this model downloaded or specify the correct path

# Define the video source
video_path = 'car7.mp4'  # Replace with your video path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define colors for drawing
line_color = (0, 255, 255)  # Yellow for the center line
bbox_color = (255, 0, 0)    # Blue for bounding boxes
font = cv2.FONT_HERSHEY_SIMPLEX
text_bg_color = (255, 255, 255)  # White background for the text area
text_color = (0, 0, 0)  # Black text color

# Fetch class names from the model
class_names = model.names  # Dictionary mapping class IDs to names

# Define vehicle class IDs based on COCO dataset
vehicle_classes = [2, 3, 5, 6, 7]  # Adjust based on the objects you want to track

# Initialize counts and crossing state
vehicles_top_to_bottom = 0  # Count of vehicles moving from top to bottom ("OUT")
vehicles_bottom_to_top = 0  # Count of vehicles moving from bottom to top ("IN")
crossed_vehicles = set()  # Track vehicles that have crossed

# Get the width and height of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Convert 2 cm to pixels (assuming 37.7952755906 pixels per inch)
cm_to_pixels = 37.7952755906 / 2.54  # pixels per cm
offset_in_pixels = int(5 * cm_to_pixels)  # Now 1 cm in pixels

# Define the center line position (horizontal)
center_y = frame_height // 2 + offset_in_pixels  # Move center 1 cm down

# Initialize tracking dictionaries
vehicle_tracks = OrderedDict()  # vehicle_id: (cx, cy, side, bbox, class_id)
disappeared = OrderedDict()      # vehicle_id: number of consecutive frames disappeared

# Define tracking parameters
MAX_DISAPPEARED = 5  # Reduced for quicker removal
vehicle_id_counter = 0  # Define a unique ID for each vehicle

# Function to calculate Euclidean distance
def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Function to smooth positions using a simple moving average
def smooth_position(prev_pos, new_pos, alpha=0.5):
    smoothed_x = int(alpha * new_pos[0] + (1 - alpha) * prev_pos[0])
    smoothed_y = int(alpha * new_pos[1] + (1 - alpha) * prev_pos[1])
    return (smoothed_x, smoothed_y)

# Function to check if a point is inside the frame
def is_inside_frame(cx, cy, width, height):
    return 0 <= cx <= width and 0 <= cy <= height

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection with YOLOv8
    results = model.predict(source=frame, conf=0.4, iou=0.5, max_det=100)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Use CPU for compatibility
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract class ID
            cls = int(box.cls[0])

            # Filter for vehicle classes
            if cls in vehicle_classes:
                # Calculate the centroid of the bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append((cx, cy, cls, (x1, y1, x2, y2)))

    # If there are no tracks yet, register all detections
    if len(vehicle_tracks) == 0:
        for det in detections:
            vehicle_id_counter += 1
            side = 'top' if det[1] < center_y else 'bottom'
            vehicle_tracks[vehicle_id_counter] = (det[0], det[1], side, det[3], det[2])  # Store class ID
            disappeared[vehicle_id_counter] = 0
    else:
        # Prepare lists of existing track centroids and new detections
        existing_ids = list(vehicle_tracks.keys())
        existing_centroids = [(track[0], track[1]) for track in vehicle_tracks.values()]
        new_centroids = [(det[0], det[1]) for det in detections]

        if len(new_centroids) > 0 and len(existing_centroids) > 0:
            # Compute distance matrix between existing tracks and new detections
            distance_matrix = np.zeros((len(existing_centroids), len(new_centroids)), dtype=np.float32)

            for i, ec in enumerate(existing_centroids):
                for j, nc in enumerate(new_centroids):
                    distance_matrix[i, j] = euclidean_distance(ec, nc)

            # Perform matching between tracks and detections
            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            assigned_tracks = set()
            assigned_detections = set()

            for r, c in zip(row_ind, col_ind):
                if distance_matrix[r, c] > 100:  # Threshold for matching
                    continue
                track_id = existing_ids[r]
                det = detections[c]

                # Smooth the position to reduce jitter
                prev_cx, prev_cy = vehicle_tracks[track_id][0], vehicle_tracks[track_id][1]
                new_cx, new_cy = det[0], det[1]
                smoothed_cx, smoothed_cy = smooth_position((prev_cx, prev_cy), (new_cx, new_cy))

                # Update the track with new position
                vehicle_tracks[track_id] = (smoothed_cx, smoothed_cy,
                                            vehicle_tracks[track_id][2],
                                            det[3], det[2])  # Update bbox and store class ID
                disappeared[track_id] = 0
                assigned_tracks.add(track_id)
                assigned_detections.add(c)

                # Check if the vehicle has crossed the center line
                prev_side = vehicle_tracks[track_id][2]
                current_side = 'bottom' if smoothed_cy > center_y else 'top'

                # Count only if the vehicle hasn't crossed before
                if prev_side == 'top' and current_side == 'bottom' and track_id not in crossed_vehicles:
                    vehicles_top_to_bottom += 1
                    crossed_vehicles.add(track_id)
                    print(f"Vehicle {track_id} crossed top to bottom. Total: {vehicles_top_to_bottom}")
                    vehicle_tracks[track_id] = (smoothed_cx, smoothed_cy, 'bottom', det[3], det[2])  # Update to bottom
                elif prev_side == 'bottom' and current_side == 'top' and track_id not in crossed_vehicles:
                    vehicles_bottom_to_top += 1
                    crossed_vehicles.add(track_id)
                    print(f"Vehicle {track_id} crossed bottom to top. Total: {vehicles_bottom_to_top}")
                    vehicle_tracks[track_id] = (smoothed_cx, smoothed_cy, 'top', det[3], det[2])  # Update to top

            # Handle unassigned tracks
            for track_id in existing_ids:
                if track_id not in assigned_tracks:
                    # Check if the vehicle is still inside the frame
                    cx, cy = vehicle_tracks[track_id][0], vehicle_tracks[track_id][1]
                    if not is_inside_frame(cx, cy, frame_width, frame_height):
                        disappeared[track_id] += 1
                        print(f"Vehicle {track_id} is outside the frame. Disappeared count: {disappeared[track_id]}")
                    else:
                        disappeared[track_id] += 1

                    if disappeared[track_id] > MAX_DISAPPEARED:
                        print(f"Removing vehicle {track_id} from tracking.")
                        del vehicle_tracks[track_id]
                        del disappeared[track_id]
                        crossed_vehicles.discard(track_id)  # Remove from crossed vehicles

        # Register new detections
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                vehicle_id_counter += 1
                side = 'top' if det[1] < center_y else 'bottom'
                vehicle_tracks[vehicle_id_counter] = (det[0], det[1], side, det[3], det[2])
                disappeared[vehicle_id_counter] = 0

    # Draw center line
    cv2.line(frame, (0, center_y), (frame_width, center_y), line_color, 2)

    # Draw bounding boxes for tracked vehicles
    for track_id, (cx, cy, side, bbox, class_id) in vehicle_tracks.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
        cv2.putText(frame, f"{class_names[class_id]} {track_id}", (x1, y1 - 5),
                    font, 0.5, (0, 255, 0), 2)

    # Add "OUT" and "IN" counters with white background and black text
    out_text = f"IN: {vehicles_top_to_bottom}"
    in_text = f"OUT: {vehicles_bottom_to_top}"

    # Draw the text for "OUT" on the left corner
    text_size_out = cv2.getTextSize(out_text, font, 1, 2)[0]
    padding = 10  # Padding around text
    cv2.rectangle(frame,
                  (10, 10),
                  (10 + text_size_out[0] + padding, 10 + text_size_out[1] + padding),
                  text_bg_color,
                  -1)
    cv2.putText(frame, out_text, (15, 10 + text_size_out[1] + 5), font, 1, text_color, 2)

    # Draw the text for "IN" on the right corner
    text_size_in = cv2.getTextSize(in_text, font, 1, 2)[0]
    cv2.rectangle(frame,
                  (frame_width - text_size_in[0] - padding - 10, 10),
                  (frame_width - 10, 10 + text_size_in[1] + padding),
                  text_bg_color,
                  -1)
    cv2.putText(frame, in_text,
                (frame_width - text_size_in[0] - padding - 5, 10 + text_size_in[1] + 5),
                font, 1, text_color, 2)

    # Show the resulting frame
    cv2.imshow("Vehicle Tracking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()