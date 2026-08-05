import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.figure as mfigure # Not strictly used, but was in original imports
import os
from PIL import Image

# Perfect human
# Shape parameters for detection
params = {
    'human_min_area': 1000,       # Minimum area for human detection
    'human_max_area': 10000,      # Maximum area for human detection
    'human_min_ratio': 0.3,       # Minimum width/height ratio for humans
    'human_max_ratio': 3.0,       # Maximum width/height ratio for humans
    'ball_min_area': 83,          # Minimum area for ball detection - Aligned with "Perfect ball"
    'ball_max_area': 900,         # Maximum area for ball detection - Aligned with "Perfect ball"
    'ball_min_ratio': 0.3,        # Minimum width/height ratio for ball - Aligned with "Perfect ball"
    'ball_max_ratio': 1.5,        # Maximum width/height ratio for ball - Updated to "Perfect ball"
    'grouping_distance': 45,      # Distance threshold for grouping contours - increased
    'dilate_iterations': 3,       # Number of dilation iterations
    'threshold_value': 20,        # Threshold value for binary segmentation
    'min_contour_area': 50,      # Minimum contour area to consider
    'tennis_player_grouping': 5,  # Special distance for tennis player parts grouping
    'vertical_grouping_bias': 1.5,  # Bias towards vertical grouping (for standing players)
    'court_divider_y': 0.5,         # Relative position of court divider (0-1)
    
    'use_player_positions': True,    # From older params
    'court_expand_top': 0.3,      # From older params (not used by current rectify_court)
    'court_expand_bottom': 0.2,   # From older params (not used by current rectify_court)
    'court_expand_left': 0.1,     # From older params (not used by current rectify_court)
    'court_expand_right': 0.1,    # From older params (not used by current rectify_court)

    'min_blue_ratio': 0.3,        # Minimum blue ratio for court detection (Updated from older: 0.08)
    'court_min_area': 20000,       # Minimum area for valid court detection (Updated from older: 20000)
    'court_max_area': 500000,      # Maximum area for valid court detection
    'court_aspect_ratio_min': 1.2, # Minimum width/height ratio for court
    'court_aspect_ratio_max': 2.2, # Maximum width/height ratio for court
    'court_max_change': 0.2,       # Maximum allowed change ratio between frames (for older detect_court)
    'horizontal_bias': 2.0,        # Weight factor for horizontal lines (from older params)
    
    'expansion_near': 0.4,         # Expansion for near side (bottom) - Kept from current for rectify_court
    'expansion_far': 0.3,          # Expansion for far side (top) - Kept from current for rectify_court
    'expansion_sides': 0.0,        # Expansion for sides (left/right) - Kept from current for rectify_court
    'margin_size': 30,              # Margin size in pixels - Kept from current for rectify_court
    
    'court_png_path': 'tennis_court.png',  # Path to tennis court PNG
    'create_court_png': False,  # Don't create court PNG if it doesn't exist
    'court_png_width': 500,    # Width of court PNG
    'court_png_height': 300,   # Height of court PNG
    'dot_size_player': 10,     # Size of player dots
    'dot_size_ball': 6,        # Size of ball dot
    'player1_color': (0, 255, 0),  # Green
    'player2_color': (0, 255, 255), # Yellow
    'ball_color': (0, 0, 255),     # Red
    'show_paths': False,        # Show movement paths (Updated from older: True)
    'path_length': 50,          # Number of positions to show in path
    'court_png_margin_x_percent': 0.05,  # Horizontal margin percent for court.png content
    'court_png_margin_y_percent': 0.05,   # Vertical margin percent for court.png content
    'allow_multiple_balls': False,
    'ball_search_radius_px': 400, 
    'player_search_radius_px': 100, 
    'ball_check_circularity': True, 
    'ball_circularity_threshold': 0.5, 
    'rectified_view_fixed_width': 600, # Kept for current rectify_court
    'rectified_view_fixed_height': 400, # Kept for current rectify_court
    'show_tennis_tracking_window': True, 
    'show_original_video_window': True,
    # Parameters from older code's params.update
    'show_ball_attributes': True,  # Show attributes of each ball (area, ratio, etc.)
    'ball_detection_threshold': 0.7, # Confidence threshold for ball detection (0-1)
    'ball_history_length': 30       # Number of frames to keep ball history
    # Removed court_detection_streak_threshold and court_corner_similarity_threshold
}

# Define standard tennis court dimensions
TENNIS_COURT_LENGTH_FT = 78.0  # Standard length
TENNIS_COURT_WIDTH_FT = 36.0   # Standard width for doubles (lines)
TENNIS_COURT_ASPECT_RATIO = TENNIS_COURT_LENGTH_FT / TENNIS_COURT_WIDTH_FT # Length / Width

# Initialize video capture and output
cap = cv2.VideoCapture('tennis.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter("output.avi", fourcc, 30.0, (1280,720)) # FPS restored to 30

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

# Initialize counters for statistics
humans_detected = 0
balls_detected = 0

# Remove trackbar creation for 'Parameters' window
# cv2.namedWindow('Parameters')
# ... (all cv2.createTrackbar calls removed) ...

# Function to group nearby contours
def group_contours(contours, max_distance):
    # If we have no contours, return empty
    if not contours:
        return []
    
    # Extract bounding boxes from contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Calculate centers of bounding boxes
    centers = [(x + w//2, y + h//2) for (x, y, w, h) in bounding_boxes]
    
    # Initialize groups
    groups = []
    processed = set()
    
    # Group contours based on distance
    for i, center in enumerate(centers):
        if i in processed:
            continue
        
        # Start a new group
        group = [i]
        processed.add(i)
        
        # Find all nearby centers
        for j, other_center in enumerate(centers):
            if j in processed or i == j:
                continue
            
            # Calculate distance between centers
            d = np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
            
            # If close enough, add to group
            if d < max_distance:
                group.append(j)
                processed.add(j)
        
        groups.append(group)
    
    # Convert groups of indices to groups of contours
    contour_groups = [[contours[i] for i in group] for group in groups]
    
    return contour_groups

# Function to create bounding box for a group of contours
def group_bounding_box(contours):
    # Combine all points from all contours
    all_points = np.vstack([contour for contour in contours])
    
    # Get bounding rectangle for all points
    x, y, w, h = cv2.boundingRect(all_points)
    
    return (x, y, w, h)

# Function to classify contour as human or ball - Updated with better ball detection
def classify_contour(x, y, w, h, area):
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Check if it's a ball using criteria from params
    if (params['ball_min_area'] <= area <= params['ball_max_area'] and
        params['ball_min_ratio'] <= aspect_ratio <= params['ball_max_ratio']):
        return "ball"
    
    # Check if it's a human
    elif (params['human_min_area'] <= area <= params['human_max_area'] and
          params['human_min_ratio'] <= aspect_ratio <= params['human_max_ratio']):
        return "human"
    
    # Otherwise unknown
    return "unknown"

# Enhanced grouping function specifically for tennis players
def group_tennis_player_boxes(contours, frame_height):
    if not contours:
        return []
        
    # Extract bounding boxes from contours
    boxes = [cv2.boundingRect(c) for c in contours]
    
    # Identify potential player boxes (somewhat large)
    player_boxes = []
    for i, (x, y, w, h) in enumerate(boxes):
        area = w * h
        if area > params['human_min_area'] / 2:  # More lenient area threshold
            player_boxes.append((i, x, y, w, h))
    
    # If no player boxes found, return regular grouping
    if not player_boxes:
        return group_contours(contours, params['grouping_distance'])
    
    # Split boxes into top and bottom court based on court divider
    divider_y = int(frame_height * params['court_divider_y'])
    top_boxes = []
    bottom_boxes = []
    
    for i, x, y, w, h in player_boxes:
        center_y = y + h//2
        if center_y < divider_y:
            top_boxes.append((i, x, y, w, h))
        else:
            bottom_boxes.append((i, x, y, w, h))
    
    # Group based on court position (stronger grouping within each court half)
    grouping_distance = params['tennis_player_grouping']
    
    # Function to group boxes in one half of the court
    def group_half_court(boxes_in_half):
        if not boxes_in_half:
            return []
            
        groups = []
        processed = set()
        
        for i, (idx_i, x_i, y_i, w_i, h_i) in enumerate(boxes_in_half):
            if i in processed:
                continue
            
            # Start a new group
            group = [idx_i]
            processed.add(i)
            
            # Check against other boxes in same half
            for j, (idx_j, x_j, y_j, w_j, h_j) in enumerate(boxes_in_half):
                if j in processed or i == j:
                    continue
                
                # Calculate centers
                center_i = (x_i + w_i//2, y_i + h_i//2)
                center_j = (x_j + w_j//2, y_j + h_j//2)
                
                # Calculate horizontal and vertical distances separately
                dx = abs(center_i[0] - center_j[0])
                dy = abs(center_i[1] - center_j[1]) / params['vertical_grouping_bias']  # Less weight to vertical distance
                
                # If they're close enough in both dimensions, group them
                if max(dx, dy) < grouping_distance:
                    group.append(idx_j)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    # Group boxes in each half of the court
    top_groups = group_half_court(top_boxes)
    bottom_groups = group_half_court(bottom_boxes)
    
    # Merge the indices with the original contour list
    result_groups = []
    
    for group in top_groups:
        result_groups.append([contours[idx] for idx in group])
        
    for group in bottom_groups:
        result_groups.append([contours[idx] for idx in group])
    
    # Handle any remaining contours (that weren't in player boxes)
    processed_indices = set()
    for group in top_groups + bottom_groups:
        processed_indices.update(group)
    
    remaining = [contours[i] for i in range(len(contours)) if i not in processed_indices]
    
    # Group remaining contours using regular method
    if remaining:
        regular_groups = group_contours(remaining, params['grouping_distance'])
        result_groups.extend(regular_groups)
    
    return result_groups

# Add court detection functionality
court_history = deque(maxlen=10)  # Store recent valid court shapes
last_valid_court = None # Will store the corners of the last confidently detected court
court_detected = False # Global flag

# Enhanced court detection function that also shows edges
def detect_court(frame):
    """Detect tennis court in the frame using blue color analysis and edge detection"""
    global court_history, last_valid_court, court_detected
    
    h, w = frame.shape[:2]
    
    # Create edge mask to ignore frame borders (5% of each edge)
    border_percent = 0.05
    border_x = int(w * border_percent)
    border_y = int(h * border_percent)
    
    edge_mask = np.ones((h, w), dtype=np.uint8) * 255
    edge_mask[:border_y, :] = 0  # Top border
    edge_mask[-border_y:, :] = 0  # Bottom border
    edge_mask[:, :border_x] = 0  # Left border
    edge_mask[:, -border_x:] = 0  # Right border
    
    # Apply edge mask to frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=edge_mask)
    
    # HSV Blue Detection - focus on blue court
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 70, 70]) # Adjusted lower blue for better court detection
    upper_blue = np.array([150, 255, 255]) # Adjusted upper blue
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Calculate blue ratio to check if court is likely in view
    # Ensure considered_area for blue_ratio is based on the unmasked part of the frame if edge_mask is used for blue_ratio calculation
    # For simplicity here, using total h*w, but for accuracy, should be cv2.countNonZero(edge_mask) if blue_mask was from masked_frame.
    # However, blue_mask is derived from masked_frame, so blue_pixels are already within the unmasked region.
    # The denominator should be the area where blue pixels *could* have been found.
    considered_area = cv2.countNonZero(edge_mask) # Area after removing borders
    blue_pixels_in_considered_area = cv2.countNonZero(blue_mask) # blue_mask is already from masked_frame
    blue_ratio = blue_pixels_in_considered_area / considered_area if considered_area > 0 else 0
    
    court_likely_in_view = blue_ratio >= params['min_blue_ratio']
    
    # Create visualization image
    court_viz = frame.copy()
    court_corners = None # Initialize to None for this frame
    
    # Edge detection on the blue mask for court visualization
    edges = cv2.Canny(blue_mask, 50, 150)
    
    # Dilate edges for better visualization
    edge_kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
    
    # Add edge overlay to visualization
    court_viz[edges_dilated > 0] = [100, 255, 100]  # Light green overlay for edges
    
    # Debug masks payload (simplified as the complex one is not used by this detect_court)
    debug_masks_payload = {"blue_mask": blue_mask, "edges_dilated": edges_dilated}

    if not court_likely_in_view:
        court_detected = False # Set main flag
        cv2.putText(court_viz, f"Court not detected (Blue ratio: {blue_ratio:.3f} < {params['min_blue_ratio']:.3f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Try to use last_valid_court if available, even if blue ratio is low (for persistence)
        if last_valid_court is not None:
            court_corners = last_valid_court
            cv2.polylines(court_viz, [np.int32(court_corners)], True, (255, 165, 0), 2) # Blue-Orange for low-blue persisted
            cv2.putText(court_viz, "Using last valid court (low blue)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            court_detected = True # Considered "detected" via strong persistence
        return court_viz, court_corners, debug_masks_payload, court_likely_in_view # Return potentially historical corners and blue status

    # Process blue mask to improve court detection (only if court_likely_in_view is True)
    close_kernel = np.ones((15, 15), np.uint8)
    blue_mask_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), np.uint8)
    blue_mask_processed = cv2.morphologyEx(blue_mask_closed, cv2.MORPH_OPEN, open_kernel)
    player_removal_kernel = np.ones((15, 15), np.uint8)
    blue_mask_no_players = cv2.morphologyEx(blue_mask_processed, cv2.MORPH_OPEN, player_removal_kernel)
    
    debug_masks_payload.update({
        "blue_mask_closed": blue_mask_closed,
        "blue_mask_processed": blue_mask_processed,
        "blue_mask_no_players": blue_mask_no_players
    })

    background_mask = cv2.dilate(blue_mask_processed, np.ones((15, 15), np.uint8))
    contours_geom, _ = cv2.findContours(blue_mask_no_players & background_mask, 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_court_found_this_frame = False
    current_frame_corners = None

    if contours_geom:
        contours_geom = sorted(contours_geom, key=cv2.contourArea, reverse=True)
        for contour in contours_geom:
            area = cv2.contourArea(contour)
            if not (params['court_min_area'] <= area <= params['court_max_area']):
                continue
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if 4 <= len(approx) <= 6:
                if len(approx) > 4:
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    approx = box.astype(np.intp)
                
                temp_corners = approx.reshape(-1, 2).astype(np.float32)
                xs = temp_corners[:, 0]; ys = temp_corners[:, 1]
                cx_geom = np.mean(xs); cy_geom = np.mean(ys)
                sorted_corners_list = []
                for corner_g in temp_corners:
                    x_g, y_g = corner_g
                    if x_g < cx_geom and y_g < cy_geom: sorted_corners_list.append([0, corner_g])
                    elif x_g > cx_geom and y_g < cy_geom: sorted_corners_list.append([1, corner_g])
                    elif x_g > cx_geom and y_g > cy_geom: sorted_corners_list.append([2, corner_g])
                    else: sorted_corners_list.append([3, corner_g])
                
                if len(sorted_corners_list) != 4: continue
                sorted_corners_list.sort(key=lambda item: item[0])
                current_frame_corners_sorted = np.array([item[1] for item in sorted_corners_list], dtype=np.float32)

                w_val = max(np.linalg.norm(current_frame_corners_sorted[1] - current_frame_corners_sorted[0]), np.linalg.norm(current_frame_corners_sorted[2] - current_frame_corners_sorted[3]))
                h_val = max(np.linalg.norm(current_frame_corners_sorted[3] - current_frame_corners_sorted[0]), np.linalg.norm(current_frame_corners_sorted[2] - current_frame_corners_sorted[1]))
                aspect_geom = w_val / h_val if h_val > 0 else 0
                
                if not (params['court_aspect_ratio_min'] <= aspect_geom <= params['court_aspect_ratio_max']):
                    continue
                
                # Consistency check with history (if history exists)
                if len(court_history) > 0:
                    last_hist_corners = court_history[-1]
                    displacements = np.linalg.norm(current_frame_corners_sorted - last_hist_corners, axis=1)
                    avg_displacement = np.mean(displacements)
                    max_allowed_displacement = min(w_val, h_val) * params['court_max_change']
                    if avg_displacement > max_allowed_displacement:
                        continue # Too much change
                current_frame_corners = current_frame_corners_sorted
                valid_court_found_this_frame = True
                break 
    if valid_court_found_this_frame:
        court_history.append(current_frame_corners)
        last_valid_court = current_frame_corners
        court_corners = current_frame_corners # Use current frame's detection
        court_detected = True
        cv2.polylines(court_viz, [np.int32(court_corners)], True, (0, 255, 0), 2) # Green for new/good detection
        cv2.putText(court_viz, "Court Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif last_valid_court is not None: # Blue was OK, geom failed, but have history
        court_corners = last_valid_court # Use last known good shape
        court_detected = True # Still considered "detected" via history
        cv2.polylines(court_viz, [np.int32(court_corners)], True, (0, 165, 255), 2) # Orange for historical
        cv2.putText(court_viz, "Using previous court (geom. failed)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else: # Blue OK, geom failed, no history
        court_detected = False
        cv2.putText(court_viz, "No court detected (geom. failed, no history)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # court_corners remains None

    if court_detected and court_corners is not None:
        divider_y_orig_frame = int(h * params['court_divider_y'])
        cv2.line(court_viz, (0, divider_y_orig_frame), (w, divider_y_orig_frame), (255, 0, 255), 1)
    
    return court_viz, court_corners, debug_masks_payload, court_likely_in_view
# Removed STATUS_MESSAGES and related globals (last_geometrically_confirmed_for_streak_corners, etc.)

# Function to create rectified view with court lines
def rectify_court(frame, corners):
    """Transform the court to a top-down view with proper tennis court aspect ratio"""
    if corners is None:
        return None, None, None
    
    # Use the globally defined tennis court aspect ratio (Length/Width)
    # TENNIS_COURT_RATIO is already defined as TENNIS_COURT_ASPECT_RATIO
    
    # Create expanded corners with different expansion for each side
    expanded_corners = corners.copy()
    
    # Calculate court center
    center = np.mean(corners, axis=0)
    
    # Calculate vectors from center to each corner
    center_to_corners = corners - center
    
    # Expand far side (top edge) - 30%
    expanded_corners[0] = corners[0] + center_to_corners[0] * params['expansion_far']  # Top-left
    expanded_corners[1] = corners[1] + center_to_corners[1] * params['expansion_far']  # Top-right
    
    # Expand near side (bottom edge) - 20%
    expanded_corners[2] = corners[2] + center_to_corners[2] * params['expansion_near']  # Bottom-right
    expanded_corners[3] = corners[3] + center_to_corners[3] * params['expansion_near']  # Bottom-left
    
    # Adjust side expansion (will only affect horizontal component)
    # Left side
    left_vector = np.array([-1.0, 0.0])  # Pure horizontal vector pointing left
    expanded_corners[0] += left_vector * params['expansion_sides'] * np.linalg.norm(center_to_corners[0])
    expanded_corners[3] += left_vector * params['expansion_sides'] * np.linalg.norm(center_to_corners[3])
    
    # Right side
    right_vector = np.array([1.0, 0.0])  # Pure horizontal vector pointing right
    expanded_corners[1] += right_vector * params['expansion_sides'] * np.linalg.norm(center_to_corners[1])
    expanded_corners[2] += right_vector * params['expansion_sides'] * np.linalg.norm(center_to_corners[2])
    
    # Calculate width and height of the expanded court
    width_top = np.linalg.norm(expanded_corners[1] - expanded_corners[0])
    width_bottom = np.linalg.norm(expanded_corners[2] - expanded_corners[3])
    height_left = np.linalg.norm(expanded_corners[3] - expanded_corners[0])
    height_right = np.linalg.norm(expanded_corners[2] - expanded_corners[1])
    
    # Use the maximum dimensions for better accuracy
    # These are calculated from expanded_corners but NOT directly used for output size anymore
    # detected_court_width = int(max(width_top, width_bottom)) 
    # detected_court_height = int(max(height_left, height_right))
    
    # Use fixed dimensions for the output rectified view
    fixed_rectified_width = params['rectified_view_fixed_width']
    fixed_rectified_height = params['rectified_view_fixed_height']

    # Define margin around the court
    margin = params['margin_size']
    
    # Define the destination points for the transform using fixed output size
    dst_points = np.array([
        [margin, margin],
        [margin + fixed_rectified_width, margin],
        [margin + fixed_rectified_width, margin + fixed_rectified_height],
        [margin, margin + fixed_rectified_height]
    ], dtype=np.float32)
    
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(expanded_corners, dst_points)
    
    # Apply the transform to the fixed output size
    output_width = fixed_rectified_width + 2 * margin
    output_height = fixed_rectified_height + 2 * margin
    rectified = cv2.warpPerspective(frame, M, (output_width, output_height))
    
    # Draw the original court boundaries as a yellow rectangle
    original_rect_points_transformed = []
    for corner in corners:
        # Apply perspective transform to original corner
        px = (M[0][0] * corner[0] + M[0][1] * corner[1] + M[0][2]) / \
             (M[2][0] * corner[0] + M[2][1] * corner[1] + M[2][2])
        py = (M[1][0] * corner[0] + M[1][1] * corner[1] + M[1][2]) / \
             (M[2][0] * corner[0] + M[2][1] * corner[1] + M[2][2])
        original_rect_points_transformed.append([int(px), int(py)])
    
    # Draw the original court boundary
    cv2.polylines(rectified, [np.array(original_rect_points_transformed)], True, (0, 255, 255), 2) # Yellow

    # Calculate bounding box of these transformed original court points
    pxs = [p[0] for p in original_rect_points_transformed]
    pys = [p[1] for p in original_rect_points_transformed]
    min_x_orig = min(pxs) if pxs else 0
    min_y_orig = min(pys) if pys else 0
    max_x_orig = max(pxs) if pxs else 0
    max_y_orig = max(pys) if pys else 0
    w_orig = max_x_orig - min_x_orig
    h_orig = max_y_orig - min_y_orig
    original_court_bbox_in_rectified_coords = (min_x_orig, min_y_orig, w_orig, h_orig)
    
    # Draw court divider in rectified view - horizontal line at 50% height
    rect_h, rect_w = rectified.shape[:2]
    divider_y_rect = int(rect_h * 0.5)
    cv2.line(rectified, (0, divider_y_rect), (rect_w, divider_y_rect), (255, 0, 255), 1)
    
    # Add expansion info to visualization
    cv2.putText(rectified, f"Far: {params['expansion_far']*100:.0f}%, Near: {params['expansion_near']*100:.0f}%, Sides: {params['expansion_sides']*100:.0f}%", 
               (10, rect_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return rectified, M, original_court_bbox_in_rectified_coords

# Function to transform points from rectified view back to original frame
def transform_point_to_original(point, inverse_matrix):
    """Transform a point from rectified view back to original frame"""
    x, y = point
    
    # Apply inverse perspective transform
    denominator = inverse_matrix[2, 0] * x + inverse_matrix[2, 1] * y + inverse_matrix[2, 2]
    orig_x = (inverse_matrix[0, 0] * x + inverse_matrix[0, 1] * y + inverse_matrix[0, 2]) / denominator
    orig_y = (inverse_matrix[1, 0] * x + inverse_matrix[1, 1] * y + inverse_matrix[1, 2]) / denominator
    
    return (int(orig_x), int(orig_y))

# Function to transform rect from rectified view back to original frame
def transform_rect_to_original(rect, inverse_matrix):
    """Transform a rectangle from rectified view back to original frame"""
    x, y, w, h = rect
    
    # Transform all four corners
    top_left = transform_point_to_original((x, y), inverse_matrix)
    top_right = transform_point_to_original((x + w, y), inverse_matrix)
    bottom_right = transform_point_to_original((x + w, y + h), inverse_matrix)
    bottom_left = transform_point_to_original((x, y + h), inverse_matrix)
    
    # Calculate bounding box of transformed points
    min_x = min(top_left[0], top_right[0], bottom_right[0], bottom_left[0])
    min_y = min(top_left[1], top_right[1], bottom_right[1], bottom_left[1])
    max_x = max(top_left[0], top_right[0], bottom_right[0], bottom_left[0])
    max_y = max(top_left[1], top_right[1], bottom_right[1], bottom_left[1])
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)

# Function to merge nearby human boxes
def merge_human_boxes(human_boxes, max_distance=500):
    """Merge human detections that are close together"""
    if len(human_boxes) <= 1:
        return human_boxes
    
    # Extract box centers
    centers = [(x + w//2, y + h//2) for (x, y, w, h) in human_boxes]
    
    # Initialize groups
    merged_boxes = []
    processed = set()
    
    # Group boxes based on distance
    for i, center in enumerate(centers):
        if i in processed:
            continue
        
        # Start a new group
        group = [human_boxes[i]]
        processed.add(i)
        
        # Find all nearby boxes
        for j, other_center in enumerate(centers):
            if j in processed or i == j:
                continue
            # Calculate distance between centers
            d = np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
            
            # If close enough, add to group
            if d < max_distance:
                group.append(human_boxes[j])
                processed.add(j)
        
        # Merge boxes in the group
        if len(group) > 1:
            # Find bounding rect that contains all boxes in group
            min_x = min(box[0] for box in group)
            min_y = min(box[1] for box in group)
            max_x = max(box[0] + box[2] for box in group)
            max_y = max(box[1] + box[3] for box in group)
            merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            merged_boxes.append(merged_box)
        else:
            # Single box, just add to merged list
            merged_boxes.append(group[0])
    
    return merged_boxes

# Improved function to merge nearby human boxes that respects court division
def merge_human_boxes(human_boxes, max_distance=500, frame_height=None):
    """Merge human detections that are close together, but never across the court divider"""
    if len(human_boxes) <= 1:
        return human_boxes
    
    # Calculate court divider y-position
    court_divider_y = None
    if frame_height:
        court_divider_y = int(frame_height * params['court_divider_y'])
    
    # Extract box centers
    centers = [(x + w//2, y + h//2) for (x, y, w, h) in human_boxes]
    
    # Determine which half of the court each box is in
    top_court_boxes = []
    bottom_court_boxes = []
    
    for i, (x, y, w, h) in enumerate(human_boxes):
        center_y = y + h//2
        # If we know the court divider position, use it
        if court_divider_y:
            if center_y < court_divider_y:
                top_court_boxes.append(i)
            else:
                bottom_court_boxes.append(i)
        # Otherwise estimate based on the frame center
        else:
            if center_y < frame_height / 2:
                top_court_boxes.append(i)
            else:
                bottom_court_boxes.append(i)
    
    # Now we'll merge boxes, but only within the same court half
    merged_boxes = []
    
    # Function to merge boxes within a specific half of the court
    def merge_boxes_in_half(half_indices):
        if not half_indices:
            return []
            
        result = []
        processed = set()
        
        for i in half_indices:
            if i in processed:
                continue
            
            # Start a new group
            group = [human_boxes[i]]
            processed.add(i)
            center_i = centers[i]
            
            # Find all nearby boxes IN THE SAME HALF
            for j in half_indices:
                if j in processed or i == j:
                    continue
                
                center_j = centers[j]
                
                # Calculate distance between centers
                d = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
                
                # If close enough, add to group
                if d < max_distance:
                    group.append(human_boxes[j])
                    processed.add(j)
            
            # Merge boxes in the group
            if len(group) > 1:
                # Find bounding rect that contains all boxes in group
                min_x = min(box[0] for box in group)
                min_y = min(box[1] for box in group)
                max_x = max(box[0] + box[2] for box in group)
                max_y = max(box[1] + box[3] for box in group)
                merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)
                result.append(merged_box)
            else:
                # Single box, just add to result
                result.append(group[0])
                
        return result
    
    # Merge boxes in each half separately
    top_merged = merge_boxes_in_half(top_court_boxes)
    bottom_merged = merge_boxes_in_half(bottom_court_boxes)
    
    # Combine results from both halves
    merged_boxes = top_merged + bottom_merged
    
    return merged_boxes

# OLDER detect_objects_on_rectified_court (from prompt, takes only rectified_frame)
# This is the long version from the prompt that includes player/ball tracking.
def detect_objects_on_rectified_court(rectified_frame):
    """Detect players and ball in the rectified court view with persistence"""
    global prev_rectified_frame, top_player_last_seen, bottom_player_last_seen
    global ball_last_seen, frames_without_top_player, frames_without_bottom_player
    global frames_without_ball, top_player_history, bottom_player_history, ball_history
    global all_ball_history, frame_counter # Ensure frame_counter is global if used for ball_attrs
    global top_player_path, bottom_player_path, ball_path 
    
    h_rect, w_rect = rectified_frame.shape[:2] # Renamed to avoid conflict with outer scope h,w
    court_divider_y = int(h_rect * 0.5)  # Court divider at middle of rectified view
    
    if 'prev_rectified_frame' not in globals() or prev_rectified_frame is None or \
       prev_rectified_frame.shape != rectified_frame.shape:
        prev_rectified_frame = rectified_frame.copy()
        return None, None, [] 
    
    diff = cv2.absdiff(rectified_frame, prev_rectified_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, params['threshold_value'], 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=params['dilate_iterations'])
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= params['min_contour_area']]
    
    # Use simpler grouping for balls, tennis-specific for players
    ball_contour_groups = group_contours(filtered_contours, params['grouping_distance'])
    # Pass h_rect (height of rectified_frame) to group_tennis_player_boxes
    player_contour_groups = group_tennis_player_boxes(filtered_contours, h_rect) 
    
    detected_human_boxes = []
    detected_ball_boxes_with_attrs = []
    
    for group in player_contour_groups:
        x, y, w_box, h_box = group_bounding_box(group)
        area = w_box * h_box
        # Use classify_contour or direct checks
        if params['human_min_area'] <= area <= params['human_max_area']:
            aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
            if params['human_min_ratio'] <= aspect_ratio <= params['human_max_ratio']:
                detected_human_boxes.append((x, y, w_box, h_box))
    
    for group in ball_contour_groups:
        x, y, w_box, h_box = group_bounding_box(group)
        area = w_box * h_box
        aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
        
        if (params['ball_min_area'] <= area <= params['ball_max_area'] and
            params['ball_min_ratio'] <= aspect_ratio <= params['ball_max_ratio']):
            # Circularity check (optional, from current Oldtest2.py, can be added if desired)
            # is_circular_enough = True
            # if params['ball_check_circularity']: ...
            # if not is_circular_enough: continue

            ratio_diff = abs(aspect_ratio - 1.0)
            area_ideal = (params['ball_min_area'] + params['ball_max_area']) / 2
            area_diff = abs(area - area_ideal) / area_ideal if area_ideal > 0 else 1.0
            confidence = 1.0 - (ratio_diff * 0.5 + area_diff * 0.5)
            
            ball_attrs = {'area': area, 'aspect_ratio': aspect_ratio, 'confidence': confidence, 'frame': frame_counter}
            detected_ball_boxes_with_attrs.append((x, y, w_box, h_box, ball_attrs))

    # Merge human boxes (already done by group_tennis_player_boxes to some extent, but this can be a final merge)
    # detected_human_boxes = merge_human_boxes(detected_human_boxes, frame_height=h_rect) # Optional refinement
    
    for ball_box_data in detected_ball_boxes_with_attrs:
        all_ball_history.append((frame_counter, ball_box_data))
        if len(all_ball_history) > params['ball_history_length'] * 10: # Keep a longer history for analysis
            all_ball_history = all_ball_history[-params['ball_history_length']*10:]
    
    detected_top_players = []
    detected_bottom_players = []
    for box in detected_human_boxes:
        x, y, w_b, h_b = box # Renamed to avoid conflict
        center_y = y + h_b//2
        if center_y < court_divider_y:
            detected_top_players.append(box)
        else:
            detected_bottom_players.append(box)
    
    top_player_rect = None
    new_top_player_assigned_this_frame = False
    if detected_top_players:
        candidate_for_top_player = None
        if top_player_last_seen:
            last_player_center_x = top_player_last_seen[0] + top_player_last_seen[2] // 2
            last_player_center_y = top_player_last_seen[1] + top_player_last_seen[3] // 2
            players_near_last_known_top = []
            for p_cand_rect in detected_top_players:
                cand_center_x = p_cand_rect[0] + p_cand_rect[2] // 2
                cand_center_y = p_cand_rect[1] + p_cand_rect[3] // 2
                dist_sq = (cand_center_x - last_player_center_x)**2 + (cand_center_y - last_player_center_y)**2
                if dist_sq < (params['player_search_radius_px']**2):
                    players_near_last_known_top.append(p_cand_rect)
            if players_near_last_known_top:
                players_near_last_known_top.sort(key=lambda b: b[2]*b[3], reverse=True)
                candidate_for_top_player = players_near_last_known_top[0]
        else:
            detected_top_players.sort(key=lambda b: b[2]*b[3], reverse=True)
            if detected_top_players:
                candidate_for_top_player = detected_top_players[0]
        if candidate_for_top_player:
            top_player_rect = candidate_for_top_player
            top_player_last_seen = top_player_rect
            frames_without_top_player = 0
            new_top_player_assigned_this_frame = True
            top_player_history.append(top_player_rect)
            # For path, use center of the player box, not feet, for general tracking
            center_x = top_player_rect[0] + top_player_rect[2]//2
            center_y = top_player_rect[1] + top_player_rect[3]//2 # Using center y
            top_player_path.append((center_x, center_y))

    if not new_top_player_assigned_this_frame:
        frames_without_top_player += 1
        if top_player_last_seen and frames_without_top_player < max_frames_without_detection:
            top_player_rect = top_player_last_seen
        else:
            top_player_rect = None
            if frames_without_top_player >= max_frames_without_detection:
                top_player_last_seen = None
    
    bottom_player_rect = None
    new_bottom_player_assigned_this_frame = False
    if detected_bottom_players:
        candidate_for_bottom_player = None
        if bottom_player_last_seen:
            last_player_center_x = bottom_player_last_seen[0] + bottom_player_last_seen[2] // 2
            last_player_center_y = bottom_player_last_seen[1] + bottom_player_last_seen[3] // 2
            players_near_last_known_bottom = []
            for p_cand_rect in detected_bottom_players:
                cand_center_x = p_cand_rect[0] + p_cand_rect[2] // 2
                cand_center_y = p_cand_rect[1] + p_cand_rect[3] // 2
                dist_sq = (cand_center_x - last_player_center_x)**2 + (cand_center_y - last_player_center_y)**2
                if dist_sq < (params['player_search_radius_px']**2):
                    players_near_last_known_bottom.append(p_cand_rect)
            if players_near_last_known_bottom:
                players_near_last_known_bottom.sort(key=lambda b: b[2]*b[3], reverse=True)
                candidate_for_bottom_player = players_near_last_known_bottom[0]
        else:
            detected_bottom_players.sort(key=lambda b: b[2]*b[3], reverse=True)
            if detected_bottom_players:
                candidate_for_bottom_player = detected_bottom_players[0]
        if candidate_for_bottom_player:
            bottom_player_rect = candidate_for_bottom_player
            bottom_player_last_seen = bottom_player_rect
            frames_without_bottom_player = 0
            new_bottom_player_assigned_this_frame = True
            bottom_player_history.append(bottom_player_rect)
            center_x = bottom_player_rect[0] + bottom_player_rect[2]//2
            center_y = bottom_player_rect[1] + bottom_player_rect[3]//2 # Using center y
            bottom_player_path.append((center_x, center_y))

    if not new_bottom_player_assigned_this_frame:
        frames_without_bottom_player += 1
        if bottom_player_last_seen and frames_without_bottom_player < max_frames_without_detection:
            bottom_player_rect = bottom_player_last_seen
        else:
            bottom_player_rect = None
            if frames_without_bottom_player >= max_frames_without_detection:
                bottom_player_last_seen = None
                
    new_ball_confirmed_for_tracking_this_frame = False
    candidate_for_primary_ball_data = None
    if ball_last_seen:
        if detected_ball_boxes_with_attrs:
            last_ball_center_x = ball_last_seen[0] + ball_last_seen[2] // 2
            last_ball_center_y = ball_last_seen[1] + ball_last_seen[3] // 2
            balls_near_last_known_with_dist = []
            for ball_cand_data in detected_ball_boxes_with_attrs:
                cand_rect = ball_cand_data[:4]
                cand_center_x = cand_rect[0] + cand_rect[2] // 2
                cand_center_y = cand_rect[1] + cand_rect[3] // 2
                dist_sq = (cand_center_x - last_ball_center_x)**2 + (cand_center_y - last_ball_center_y)**2
                if dist_sq < (params['ball_search_radius_px']**2):
                    balls_near_last_known_with_dist.append((ball_cand_data, dist_sq))
            if balls_near_last_known_with_dist:
                balls_near_last_known_with_dist.sort(key=lambda item: item[1])
                candidate_for_primary_ball_data = balls_near_last_known_with_dist[0][0]
    else:
        if detected_ball_boxes_with_attrs:
            # Sort by confidence or area if confidence is not reliable enough
            # Using area as a proxy for now
            sorted_by_area = sorted(detected_ball_boxes_with_attrs, key=lambda b_data: b_data[4]['area'], reverse=True) # b_data[4] is attrs
            if sorted_by_area: # Check if the list is not empty after sorting
                candidate_for_primary_ball_data = sorted_by_area[0]

    if candidate_for_primary_ball_data:
        ball_last_seen = candidate_for_primary_ball_data[:4]
        frames_without_ball = 0
        new_ball_confirmed_for_tracking_this_frame = True
        ball_history.append(ball_last_seen)
        center_x = ball_last_seen[0] + ball_last_seen[2]//2
        center_y = ball_last_seen[1] + ball_last_seen[3]//2
        ball_path.append((center_x, center_y))
        
    if not new_ball_confirmed_for_tracking_this_frame:
        frames_without_ball += 1
        if frames_without_ball >= max_frames_without_ball:
            ball_last_seen = None
            # ball_path.clear() # Optional: clear path when ball is lost
            
    balls_to_display_this_frame = []
    if params['allow_multiple_balls']:
        if detected_ball_boxes_with_attrs:
            balls_to_display_this_frame = [b[:4] for b in detected_ball_boxes_with_attrs]
    else: # Single ball mode
        if ball_last_seen and frames_without_ball < max_frames_without_ball:
            balls_to_display_this_frame = [ball_last_seen]
            
    prev_rectified_frame = rectified_frame.copy()
    return top_player_rect, bottom_player_rect, balls_to_display_this_frame

# Add function to detect balls in original (unrectified) frame
def detect_balls_in_original_frame(frame1, frame2):
    """Detect tennis balls using the original method in unrectified frames"""
    # Calculate difference between frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, params['threshold_value'], 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=params['dilate_iterations'])
    
    # Find contours in the image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by minimum area
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= params['min_contour_area']]
    
    # Group nearby contours (likely same object)
    contour_groups = group_contours(filtered_contours, params['grouping_distance'])
    
    # Store detected balls
    detected_balls = []
    
    # Process each group of contours
    for group in contour_groups:
        # Get combined bounding box for the group
        x, y, w, h = group_bounding_box(group)
        
        # Calculate area of the combined contour
        area = w * h
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Check if it meets ball criteria
        if (params['ball_min_area'] <= area <= params['ball_max_area'] and
            params['ball_min_ratio'] <= aspect_ratio <= params['ball_max_ratio']):
            
            ratio_diff = abs(aspect_ratio - 1.0)
            area_ideal = (params['ball_min_area'] + params['ball_max_area']) / 2
            # Ensure area_ideal is not zero to prevent division by zero
            area_diff = abs(area - area_ideal) / area_ideal if area_ideal > 0 else 1.0 
            
            confidence = 1.0 - (ratio_diff * 0.5 + area_diff * 0.5)
            
            ball_attrs = {
                'area': area,
                'aspect_ratio': aspect_ratio,
                'confidence': confidence,
                'frame': frame_counter # Ensure frame_counter is accessible
            }
            detected_balls.append((x, y, w, h, ball_attrs))
    
    return detected_balls

# Global variables for matplotlib figure and axes
fig_matplotlib = None
ax_matplotlib = None

# Add new function to display court.png in matplotlib
def display_court_matplotlib(ax, top_player, bottom_player, ball_objects,
                             original_court_bbox_in_rectified,
                             current_rect_w, current_rect_h):
    """Display court.png padded uniformly with white border based on a percentage of its size,
    then plot players and balls such that out-of-court positions fall within the padding."""
    import cv2
    import numpy as np

    # Load court image (keep 3 channels)
    court = cv2.imread(params['court_png_path'], cv2.IMREAD_UNCHANGED)
    if court is None:
        print(f"Cannot load {params['court_png_path']}")
        return
    # Handle alpha
    if court.ndim == 3 and court.shape[2] == 4:
        b,g,r,a = cv2.split(court)
        alpha = a.astype(np.float32)/255.0
        rgb = cv2.merge([r,g,b]).astype(np.float32)
        court_rgb = rgb*alpha[...,None] + 255*(1-alpha[...,None])
    else:
        court_rgb = cv2.cvtColor(court, cv2.COLOR_BGR2RGB).astype(np.float32)
    court_img = np.clip(court_rgb/255.0, 0.0, 1.0)

    h, w = court_img.shape[:2]
    # Percentage of padding to add on each side
    pad_pct = params.get('court_png_pad_percent', 0.3)
    pad_x = int(w * pad_pct)
    pad_y = int(h * pad_pct)

    # Create white canvas with uniform padding
    new_w = w + 2*pad_x
    new_h = h + 2*pad_y
    canvas = np.full((new_h, new_w, 3), (1.0, 0.5, 0.0), dtype=np.float32) # Orange background
    # Place court at center
    canvas[pad_y:pad_y+h, pad_x:pad_x+w] = court_img

    # Display canvas
    ax.cla()
    ax.imshow(canvas)
    ax.set_xlim(0, new_w)
    ax.set_ylim(new_h, 0)
    ax.set_aspect('equal')

    # Ensure court bbox exists
    if original_court_bbox_in_rectified is None:
        ax.axis('off'); ax.set_title('No court bbox')
        return
    x0,y0,w0,h0 = original_court_bbox_in_rectified

    # Map rectified coords to canvas coords
    def map_to_canvas(rx, ry):
        nx = (rx - x0) / w0  # <0 or >1 if outside
        ny = (ry - y0) / h0
        px = pad_x + nx * w
        py = pad_y + ny * h
        return px, py

    # Plot path of deque of (x,y)
    def plot_path(path, color):
        if not params['show_paths'] or len(path)<2: return
        pts = [map_to_canvas(x,y) for x,y in path]
        xs,ys = zip(*pts)
        ax.plot(xs, ys, color+'-', linewidth=1, alpha=0.5)

    # Top player foot
    if top_player:
        # For older detection, top_player is (x,y,w,h)
        # Path stores (center_x, center_y)
        rx = top_player[0] + top_player[2]/2
        ry = top_player[1] + top_player[3]/2 # Using center for consistency with path
        px,py = map_to_canvas(rx,ry)
        ax.plot(px, py, 'go', markersize=params['dot_size_player'])
    plot_path(top_player_path, 'g')

    # Bottom player foot
    if bottom_player:
        rx = bottom_player[0] + bottom_player[2]/2
        ry = bottom_player[1] + bottom_player[3]/2 # Using center for consistency with path
        px,py = map_to_canvas(rx,ry)
        ax.plot(px, py, 'yo', markersize=params['dot_size_player'])
    plot_path(bottom_player_path, 'y')

    # Balls center
    for b in (ball_objects or []):
        rx = b[0] + b[2]/2
        ry = b[1] + b[3]/2
        px,py = map_to_canvas(rx,ry)
        ax.plot(px, py, 'ro', markersize=params['dot_size_ball'])
    plot_path(ball_path, 'r')

    ax.axis('off')
    ax.set_title('Tennis Court Tracking')

frame_counter = 0
skip_frames = 1  # Process every frame by default

# Initialize tracking globals for the older detection logic
prev_rectified_frame = None
top_player_history = deque(maxlen=params.get('ball_history_length', 30)) # Using ball_history_length for player history too
bottom_player_history = deque(maxlen=params.get('ball_history_length', 30))
ball_history = deque(maxlen=params.get('ball_history_length', 15)) # Shorter for ball often

top_player_last_seen = None
bottom_player_last_seen = None
ball_last_seen = None
frames_without_top_player = 0
frames_without_bottom_player = 0
frames_without_ball = 0
# max_frames_without_detection should be defined, e.g., from older params or a new one
max_frames_without_detection = params.get('max_frames_player_loss', 50) # Example, add to params if needed
max_frames_without_ball = params.get('max_frames_ball_loss', 15)     # Example, add to params if needed
if 'max_frames_player_loss' not in params: params['max_frames_player_loss'] = 50
if 'max_frames_ball_loss' not in params: params['max_frames_ball_loss'] = 15


top_player_path = deque(maxlen=params['path_length'])
bottom_player_path = deque(maxlen=params['path_length'])
ball_path = deque(maxlen=params['path_length'])
all_ball_history = []


# Modify the main loop to include matplotlib visualization
while cap.isOpened():
    frame_counter += 1

    # Initialize matplotlib figure once if it hasn't been
    if fig_matplotlib is None:
        # ... (matplotlib initialization as before, using court.png aspect ratio if possible) ...
        fig_w_inches_default = 8 
        fig_h_inches_default = fig_w_inches_default * (params['court_png_height']/params['court_png_width']) 
        
        if os.path.exists(params['court_png_path']):
            try:
                temp_court_img = plt.imread(params['court_png_path'])
                img_h_px, img_w_px = temp_court_img.shape[:2]
                if img_w_px > 0 and img_h_px > 0:
                    aspect_ratio_img = float(img_h_px) / img_w_px
                    fig_h_calculated = fig_w_inches_default * aspect_ratio_img
                    fig_matplotlib, ax_matplotlib = plt.subplots(figsize=(fig_w_inches_default, fig_h_calculated))
                else:
                    fig_matplotlib, ax_matplotlib = plt.subplots(figsize=(fig_w_inches_default, fig_h_inches_default))
            except Exception as e:
                print(f"Error loading court.png for Matplotlib fig setup: {e}. Using default.")
                fig_matplotlib, ax_matplotlib = plt.subplots(figsize=(fig_w_inches_default, fig_h_inches_default))
        else:
            fig_matplotlib, ax_matplotlib = plt.subplots(figsize=(fig_w_inches_default, fig_h_inches_default))
        plt.ion()

    rectified_view = None       # Ensure these are reset/defined outside the conditional block
    perspective_matrix = None
    original_court_bbox_rect = None
    top_player_rect = None
    bottom_player_rect = None
    ball_rect_objects = [] # Initialize as empty list for safety
    rect_w_current, rect_h_current = 0,0
    court_currently_visible_by_blue = False # Initialize


    # Only process frames according to skip_frames value
    if frame_counter % skip_frames == 0:
        # Update court detection using current frame1 (older version)
        court_viz, court_corners, court_debug_masks, court_currently_visible_by_blue = detect_court(frame1)
        
        # Create rectified view ONLY if court was detected (court_corners is not None)
        if court_corners is not None:
            rectified_view, perspective_matrix, original_court_bbox_rect = rectify_court(frame1, court_corners)
            if rectified_view is not None: # Check if rectification was successful
                 rect_h_current, rect_w_current = rectified_view.shape[:2]
        
        # Calculate inverse matrix for coordinate transformation if perspective_matrix exists
        inverse_matrix = np.linalg.inv(perspective_matrix) if perspective_matrix is not None else None

    # Reset detection counters for this iteration
    humans_detected = 0
    # balls_detected = 0 # This variable seems unused, ball_rect_objects length is used

    # Prepare visualization copies
    original_frame_viz = frame1.copy() # Start with original frame
    # Overlay court_viz (which has court lines and status) onto original_frame_viz
    # This ensures court status is always on original_frame_viz
    if court_viz is not None:
         original_frame_viz = court_viz # court_viz is frame1 + court drawings + status text

    should_draw_objects = court_currently_visible_by_blue and court_detected

    # Process rectified view and detect objects ONLY if rectified_view is available
    if rectified_view is not None: # original_court_bbox also implies successful rectification
        # Call detect_objects_on_rectified_court (older version, takes only rectified_view)
        top_player_rect, bottom_player_rect, ball_rect_objects = detect_objects_on_rectified_court(rectified_view)

        rectified_viz = rectified_view.copy()
        # rect_h_current, rect_w_current are already set if rectified_view is not None
        
        # Draw divider line in rectified view
        divider_y = int(rect_h_current * 0.5) 
        cv2.line(rectified_viz, (0, divider_y), (rect_w_current, divider_y), (255,0,255), 1)

        if should_draw_objects:
            # Draw top player detection
            if top_player_rect:
                x,y,w,h = top_player_rect
                cv2.rectangle(rectified_viz, (x,y), (x+w,y+h), (0,255,0), 2)
                label = "Player 1" if frames_without_top_player == 0 else f"Player 1 (cached {frames_without_top_player})"
                cv2.putText(rectified_viz, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                humans_detected += 1
                if inverse_matrix is not None:
                    orig_rect = transform_rect_to_original(top_player_rect, inverse_matrix)
                    cv2.rectangle(original_frame_viz, (orig_rect[0],orig_rect[1]),
                                  (orig_rect[0]+orig_rect[2], orig_rect[1]+orig_rect[3]), (0,255,0), 2)
                    cv2.putText(original_frame_viz, "Player 1", (orig_rect[0],orig_rect[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Draw bottom player detection
            if bottom_player_rect:
                x,y,w,h = bottom_player_rect
                cv2.rectangle(rectified_viz, (x,y), (x+w,y+h), (0,255,255), 2)
                label = "Player 2" if frames_without_bottom_player == 0 else f"Player 2 (cached {frames_without_bottom_player})"
                cv2.putText(rectified_viz, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                humans_detected += 1
                if inverse_matrix is not None:
                    orig_rect = transform_rect_to_original(bottom_player_rect, inverse_matrix)
                    cv2.rectangle(original_frame_viz, (orig_rect[0],orig_rect[1]),
                                  (orig_rect[0]+orig_rect[2], orig_rect[1]+orig_rect[3]), (0,255,255), 2)
                    cv2.putText(original_frame_viz, "Player 2", (orig_rect[0],orig_rect[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            # Process ball detection (ball_rect_objects is always a list from the restored function)
            if ball_rect_objects: 
                for i, ball_item_rect in enumerate(ball_rect_objects):
                    x,y,w,h = ball_item_rect
                    cv2.rectangle(rectified_viz, (x,y), (x+w,y+h), (0,0,255), 2)
                    area = w*h
                    aspect_ratio = float(w)/h if h > 0 else 0
                    label_text = f"Ball"
                    if params['allow_multiple_balls'] and len(ball_rect_objects) > 1:
                        label_text += f" {i+1}"
                    is_cached_primary = False
                    if not params['allow_multiple_balls'] and len(ball_rect_objects) == 1 and \
                       ball_last_seen and ball_item_rect == ball_last_seen and \
                       frames_without_ball > 0 and frames_without_ball < max_frames_without_ball:
                        is_cached_primary = True
                    if is_cached_primary:
                        label_text += f" (cached {frames_without_ball})"
                    if params['show_ball_attributes'] or not params['allow_multiple_balls']: # Ensure params['show_ball_attributes'] exists
                         label_text += f": {area}px² | {aspect_ratio:.2f}"
                    cv2.putText(rectified_viz, label_text, 
                                (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                    if inverse_matrix is not None:
                        orig_rect = transform_rect_to_original(ball_item_rect, inverse_matrix) 
                        cv2.rectangle(original_frame_viz, (orig_rect[0],orig_rect[1]),
                                      (orig_rect[0]+orig_rect[2], orig_rect[1]+orig_rect[3]), (0,0,255), 2)
                        cv2.putText(original_frame_viz, label_text, 
                                    ( orig_rect[0], orig_rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    # else: # This else corresponds to "if rectified_view is not None"
        # If rectified_view is None, players and balls are not detected/drawn.
        # humans_detected remains 0, ball_rect_objects remains empty or None.
        # If should_draw_objects is False, humans_detected also remains 0.

    # Display overall detection information on original_frame_viz
    # The court status text is already on original_frame_viz from detect_court's court_viz
    if should_draw_objects:
        player_ball_count_text = f"Players: {humans_detected}/2, Balls: {len(ball_rect_objects) if ball_rect_objects else 0}"
    elif court_detected : # Court is detected (maybe historical) but blue ratio is currently low
        player_ball_count_text = "Players: N/A, Balls: N/A (Low Court Visibility)"
    else: # Court not detected at all
        player_ball_count_text = "Players: 0/2, Balls: 0"
        
    # The text from detect_court is at (10,30) or (10,60). This will overwrite/be overwritten if at same Y.
    # Let's ensure court status from detect_court is at (10,30) and this one at (10,60)
    # detect_court puts its primary status at (10,30), and secondary at (10,60) if using last_valid_court.
    # To avoid overlap, let's put player_ball_count_text at (10, frame_height - 20) or similar.
    # For simplicity, if detect_court uses (10,60), this will overwrite.
    # A better way would be to have detect_court return its text lines, or draw this lower.
    # For now, keeping it at (10,60) as per previous structure, but be aware of potential overlap.
    cv2.putText(original_frame_viz, player_ball_count_text, 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Moved to Y=90


    # Show all windows including the court visualization
    if params.get('show_rectified_view_window', True): # Assuming a new param, or default to True
        if rectified_view is not None: 
            cv2.namedWindow("Rectified View", cv2.WINDOW_NORMAL)
            cv2.imshow("Rectified View", rectified_viz)
            cv2.resizeWindow("Rectified View", 800, 600)
        else: 
            try:
                cv2.destroyWindow("Rectified View") 
            except cv2.error:
                pass # Window was not open, ignore.
    else:
        try:
            cv2.destroyWindow("Rectified View")
        except cv2.error:
            pass # Window was not open, ignore.
        
    if params['show_tennis_tracking_window']:
        cv2.namedWindow("Tennis Tracking", cv2.WINDOW_NORMAL)
        cv2.imshow("Tennis Tracking", original_frame_viz) 
        cv2.resizeWindow("Tennis Tracking", 800, 600)
    else:
        try:
            cv2.destroyWindow("Tennis Tracking")
        except cv2.error:
            pass # Window was not open, ignore.

    if params['show_original_video_window']:
        cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
        cv2.imshow("Original Video", frame1) 
        cv2.resizeWindow("Original Video", 800, 600)
    else:
        try:
            cv2.destroyWindow("Original Video")
        except cv2.error:
            pass # Window was not open, ignore.

    # Remove "Controls" window display
    # cv2.namedWindow("Controls", cv2.WINDOW_AUTOSIZE)
    # cv2.moveWindow("Controls", 0, 0)
    
    # ... (debug windows commented out) ...

    # Write output images for recording
    output_image = cv2.resize(original_frame_viz, (1280,720))
    out.write(output_image)

    # Add matplotlib visualization if rectified view is available AND court is detected
    if rectified_view is not None and ax_matplotlib is not None and original_court_bbox_rect is not None:
        if should_draw_objects:
            display_court_matplotlib(ax_matplotlib, top_player_rect, bottom_player_rect, ball_rect_objects, 
                                     original_court_bbox_rect, 
                                     rect_w_current, rect_h_current) 
        else: # Court might be visible (historical), but blue ratio bad for current objects
             display_court_matplotlib(ax_matplotlib, None, None, [], # No players/balls
                                     original_court_bbox_rect, 
                                     rect_w_current, rect_h_current)
        fig_matplotlib.canvas.draw_idle() 
        plt.pause(0.001) 
    elif fig_matplotlib is not None: # If no rectified view but matplotlib exists, clear it or show empty
        ax_matplotlib.cla()
        ax_matplotlib.set_title('Tennis Court Tracking (No Court Detected)')
        ax_matplotlib.axis('off')
        # Optionally, display the blank court.png if desired, or just leave it blank
        # For now, just clear and title it.
        fig_matplotlib.canvas.draw_idle()
        plt.pause(0.001)


    # Update frame sequence: shift frames for next iteration
    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret or cv2.waitKey(40) == 27:  # break on ESC or end of video
        break

cv2.destroyAllWindows()
cap.release()
out.release()

if fig_matplotlib is not None:
    plt.ioff() # Turn off interactive mode
    plt.show() # Display the final plot until closed by user