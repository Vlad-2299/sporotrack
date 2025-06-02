import numpy as np
import cv2 as cv
from tqdm import tqdm
import random
import math
import yaml
from pathlib import Path

from filterpy.kalman import KalmanFilter


def read_yaml():
    with open(Path('config') / 'settings.yaml', 'r') as file:
        return yaml.safe_load(file)
    

def write_yaml(data):
    with open(Path('config') / 'settings.yaml', 'w') as file:
        yaml.safe_dump(data, file)
        


def normalize_frame(image):
    return (image / image.max() * 255).astype(np.uint8)


def img_otsu_masking(image):
    '''
    - Applies bluring while preserving edges
    - Adaptive treshold masking
    '''
    img_blur = cv.bilateralFilter(image,9,55,125)
    th = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 11, 2)
    return th


def img_masking(image, pix_threshold):
    '''
    - Applies bluring while preserving edges
    - Treshold masking
    '''
    # img_blur = cv.GaussianBlur(image,(5,5),2.5)
    # img_blur = cv.bilateralFilter(image, 5, 75,125)
    return np.where(image > pix_threshold, 1, 0).astype(np.uint8)


def get_histogram_equalization(frame, _params_histogram):
    percentile = _params_histogram["pix_bright_percent"]
    threshold = np.percentile(frame, percentile)
    weight_mask = np.clip((frame.astype(float) - threshold) / (255 - threshold), 0, 1)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(frame)
    result = (frame * (1 - weight_mask) + enhanced * weight_mask).astype(np.uint8)
    return result


def get_edges(img, params_bfilter, params_canny):
    blurred = cv.bilateralFilter(img, params_bfilter["pix_neighborhood"], params_bfilter["sigma_colorspace"], params_bfilter["sigma_coordspace"])
    edges = cv.Canny(blurred, params_canny["pix_not_edge_threshold"], params_canny["pix_is_edge_threshold"])
    return edges


def apply_morph(img, params_morph):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dilated = cv.dilate(img, kernel, iterations=params_morph["dilation_iter"])
    eroded = cv.erode(dilated, kernel, iterations=params_morph["erosion_iter"])
    return eroded
    

def get_countours(img, params_bfilter, params_canny, params_morph):
    edges = get_edges(img, params_bfilter, params_canny)
    morph = apply_morph(edges, params_morph)
    contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    return contours
    

def filter_ellipse_countours(contours, params_ellipse):
    min_area = params_ellipse["min_area"]
    max_area = params_ellipse["max_area"]
    
    elipse_valid = []
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for i, c in enumerate(contours):
        if c.shape[0] < 5: continue
        area = cv.contourArea(c)
        if not min_area <= area <= max_area: continue

        perimeter = cv.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.25: continue
        
        ellipse = cv.fitEllipse(c)
        min_diameter = math.floor(2 * math.sqrt(min_area / math.pi) * 0.8)
        max_diameter = math.ceil(2 * math.sqrt(max_area / math.pi) * 1.8)
        (center_x, center_y), (width, height), angle = ellipse
        if min_diameter <= max(width, height) <= max_diameter:
            elipse_valid.append(ellipse)

    return elipse_valid
    

def get_ellipse_frame_data(img, _params_bfilter, _params_canny, _params_morph, _params_ellipse):
    contours = get_countours(img, _params_bfilter, _params_canny, _params_morph)
    valid_ellipsis = filter_ellipse_countours(contours, _params_ellipse)
    frame_data = np.zeros((len(valid_ellipsis), 5))
    
    for i, ellipse in enumerate(valid_ellipsis):
        (center_x, center_y), (width, height), angle = ellipse
        frame_data[i] = [center_x, center_y, width, height, angle]
    
    return frame_data


def generate_random_color(index=None):
    if index is not None:
        golden_ratio = 0.618033988749895
        h = (index * golden_ratio) % 1.0 
        s = 0.8
        v = 0.95
    else:
        h = random.random()
        s = 0.7 + random.random() * 0.3 
        v = 0.7 + random.random() * 0.3
    
    hsv_color = np.uint8([[[h*179, s*255, v*255]]])
    rgb_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2RGB)[0][0]
    
    return tuple(int(c) for c in rgb_color)
    
    

def kalman_filter_ellipse_tracking(ellipse_array, _params_tracker):
    def initialize_kalman_filter():
        # Updated to track 5 parameters: center_x, center_y, major_axis, minor_axis, angle
        kf = KalmanFilter(dim_x=10, dim_z=5)
        kf.x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        kf.F = np.eye(10)
        # Position and velocity relationships
        kf.F[0, 5] = kf.F[1, 6] = kf.F[2, 7] = kf.F[3, 8] = kf.F[4, 9] = 1.0
        kf.H = np.zeros((5, 10))
        np.fill_diagonal(kf.H, 1.0)
        # Measurement noise (position, size, angle)
        kf.R = np.diag([5.0, 5.0, 2.0, 2.0, 10.0])
        kf.Q = np.eye(10) * 0.1
        kf.Q[5:, 5:] = np.eye(5) * 0.01  # Lower process noise for velocities
        kf.P = np.eye(10) * 100
        return kf
    
    def initialize_kalman_with_detection(detection):
        """Initialize Kalman filter with actual detection values"""
        kf = initialize_kalman_filter()
        # Set initial state to the actual detection
        kf.x[:5] = detection
        # Reduce initial uncertainty since we have a real measurement
        kf.P = np.eye(10) * 10
        return kf
    
    def greedy_minima_assignment(cost_matrix):
        flat_indices = np.argsort(cost_matrix.ravel())
        nrows, ncols = cost_matrix.shape
        selected_cols = set()
        row_assignments = -np.ones(nrows, dtype=int)
        
        for flat_index in flat_indices:
            row, col = divmod(flat_index, ncols)
            if row_assignments[row] == -1 and col not in selected_cols:
                row_assignments[row] = col
                selected_cols.add(col)
                if len(selected_cols) == min(nrows, ncols):
                    break
        return row_assignments

    n_frames, n_params, max_detections = ellipse_array.shape
    initial_capacity = max_detections * 2
    tracked_ellipses = np.full((n_frames, n_params, initial_capacity), np.nan)
    
    kfs = [initialize_kalman_filter() for _ in range(initial_capacity)]
    last_ellipses = np.full((initial_capacity, 5), np.inf)  # Updated to 5 parameters
    last_seen_frame = np.full(initial_capacity, -1)
    track_used = np.zeros(initial_capacity, dtype=bool)
    track_active = np.zeros(initial_capacity, dtype=bool)  # Currently active tracks
    next_track_id = 0
    
    
    for frame_idx in tqdm(range(n_frames), desc='Tracking ellipses'):
        ellipses = ellipse_array[frame_idx].T
        valid_ellipse_idx = [i for i, ellipse in enumerate(ellipses) if not np.isnan(ellipse).any()]
        
        if not valid_ellipse_idx:
            continue
        
        current_ellipses = np.array([ellipses[i][:5] for i in valid_ellipse_idx])  # Take only first 5 parameters
        n_detections = len(current_ellipses)
        
        # Update track statuses - mark tracks as inactive if missing too long
        for track_idx in range(next_track_id):
            if track_used[track_idx] and track_active[track_idx]:
                if frame_idx - last_seen_frame[track_idx] > _params_tracker['max_missing_frames']:
                    track_active[track_idx] = False
        
        
        # Multiple detections - build cost matrix for ALL used tracks
        cost_matrix = np.full((next_track_id, n_detections), np.inf)
        
        for track_idx in range(next_track_id):
            if not track_used[track_idx]:
                continue
            
            if not track_active[track_idx]:
                cost_matrix[track_idx, :] = np.inf
                continue
            
            kfs[track_idx].predict()
            
            # Kalman filter prediction 
            predicted_state = kfs[track_idx].x[:5]
            predicted_center = predicted_state[:2]
            predicted_major = predicted_state[2]
            predicted_minor = predicted_state[3]
            predicted_angle = predicted_state[4]
            
            # Get last actual detection for Euclidean distance calculation
            last_ellipse = last_ellipses[track_idx]
            last_center = last_ellipse[:2]
            last_major = last_ellipse[2]
            last_minor = last_ellipse[3]
            last_angle = last_ellipse[4]
            
            # Calculate multi-dimensional cost for each detection
            costs = []
            for det_idx in range(n_detections):
                current_ellipse = current_ellipses[det_idx]
                current_center = current_ellipse[:2]
                current_major = current_ellipse[2]
                current_minor = current_ellipse[3]
                current_angle = current_ellipse[4]
                
                # POSITION COST (60% Kalman, 40% Euclidean)
                kalman_position_dist = np.linalg.norm(predicted_center - current_center)
                euclidean_position_dist = np.linalg.norm(last_center - current_center)
                position_cost = 0.6 * kalman_position_dist + 0.4 * euclidean_position_dist
                
                # SIZE COST (60% Kalman, 40% Euclidean)
                avg_major_k = (predicted_major + current_major) / 2
                avg_minor_k = (predicted_minor + current_minor) / 2
                if avg_major_k > 0 and avg_minor_k > 0:
                    kalman_size_dist = (abs(predicted_major - current_major) / avg_major_k + 
                                    abs(predicted_minor - current_minor) / avg_minor_k) * 50
                else:
                    kalman_size_dist = 1000
                
                # Euclidean size distance
                avg_major_e = (last_major + current_major) / 2
                avg_minor_e = (last_minor + current_minor) / 2
                if avg_major_e > 0 and avg_minor_e > 0:
                    euclidean_size_dist = (abs(last_major - current_major) / avg_major_e + 
                                        abs(last_minor - current_minor) / avg_minor_e) * 50
                else:
                    euclidean_size_dist = 1000
                
                size_cost = 0.6 * kalman_size_dist + 0.4 * euclidean_size_dist
                
                # ANGLE COST (60% Kalman, 40% Euclidean)
                # Kalman angle distance
                kalman_angle_diff = abs(predicted_angle - current_angle)
                kalman_angle_diff = min(kalman_angle_diff, 180 - kalman_angle_diff)
                kalman_angle_dist = kalman_angle_diff * 2
                
                # Euclidean angle distance
                euclidean_angle_diff = abs(last_angle - current_angle)
                euclidean_angle_diff = min(euclidean_angle_diff, 180 - euclidean_angle_diff)
                euclidean_angle_dist = euclidean_angle_diff * 2
                
                angle_cost = 0.6 * kalman_angle_dist + 0.4 * euclidean_angle_dist
                
                # Combined weighted cost (no area cost)
                total_cost = (position_cost + size_cost + angle_cost) 
                
                costs.append(total_cost)
            
            cost_matrix[track_idx, :] = costs
        
        assignments = greedy_minima_assignment(cost_matrix)
        assigned_detections = set()
        
        for track_idx, det_idx in enumerate(assignments):
            if det_idx == -1 or not track_used[track_idx] or not track_active[track_idx]:
                continue
            
            current_ellipse = current_ellipses[det_idx]
            predicted_center = last_ellipses[track_idx][:2]
            current_center = current_ellipse[:2]
            center_distance = np.linalg.norm(predicted_center - current_center)

            if center_distance > _params_tracker['max_reassign_dist']: continue

            assigned_detections.add(det_idx)
            
            kfs[track_idx].update(current_ellipse)
            tracked_ellipses[frame_idx, :5, track_idx] = current_ellipse  # Only store first 5 parameters
            last_ellipses[track_idx] = current_ellipse
            last_seen_frame[track_idx] = frame_idx
        
        # Create new tracks for unassigned detections
        for det_idx in range(n_detections):
            if det_idx in assigned_detections:
                continue
            
            new_track_idx = next_track_id
            next_track_id += 1
            
            if new_track_idx >= tracked_ellipses.shape[2]:
                new_capacity = tracked_ellipses.shape[2] * 2
                new_tracked = np.full((n_frames, n_params, new_capacity), np.nan)
                new_tracked[:, :, :tracked_ellipses.shape[2]] = tracked_ellipses
                tracked_ellipses = new_tracked
                
                last_ellipses = np.vstack([last_ellipses, np.full((new_capacity - last_ellipses.shape[0], 5), np.inf)])  # 5 parameters
                last_seen_frame = np.append(last_seen_frame, np.full(new_capacity - len(last_seen_frame), -1))
                track_used = np.append(track_used, np.zeros(new_capacity - len(track_used), dtype=bool))
                track_active = np.append(track_active, np.zeros(new_capacity - len(track_active), dtype=bool))
                
                for _ in range(new_capacity - len(kfs)):
                    kfs.append(initialize_kalman_filter())
            
            current_ellipse = current_ellipses[det_idx]
            kfs[new_track_idx] = initialize_kalman_with_detection(current_ellipse)
            kfs[new_track_idx].update(current_ellipse)
            tracked_ellipses[frame_idx, :5, new_track_idx] = current_ellipse  
            last_ellipses[new_track_idx] = current_ellipse
            last_seen_frame[new_track_idx] = frame_idx
            track_used[new_track_idx] = True
            track_active[new_track_idx] = True
    
    return tracked_ellipses


def remove_incomplete_tracks(tracks, portion=.01):
    assert portion >= 0 and portion <= 1, f'Choose a valid portion (0-1) of missing data'
    valid_indices = []

    for mosq_idx in range(tracks.shape[-1]):
        if np.sum(np.isnan(tracks[:, 0, mosq_idx]))/tracks.shape[0] < portion:
            valid_indices.append(mosq_idx)
    
    if valid_indices:
        return tracks[:, :, valid_indices]
    else:
        return np.zeros((tracks.shape[0], tracks.shape[1], 0))


def draw_ellipse_movement_data(frame, stats, original_width):
    if stats['valid_ellipses'] == 0: return frame

    if 'avg_magnitude' in stats and stats['avg_magnitude'] > 0:
        height = frame.shape[0]
        origin = math.floor(height * 0.12)
        max_radius = math.floor(origin * 0.80)
        
        arrow_start_movement = (origin, origin)
        movement_magnitude = np.sqrt(stats['avg_movement_x']**2 + stats['avg_movement_y']**2)
        
        if movement_magnitude > 0:
            unit_x = stats['avg_movement_x'] / movement_magnitude
            unit_y = stats['avg_movement_y'] / movement_magnitude
            
            scaled_magnitude = min(movement_magnitude, max_radius)
            
            arrow_end_movement = (
                int(arrow_start_movement[0] + unit_x * scaled_magnitude),
                int(arrow_start_movement[1] + unit_y * scaled_magnitude)
            )
        else:
            arrow_end_movement = arrow_start_movement
        
        # Draw the arrow
        cv.arrowedLine(frame, arrow_start_movement, arrow_end_movement, (160,200,120), 1)
        cv.circle(frame, arrow_start_movement, max_radius, (20,61,96), 1)
        # Display the magnitude value
        cv.putText(frame, f"{stats['avg_magnitude']:.2f}", 
                 (arrow_start_movement[0], arrow_start_movement[1] - 10),
                 cv.FONT_HERSHEY_SIMPLEX, 0.2, (221,235,157), 1)
    
    return frame


def add_histogram_to_image(h_frame, frame_np):
    hist = cv.calcHist([h_frame], [0], None, [25], [0, 256])
    
    hist_height = 100  # Height of histogram area
    
    epsilon = 1e-5
    log_hist = np.log(hist + epsilon)
    
    hist_normalized = log_hist * (hist_height / log_hist.max()) if log_hist.max() > 0 else log_hist
    
    for i in range(len(hist)):
        if hist[i][0] > 0 and hist_normalized[i][0] < 2:
            hist_normalized[i][0] = 2
    
    padding_height = hist_height + 10  # Add 10px for padding
    padded_image = np.zeros((frame_np.shape[0] + padding_height, frame_np.shape[1], 3), dtype=np.uint8)
    
    padded_image[:frame_np.shape[0], :, :] = frame_np
    
    hist_y_offset = frame_np.shape[0] + 5  # 5px padding from the image
    
    bar_width = frame_np.shape[1] // 25
    
    for i in range(25):
        h = int(hist_normalized[i][0])
        x_pos = i * bar_width
        
        if hist[i][0] > 0:  # Check original histogram for non-zero values
            cv.rectangle(
                padded_image,
                (x_pos, hist_y_offset + hist_height - h),
                (x_pos + bar_width - 1, hist_y_offset + hist_height),
                (255, 255, 255),
                -1 
            )
    
    return padded_image

