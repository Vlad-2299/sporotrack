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
    return np.where(image > pix_threshold, image, 0).astype(np.uint8)


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
    kernel = np.ones((3, 3), np.uint8)
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
    
    

def kalman_filter_ellipse_tracking(frames):
    def initialize_kalman_filter():
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([0., 0., 0., 0.])  # Initial state
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                        [0, 1, 0, 0]])
        kf.R = np.eye(2) * 0.05  # Measurement noise
        kf.Q = np.diag([0.1, 0.1, 0.01, 0.01])  # Process noise
        kf.P = np.diag([1, 1, 1, 1])
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

    n_frames, _, initial_n_tracks = frames.shape
    print(frames.shape)
    output = np.full((n_frames, 5, initial_n_tracks), np.nan)  # 5 parameters for ellipses
    
    kfs = [initialize_kalman_filter() for _ in range(initial_n_tracks)]
    last_positions = np.full((initial_n_tracks, 2), np.inf)  # Only tracking center positions
    
    current_n_tracks = initial_n_tracks  # Keep track of the current number of tracks
    
    for frame_idx in tqdm(range(n_frames), desc='Tracking bodies...'):
        ellipses = frames[frame_idx].T  # Shape: (n_ellipses, 5)
        valid_ellipses_idx = [i for i, ellipse in enumerate(ellipses) if not np.isnan(ellipse).any()]
        
        if not valid_ellipses_idx: continue  # No valid detections, skip this frame
            
        n_valid_detections = len(valid_ellipses_idx)
        
        if n_valid_detections > current_n_tracks:
            for _ in range(n_valid_detections - current_n_tracks):
                kfs.append(initialize_kalman_filter())
                last_positions = np.vstack([last_positions, [np.inf, np.inf]])
            current_n_tracks = n_valid_detections
            
        if current_n_tracks > output.shape[2]:
            new_capacity = max(current_n_tracks, output.shape[2] * 2)
            new_output = np.full((n_frames, 5, new_capacity), np.nan)
            new_output[:, :, :output.shape[2]] = output  # Copy old data
            output = new_output
            
        cost_matrix = np.full((current_n_tracks, n_valid_detections), np.inf)
        
        for obj_idx in range(current_n_tracks):
            kfs[obj_idx].predict()
            predicted_c = kfs[obj_idx].x[:2] 
            
            if np.isinf(last_positions[obj_idx]).any(): continue  # Skip objects with no previous valid detection
                
            current_centers = np.array([ellipses[idx][:2] for idx in valid_ellipses_idx])
            kalman_distances = np.linalg.norm(predicted_c - current_centers, axis=1)
            euclid_distances = np.linalg.norm(last_positions[obj_idx] - current_centers, axis=1) if frame_idx > 0 else np.zeros(len(valid_ellipses_idx))
            
            combined_distances =  kalman_distances + euclid_distances
            cost_matrix[obj_idx, :] = combined_distances
            
        associations = greedy_minima_assignment(cost_matrix)
        
        for c, r in enumerate(associations):
            if r == -1:
                continue
                
            current_ellipse = ellipses[valid_ellipses_idx[r]]
            current_center = current_ellipse[:2]
            
            # Check if distance from last position is reasonable
            if np.isinf(last_positions[c]).any() or np.linalg.norm(current_center - last_positions[c]) < 75 or c < len(last_positions):
                # Update Kalman filter with just the center position
                kfs[c].update(current_center)
                output[frame_idx, :, c] = current_ellipse
                last_positions[c] = current_center
            else:
                # New object detected
                new_idx = np.sum(~np.isnan(output[0, 0]))
                if new_idx >= output.shape[2]:
                    output = np.concatenate([output, np.full((n_frames, 5, 1), np.nan)], axis=2)
                    last_positions = np.vstack([last_positions, [np.inf, np.inf]])
                
                output[frame_idx, :, new_idx] = current_ellipse
                last_positions[new_idx] = current_center
                
                kfs.append(initialize_kalman_filter())
                kfs[-1].update(current_center)
    
    # Trim output to actual tracks
    #valid_indices = ~np.isnan(output[0, 0])
    #output = output[:, :, valid_indices]
    
    return output


def remove_incomplete_tracks(tracks, portion=.90):
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

