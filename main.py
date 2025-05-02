import os
import sys
import numpy as np
import cv2 as cv
import pandas as pd
import tifffile as tiff
from scipy import ndimage
from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import math
import random


from utils import utils

init()




def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def show_main_menu():
    """Display the main options menu with colors and box"""
    clear_screen()
    # Top border
    print(f"{Fore.CYAN}╔═ SPOROTRACKER{Style.RESET_ALL}")
    
    # Menu items with side borders
    print(f"{Fore.CYAN}║{Fore.GREEN} 1. {Fore.WHITE}Detect & Track Sporozoites{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 2. {Fore.WHITE}Apply Histogram Equalization{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 3. {Fore.WHITE}Apply Threshold Masking{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 4. {Fore.WHITE}Detect Edges{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 5. {Fore.WHITE}Detect Contours{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 6. {Fore.WHITE}Detect All Bodies{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 7. {Fore.WHITE}Trim tif{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 8. {Fore.WHITE}Crop tif{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN} 9. {Fore.WHITE}View Settings{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.RED} 0. {Fore.WHITE}Exit{Style.RESET_ALL}")
    return input(f"\n{Fore.YELLOW}Select an option (0-9): {Fore.WHITE}")


def info_message(msg):
    print(f"\n{Fore.YELLOW}{msg}{Style.RESET_ALL}")
    input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
    
    
def error_message(msg):
    print(f"\n{Fore.RED}{msg}{Style.RESET_ALL}")
    input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")


def exit_message(msg):
    print(f"\n{Fore.GREEN}{msg}{Style.RESET_ALL}")
    sys.exit(0)


def validate_settings(settings):
    print()



def main():

    while True:
        choice = show_main_menu()
        
        if choice == '0':
            exit_message("Bye!")
            
        elif choice == '1':
            settings = utils.read_yaml()
            _input_file_name = settings["input_file_name"]
            _input_file = tiff.imread(f'data/{_input_file_name}.tif')
            _params_mask = settings["img_mask"]
            _params_histogram = settings["img_histogram_eq"]
            _params_bfilter = settings["img_smoothing"]
            _params_canny = settings["img_edges"]
            _params_morph = settings["img_morph"]
            _params_ellipse = settings["body_detection"]
            _show_movement_mag = settings["display_move_mag"]
            _hist_points = settings["history_points"]
            _overlay_original = settings["overlay_original_frame"]
            _export_csv = settings["export_csv"]
            
            max_detections = 0
            track_data = [] 
            
            for i in tqdm(range(len(_input_file)), desc="Processing frames..."):
                frame = _input_file[i]
                norm_frame = utils.normalize_frame(frame)
                eq_frame = utils.get_histogram_equalization(norm_frame, _params_histogram)
                if _params_mask["active"]: eq_frame = utils.img_masking(eq_frame, _params_mask["pix_threshold"])
                ellipses = utils.get_ellipse_frame_data(eq_frame, _params_bfilter, _params_canny, _params_morph, _params_ellipse)
                track_data.append(ellipses)
                max_detections = len(ellipses) if len(ellipses) > max_detections else max_detections

            ellipses_array = np.full((len(_input_file), 5, max_detections), np.nan)
            
            for frame_idx, frame_ellipses in enumerate(track_data):
                if isinstance(frame_ellipses, np.ndarray) and frame_ellipses.size > 0:
                    n_ellipses = min(frame_ellipses.shape[0], max_detections)
                    for i in range(n_ellipses):
                        ellipses_array[frame_idx, :, i] = frame_ellipses[i]

            print(f" -> For {ellipses_array.shape[0]} frames, a max of {ellipses_array.shape[-1]} bodies were detected")
            
            tracked_ellipses = utils.kalman_filter_ellipse_tracking(ellipses_array)
            #tracked_ellipses = remove_incomplete_tracks(tracked_ellipses)
            
            assert tracked_ellipses.shape[0] == len(_input_file), 'Frame size missmatch'

            new_frame_array = []
            csv_data = []
            param_names = ['center_x', 'center_y', 'major_axis', 'minor_axis', 'angle']
            for i_f, frame in enumerate(_input_file):
                frame_np = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                if _overlay_original: frame_np[:, :, 1] = frame
                stats = {
                    'avg_angle_rad': 0,
                    'avg_movement_x': 0,
                    'avg_movement_y': 0,
                    'avg_magnitude': 0,
                    'valid_ellipses': 0
                }
                sum_movement_x = 0
                sum_movement_y = 0
                sum_magnitude = 0
                valid_ellipses = 0
                for obj_idx in range(max_detections):

                    ellipse_params = tracked_ellipses[i_f, :, obj_idx]
                    if np.isnan(ellipse_params[0]): continue
                    valid_ellipses += 1
                    ellipse_sum_movement_x = 0
                    ellipse_sum_movement_y = 0
                    ellipse_sum_magnitude = 0
                    if _export_csv: 
                        row = {
                            'frame': i_f,
                            'ellipse_id': obj_idx
                        }
                        for i_e, param_name in enumerate(param_names):
                            row[param_name] = ellipse_params[i_e]
                        csv_data.append(row)
                    
                    center = (int(ellipse_params[0]), int(ellipse_params[1]))
                    axes = (int(ellipse_params[2]/2), int(ellipse_params[3]/2))
                    angle = ellipse_params[4]
                    color = utils.generate_random_color(obj_idx)
                    
                    cv.ellipse(frame_np, center, axes, angle, 0, 360, color, -1)
                    cv.putText(frame_np, f"#{obj_idx}", (center[0] + 10, center[1]), cv.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                    
                    for past_frame in range(max(0, i-_hist_points), i):
                        past_params = tracked_ellipses[past_frame, :, obj_idx]
                        if not np.isnan(past_params[0]):
                            past_center = (int(past_params[0]), int(past_params[1]))
                            cv.circle(frame_np, past_center, 1, color, -1)
                            
                            if _show_movement_mag:
                                movement_x = ellipse_params[0] - past_params[0]
                                movement_y = ellipse_params[1] - past_params[1]
                                magnitude = np.sqrt(movement_x**2 + movement_y**2)      
                                ellipse_sum_movement_x += movement_x
                                ellipse_sum_movement_y += movement_y
                                ellipse_sum_magnitude += magnitude
                    if _show_movement_mag:
                        sum_movement_x += ellipse_sum_movement_x
                        sum_movement_y += ellipse_sum_movement_y
                        sum_magnitude += ellipse_sum_magnitude
                                
                if _show_movement_mag and valid_ellipses > 0:
                    stats['avg_movement_x'] = sum_movement_x / valid_ellipses
                    stats['avg_movement_y'] = sum_movement_y / valid_ellipses
                    stats['avg_magnitude'] = sum_magnitude / valid_ellipses
                    stats['valid_ellipses'] = valid_ellipses
                    frame_np = utils.draw_ellipse_movement_data(frame_np, stats, frame.shape[1])
                
                new_frame_array.append(frame_np)
                
            tiff.imwrite(f'outputs/{_input_file_name}_tracked.tif', np.array(new_frame_array))    
            if _export_csv:
                df = pd.DataFrame(csv_data)
                df.to_csv(f'outputs/{_input_file_name}_tracks.csv', index=False)
            
            
            info_message(f"File {_input_file_name}_tracked.tif saved")
            
            
        elif choice == '2':
            settings = utils.read_yaml()
            _input_file_name = settings["input_file_name"]
            _input_file = tiff.imread(f'data/{_input_file_name}.tif')
            _params_histogram = settings["img_histogram_eq"]
            track_data = [] 
            
            for i in tqdm(range(len(_input_file)), desc="Processing frames..."):
                frame = _input_file[i]
                frame_mask = utils.normalize_frame(frame)
                eq_frame = utils.get_histogram_equalization(frame_mask, _params_histogram)
                track_data.append(eq_frame)
                
            tiff.imwrite(f'outputs/{_input_file_name}_histeq.tif', np.array(track_data))    
            
            info_message(f"File {_input_file_name}_histeq.tif saved")
            
            
        elif choice == '3':
            settings = utils.read_yaml()
            _input_file_name = settings["input_file_name"]
            _input_file = tiff.imread(f'data/{_input_file_name}.tif')
            _params_mask = settings["img_mask"]
            
            track_data = [] 
            for i in tqdm(range(len(_input_file)), desc="Processing frames..."):
                frame = _input_file[i]
                eq_frame = utils.get_histogram_equalization(utils.normalize_frame(frame), _params_histogram)
                mask = utils.img_masking(eq_frame, _params_mask["pix_threshold"])
                track_data.append(mask)
                
            tiff.imwrite(f'outputs/{_input_file_name}_mask.tif', np.array(track_data))    
            
            info_message(f"File {_input_file_name}_mask.tif saved")
            
            
        
        elif choice == '4':
            settings = utils.read_yaml()
            _input_file_name = settings["input_file_name"]
            _input_file = tiff.imread(f'data/{_input_file_name}.tif')
            _params_histogram = settings["img_histogram_eq"]
            _params_bfilter = settings["img_smoothing"]
            _params_canny = settings["img_edges"]
            
            track_data = [] 
            for i in tqdm(range(len(_input_file)), desc="Processing frames..."):
                frame = _input_file[i]
                norm_frame = utils.normalize_frame(frame)
                eq_frame = utils.get_histogram_equalization(norm_frame, _params_histogram)
                edges = utils.get_edges(eq_frame, _params_bfilter, _params_canny)
                track_data.append(edges)
                
            tiff.imwrite(f'outputs/{_input_file_name}_edges.tif', np.array(track_data))    
            info_message(f"File {_input_file_name}_edges.tif saved")
            
            
        elif choice == '5':
            settings = utils.read_yaml()
            _input_file_name = settings["input_file_name"]
            _input_file = tiff.imread(f'data/{_input_file_name}.tif')
            _params_histogram = settings["img_histogram_eq"]
            _params_bfilter = settings["img_smoothing"]
            _params_canny = settings["img_edges"]
            _params_morph = settings["img_morph"]
            _params_ellipse = settings["body_detection"]
            _overlay_original = settings["overlay_original_frame"]
            
            track_data = [] 
            for i in tqdm(range(len(_input_file)), desc="Processing frames..."):
                frame = _input_file[i]
                frame_np = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                color = 222
                if _overlay_original:
                    frame_np = frame
                    frame_np = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
                    color = (0, 222, 0)
                norm_frame = utils.normalize_frame(frame)
                eq_frame = utils.get_histogram_equalization(norm_frame, _params_histogram)
                contours = utils.get_countours(eq_frame, _params_bfilter, _params_canny, _params_morph)
                
                cv.drawContours(frame_np, contours, -1, color, 1)
                track_data.append(frame_np)
            tiff.imwrite(f'outputs/{_input_file_name}_contours.tif', np.array(track_data)) 
            info_message(f"File {_input_file_name}_contours.tif saved")
            
        elif choice == '6':
            settings = utils.read_yaml()
            _input_file_name = settings["input_file_name"]
            _input_file = tiff.imread(f'data/{_input_file_name}.tif')
            _params_histogram = settings["img_histogram_eq"]
            _params_bfilter = settings["img_smoothing"]
            _params_canny = settings["img_edges"]
            _params_morph = settings["img_morph"]
            _show_movement_mag = settings["display_move_mag"]
            _hist_points = settings["history_points"]
            
            track_data = [] 
            for i in tqdm(range(len(_input_file)), desc="Processing frames..."):
                frame = _input_file[i]
                frame_np = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
                norm_frame = utils.normalize_frame(frame)
                eq_frame = utils.get_histogram_equalization(norm_frame, _params_histogram)
                contours = utils.get_countours(eq_frame, _params_bfilter, _params_canny, _params_morph)
                contours = sorted(contours, key=cv.contourArea, reverse=True)
                for i, c in enumerate(contours):
                    if c.shape[0] < 5: continue
                    area = cv.contourArea(c)
                    ellipse = cv.fitEllipse(c)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))  # Extract x,y from the center tuple
                    axes = (int(ellipse[1][0]/2), int(ellipse[1][1]/2))  # Extract major,minor axes and divide by 2
                    angle = ellipse[2]
                    cv.ellipse(frame_np, center, axes, angle, 0, 360, (0, 222, 0), 1)
                    cv.putText(frame_np, f"A: {math.ceil(area)}", center, cv.FONT_HERSHEY_SIMPLEX, 0.25, (160,200,120), 1)
                track_data.append(frame_np)
            
            tiff.imwrite(f'outputs/{_input_file_name}_dets.tif', np.array(track_data)) 
            info_message(f"File {_input_file_name}_dets.tif saved")
        else:
            error_message("Invalid option. Please try again.")
            


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}Terminating...{Style.RESET_ALL}")
        sys.exit(0)
