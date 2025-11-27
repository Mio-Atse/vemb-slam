import os
import numpy as np
from datetime import datetime

def create_timestamped_log_folder(base_log_dir="logs"):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_folder = os.path.join(base_log_dir, f"log_{timestamp}")
    subfolders = ['log', 'point_cloud_file', 'camera_trajectory', 'frame_point_cloud_file']
    
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_folder, subfolder), exist_ok=True)
    
    return base_folder

def load_camera_parameters(calibration_file):
    try:
        data = np.load(calibration_file)
        if 'Camera_matrix' in data:
            camera_matrix = data['Camera_matrix']
        else:
            raise KeyError("Camera_matrix not found in calibration file")
            
        if 'distCoeff' in data:
            dist_coeffs = data['distCoeff'].ravel()
        else:
            dist_coeffs = np.zeros(5)
            
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading camera parameters: {str(e)}")
        print("Using default camera parameters")
        return np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1]]), np.zeros(5)

class Logger:
    def __init__(self, log_folder):
        self.log_filename = os.path.join(log_folder, 'slam_log.txt')
        
    def log(self, message):
        formatted_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
        print(formatted_message)
        with open(self.log_filename, 'a') as log_file:
            log_file.write(formatted_message + '\n')
            
    def log_parameters(self, params):
        with open(self.log_filename, 'w') as log_file:
            log_file.write(f"SLAM Parameters Log\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for key, value in params.items():
                log_file.write(f"{key}: {value}\n")
