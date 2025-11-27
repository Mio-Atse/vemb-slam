import argparse
import math
import os
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime

import airsim
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Ensure local script directory is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add the 'zoe' and 'Metric3D' directories to the Python path
zoe_dir = os.path.join(current_dir, 'zoe')
metric3d_dir = os.path.join(current_dir, 'metric3dd')
sys.path.append(zoe_dir)
sys.path.append(metric3d_dir)

# Import point cloud generators
from point_cloud_generator import PointCloudGenerator as MidasPointCloudGenerator
from zoe.point_cloud_generator import PointCloudGenerator as ZoePointCloudGenerator
from metric3dd.point_cloud_generator import PointCloudGenerator as Metric3DPointCloudGenerator
from pose_estimator import PoseEstimator

#timestap function for folders

def create_timestamped_log_folder():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_folder = f"logs/log_{timestamp}"
    subfolders = ['log', 'point_cloud_file', 'camera_trajectory', 'frame_point_cloud_file']
    
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_folder, subfolder), exist_ok=True)
    
    return base_folder

class MotionEstimator:
    def __init__(self, distance_threshold=5.0, rotation_threshold=5):
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.last_time = None
        self.client = None
        
        # Track velocity and distance components
        self.last_vel_x = 0.0
        self.last_vel_y = 0.0
        self.total_dist_x = 0.0
        self.total_dist_y = 0.0
        
        # Add dampening factor to prevent velocity accumulation
        self.velocity_dampening = 0.95  # Reduces velocity slightly each update
        # Add minimum acceleration threshold to reduce noise
        self.min_acc_threshold = 0.1  # m/s²
        
    def connect_airsim(self):
        """
        Connect to AirSim simulator
        """
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("Connected to AirSim")
            return True
        except Exception as e:
            print(f"Failed to connect to AirSim: {str(e)}")
            return False

    def is_drone_armed(self):
        """
        Check if drone is armed/flying
        """
        state = self.client.getMultirotorState()
        return state.landed_state == airsim.LandedState.Flying

    def reset_motion_values(self):
        """
        Reset accumulated motion values
        """
        self.total_dist_x = 0.0
        self.total_dist_y = 0.0
        self.last_vel_x *= 0.5  # Reduce velocity instead of zeroing to prevent sudden changes
        self.last_vel_y *= 0.5


    def update_motion_estimation(self):
            
        try:
            current_time = time.time()
            imu_data = self.client.getImuData()
            
            # Add angular velocity threshold check
            angular_vel = abs(imu_data.angular_velocity.z_val)
            
            # Decrease frame interval during high angular velocity
            if angular_vel > self.rotation_threshold:
                self.frame_interval = 30  # Process every frame during turns
                
            # Add dampening to prevent accumulation during turns
            self.velocity_dampening = 0.95  # Reduces velocity slightly each update
            
            # Reset accumulated motion values if turn is too sharp
            if angular_vel > self.rotation_threshold * 1.5:
                self.reset_motion_values()
                
            return True

        except Exception as e:
            print(f"Error in motion estimation: {str(e)}")
            return False
        
    def get_camera_image(self):
        """
        Get camera image from AirSim
        """
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            response = responses[0]
            
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            return True, img_bgr
        except Exception as e:
            print(f"Error getting AirSim camera image: {str(e)}")
            return False, None
                        
class MatplotlibVisualizer:
    def __init__(self):
        plt.ion()  # Enable interactive mode before creating figure
        dpi = 100  # You can adjust this value if needed
        width_inches = 640 / dpi
        height_inches = 480 / dpi
        
        self.fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.points = None
        self.colors = None
        self.camera_positions = []
        self.is_running = True
        
        # Set up the plot
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('SLAM Visualization')
        
        self.fig.show()
        plt.pause(0.1)  # Add small pause to ensure window appears
    

    def close(self):
        plt.close(self.fig)  # Close the figure window
        self.is_running = False


    def update(self, points, colors, camera_position):
        if not self.is_running:
            return
            
        try:
            self.ax.cla()  # Clear the current plot
            
            # Plot point cloud
            if len(points) > 0:
                # Downsample points for visualization if necessary
                max_points = 5000
                if len(points) > max_points:
                    idx = np.random.choice(len(points), max_points, replace=False)
                    vis_points = points[idx]
                    vis_colors = colors[idx]
                else:
                    vis_points = points
                    vis_colors = colors
                
                self.ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                              c=vis_colors, s=1)
            
            # Update camera trajectory
            if camera_position is not None:
                self.camera_positions.append(camera_position)
                positions = np.array(self.camera_positions)
                self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2)
                self.ax.plot([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 'r*', markersize=10)
            
            # Adjust view based on points
            if len(points) > 0:
                max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                                    points[:, 1].max() - points[:, 1].min(),
                                    points[:, 2].max() - points[:, 2].min()]).max() / 2.0
                mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
                mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
                mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
                self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('SLAM Visualization')
            
            plt.draw()
            plt.pause(0.01)  # Add small pause to allow plot to update
            
        except Exception as e:
            print(f"Error updating visualization: {str(e)}")



class SLAMSystem:
    def __init__(self, video_path, calibration_file, orb_features = 5000, frame_interval = 30, real_time_vis=False, end_vis=True, 
                 model='zoe', save_point_clouds=False, distance_threshold=5.0, 
                 rotation_threshold=0.5, output_pcd_name='global_map.pcd', 
                 voxel_size=0.02, min_depth=2.0, max_depth=10.0, min_brightness=10, 
                 feature_match_ratio=0.75, essential_matrix_threshold=1.0, output_format='pcd'):
        
        
        
        # Initialize motion estimator
        self.motion_estimator = MotionEstimator(
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold
        )
        
        # Connect to MAVLink
        if not self.motion_estimator.connect_airsim():
            print("Warning: Failed to connect to AirSim. Using fixed frame interval.")
            self.frame_interval = frame_interval
        else:
            print("Successfully connected to AirSim for dynamic frame interval.")
        

        #model path definitions
        if model == 'midas':
            self.generator = MidasPointCloudGenerator(calibration_path=calibration_file, voxel_size=voxel_size, min_brightness = min_brightness, max_depth = max_depth, min_depth = min_depth)
            self.pcd_folder = 'pcd_midas'
        elif model == 'metric3d':
            self.generator = Metric3DPointCloudGenerator(calibration_path=calibration_file, voxel_size=voxel_size, min_brightness = min_brightness, max_depth = max_depth, min_depth = min_depth)
            self.pcd_folder = 'pcd_metric'
        else:  # default to zoe
            self.generator = ZoePointCloudGenerator(calibration_path=calibration_file, voxel_size=voxel_size, min_brightness = min_brightness, max_depth = max_depth, min_depth = min_depth)
            self.pcd_folder = 'pcd_zoe'

        self.save_point_clouds = save_point_clouds
        self.output_format = output_format
        #Folder definitions
        self.base_folder = create_timestamped_log_folder()
        self.log_folder = os.path.join(self.base_folder, 'log')
        if save_point_clouds:
            self.save_pcd_folder = os.path.join(self.base_folder, 'frame_point_cloud_file')
        else:
            self.save_pcd_folder = os.path.join(self.base_folder, 'point_cloud_file')
            self.trajectory_folder = os.path.join(self.base_folder, 'camera_trajectory')
        
        
        #local definitions of parameters
        self.model = model
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.camera_matrix, self.dist_coeffs = self.load_camera_parameters()        # Digital cameras don't have distortion
        self.dist_coeffs = np.zeros(5)
        self.frame_interval = frame_interval
        self.output_pcd_name = output_pcd_name
        self.voxel_size = voxel_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_brightness = min_brightness
        self.orb_features = orb_features
        self.feature_match_ratio = feature_match_ratio
        self.essential_matrix_threshold = essential_matrix_threshold
        
        


        #estimator definitions
        focal_length = (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2
        principal_point = (self.camera_matrix[0, 2], self.camera_matrix[1, 2])
        self.estimator = PoseEstimator(focal_length, principal_point, self.orb_features, feature_match_ratio, essential_matrix_threshold)
        

        #video capture, camera, global map, frame definitions
        #self.cap = cv2.VideoCapture(video_path)
        self.global_map = o3d.geometry.PointCloud()
        self.prev_frame = None

        self.camera_to_world = np.array([
            [0, 0, 1, 0],  # Old Z becomes new X (forward)
            [-1, 0, 0, 0], # Old X becomes negative Y (right to left)
            [0, -1, 0, 0], # Old Y becomes negative Z (up to down)
            [0, 0, 0, 1]
        ])
        
        self.world_to_camera = np.linalg.inv(self.camera_to_world)
        self.pose = np.eye(4)
        self.real_time_vis = real_time_vis
        self.end_vis = end_vis
        self.camera_positions = []
        self.pcd_list = []

        #real time visualization definitions
        self.is_running = True
        self.vis = None
        self.vis_updated = threading.Event()  # Initialize the Event

        if self.real_time_vis:
            self.vis = MatplotlibVisualizer()
            #self.vis_thread = threading.Thread(target=self.visualization_loop)
            #self.vis_thread.start()

        self.local_map_size = 20  # Number of frames to keep in local map
        self.local_maps = []  # List to store recent point clouds
        # Initialize logging
        self.log_filename = os.path.join(self.log_folder, 'slam_log.txt')
        self.log_parameters()

    def log_parameters(self):
        with open(self.log_filename, 'w') as log_file:
            log_file.write(f"SLAM Parameters Log\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            log_file.write(f"Model: {self.model}\n")
            log_file.write(f"Video Path: {self.video_path}\n")
            log_file.write(f"Calibration File: {self.calibration_file}\n")
            log_file.write(f"Output PCD Name: {self.output_pcd_name}\n")
            log_file.write(f"Voxel Size: {self.voxel_size}\n")
            log_file.write(f"Minimum Depth: {self.min_depth}\n")
            log_file.write(f"Maximum Depth: {self.max_depth}\n")
            log_file.write(f"Minimum Brightness: {self.min_brightness}\n")
            log_file.write(f"ORB Features: {self.orb_features}\n")
            log_file.write(f"Feature Match Ratio: {self.feature_match_ratio}\n")
            log_file.write(f"Essential Matrix Threshold: {self.essential_matrix_threshold}\n")

    def log_message(self, message):
        formatted_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
        print(formatted_message)
        with open(self.log_filename, 'a') as log_file:
            log_file.write(formatted_message + '\n')

   
    def load_camera_parameters(self):
        try:
            data = np.load(self.calibration_file)
            camera_matrix = data['Camera_matrix']
            dist_coeffs = data['distCoeff'].ravel()
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients: {dist_coeffs}")
            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"Error loading camera parameters: {str(e)}")
            print("Using default camera parameters")
            return np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1]]), np.zeros(5)


    def save_point_cloud(self, pcd, index):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:] / 255.0)

        filename = f"frame_{index:04d}.{self.output_format}"
        filepath = os.path.join(self.save_pcd_folder, filename)

        if self.output_format == 'pcd':
            o3d.io.write_point_cloud(filepath, pcd_o3d)
        elif self.output_format == 'txt':
            np.savetxt(filepath, pcd, fmt='%.6f', header='x y z r g b', comments='')

        print(f"Saved point cloud: {filepath}")
        

    def update_visualization(self):
        if self.real_time_vis and self.vis:
            try:
                # Convert Open3D point cloud to numpy arrays
                points = np.asarray(self.global_map.points)
                colors = np.asarray(self.global_map.colors)
                
                if len(points) > 0:  # Only update if there are points to display
                    if len(self.camera_positions) > 0:
                        camera_position = self.camera_positions[-1]
                    else:
                        camera_position = None
                        
                    self.vis.update(points, colors, camera_position)
                    
                self.vis_updated.set()
            except Exception as e:
                print(f"Visualization update error: {str(e)}")

    def visualization_loop(self):
        while self.is_running:
            self.vis_updated.wait()
            self.update_visualization()
            self.vis_updated.clear()
            time.sleep(0.1)  # Add small delay to prevent excessive updates

    
    
    def run(self):
        frame_count = 0
        processed_frame_count = 0
        depth_prev = None

        try:
            while True:
                ret, frame = self.motion_estimator.get_camera_image()
                
                if not ret:
                    self.log_message("End of video stream.")
                    break
                
               
                should_process = self.motion_estimator.update_motion_estimation()

                
                if (frame_count % self.frame_interval == 0):
                    self.log_message(f"Processing frame {frame_count}")

                    # Get point cloud in world coordinates (X-forward, Z-up)
                    point_cloud_np, depth_map = self.generator.process_frame_with_depth(frame)

                    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                    if self.save_point_clouds:
                        self.save_point_cloud(point_cloud_np, frame_count)
                    else:
                        # Create current point cloud
                        current_pcd = o3d.geometry.PointCloud()
                        current_pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])
                        current_pcd.colors = o3d.utility.Vector3dVector(point_cloud_np[:, 3:] / 255.0)

                        # Remove outliers
                        cl, ind = current_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                        current_pcd = current_pcd.select_by_index(ind)

                        if self.prev_frame is not None and depth_prev is not None:
                            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
                            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            kp1, desc1 = self.estimator.detect_and_compute(gray_prev)
                            kp2, desc2 = self.estimator.detect_and_compute(gray_current)

                            matches = self.estimator.match_features(desc1, desc2)
                            if len(matches) < 8:
                                self.log_message(f"Frame {frame_count}: Insufficient matches found ({len(matches)}). Skipping pose estimation.")
                            else:
                                # Get matched keypoints
                                pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
                                pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

                                # Initialize lists for 3D-2D correspondences
                                pts1_3d = []
                                pts2_2d = []

                                for i, (u, v) in enumerate(pts1):
                                    u_int, v_int = int(round(u)), int(round(v))
                                    if 0 <= u_int < depth_prev.shape[1] and 0 <= v_int < depth_prev.shape[0]:
                                        d = depth_prev[v_int, u_int]
                                        if d > 0:
                                            # Compute 3D point in camera coordinates
                                            x = (u - self.camera_matrix[0, 2]) * d / self.camera_matrix[0, 0]
                                            y = (v - self.camera_matrix[1, 2]) * d / self.camera_matrix[1, 1]
                                            z = d
                                            
                                            # Convert 3D point from camera to world coordinates
                                            pt_cam = np.array([x, y, z, 1.0])
                                            pt_world = self.camera_to_world @ pt_cam
                                            pts1_3d.append(pt_world[:3])
                                            pts2_2d.append(pts2[i])

                                pts1_3d = np.array(pts1_3d).reshape(-1, 3)
                                pts2_2d = np.array(pts2_2d).reshape(-1, 2)

                                if len(pts1_3d) < 6:
                                    self.log_message(f"Frame {frame_count}: Not enough 3D-2D correspondences ({len(pts1_3d)}). Skipping pose estimation.")
                                else:
                                    # Convert points back to camera coordinates for PnP
                                    pts1_3d_cam = []
                                    for pt in pts1_3d:
                                        pt_world = np.append(pt, 1.0)
                                        pt_cam = self.world_to_camera @ pt_world
                                        pts1_3d_cam.append(pt_cam[:3])
                                    
                                    pts1_3d_cam = np.array(pts1_3d_cam)

                                    # Use PnP to estimate pose in camera coordinates
                                    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                                        pts1_3d_cam, pts2_2d, self.camera_matrix, self.dist_coeffs
                                    )

                                    if retval:
                                      
                                        # Convert rotation vector to rotation matrix
                                        R_cam, _ = cv2.Rodrigues(rvec)
                                        T_cam = np.eye(4)
                                        T_cam[:3, :3] = R_cam
                                        T_cam[:3, 3] = tvec.flatten()

                                        # Convert camera transformation to world coordinates
                                        T_world = self.camera_to_world @ T_cam @ self.world_to_camera
                                        
                                        # Update the global pose
                                        self.pose = self.pose @ np.linalg.inv(T_world)

                                        # Update pose history
                                        self.estimator.update_pose_history(self.pose)
                                        
                                        # Filter outliers
                                        self.estimator.filter_trajectory_outliers()
                                        
                                        # Optimize pose graph periodically (every 10 frames)
                                        if processed_frame_count % 10 == 0:
                                            try:
                                                optimized_poses = self.estimator.optimize_pose_graph()
                                                if optimized_poses is not None:
                                                    self.pose = optimized_poses[-1]
                                            except Exception as e:
                                                self.log_message(f"Frame {frame_count}: Pose graph optimization failed: {str(e)}")
                                        
                                        try:
                                            # Apply smoothing
                                            smoothed_pose = self.estimator.smooth_trajectory()
                                            self.pose = smoothed_pose

                                            # Perform ICP with previous point cloud if available
                                            if len(self.local_maps) > 0:
                                                try:
                                                    result_icp = o3d.pipelines.registration.registration_icp(
                                                        current_pcd,
                                                        self.local_maps[-1],
                                                        max_correspondence_distance=0.05,
                                                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                                                    )
                                                    
                                                    if result_icp.fitness > 0.6:  # Good alignment threshold
                                                        current_pcd.transform(result_icp.transformation)
                                                        self.pose = self.pose @ result_icp.transformation
                                                except Exception as e:
                                                    self.log_message(f"Frame {frame_count}: ICP registration failed: {str(e)}")

                                            # Update local maps
                                            self.local_maps.append(current_pcd)
                                            if len(self.local_maps) > self.local_map_size:
                                                self.local_maps.pop(0)

                                            # Store the camera position
                                            camera_position = self.pose[:3, 3]
                                            self.camera_positions.append(camera_position)

                                        except Exception as e:
                                            self.log_message(f"Frame {frame_count}: Trajectory smoothing failed: {str(e)}")
                                            # Use unsmoothed pose as fallback
                                            camera_position = self.pose[:3, 3]
                                            self.camera_positions.append(camera_position)
                                    else:
                                        self.log_message(f"Frame {frame_count}: PnP pose estimation failed.")
                        else:
                            # For the first frame, initialize pose in world coordinates
                            self.pose = np.eye(4)
                            camera_position = self.pose[:3, 3]
                            self.camera_positions.append(camera_position)

                        # Transform current point cloud to global coordinate frame
                        current_pcd.transform(self.pose)
                        current_pcd.voxel_down_sample(voxel_size=self.voxel_size)
                        self.global_map += current_pcd
                        
                        if self.real_time_vis:
                            self.update_visualization()

                        # Store the current point cloud
                        self.pcd_list.append(current_pcd)

                        # Prepare for next iteration
                        self.prev_frame = frame.copy()
                        depth_prev = depth_map.copy()

                        processed_frame_count += 1

                frame_count += 1

        except Exception as e:
            self.log_message(f"An error occurred during SLAM processing: {str(e)}")

        finally:
            if not self.save_point_clouds:
                if len(self.global_map.points) > 0:
                    pcd_name = os.path.join(self.save_pcd_folder, self.output_pcd_name)
                    o3d.io.write_point_cloud(pcd_name, self.global_map)
                    self.log_message(f"Saved global map to '{pcd_name}'.")
                else:
                    self.log_message("Global map is empty; skipping save.")

                if self.real_time_vis:
                    self.is_running = False
                    if self.vis:
                        self.vis.close()
                
                if self.end_vis:

                    o3d.visualization.draw_geometries([self.global_map])
                
            if len(self.camera_positions) > 0:
                camera_positions = np.array(self.camera_positions)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 'r*-')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Camera Trajectory')

                cam_traj_name = os.path.join(self.trajectory_folder, 'camera_trajectory.txt')
            
                # Save camera trajectory to file
                np.savetxt(cam_traj_name, camera_positions, delimiter=",", header="X,Y,Z", comments="")
                self.log_message(f"Saved camera trajectory to {cam_traj_name}.")

def main():
    parser = argparse.ArgumentParser(description="Run SLAM system with optional real-time visualization or save point clouds")
    parser.add_argument("--video_path", type=str, default="videos/sample.mp4", help="Path to input video")
    parser.add_argument("--frame_interval", type=int, default=30, help="Frame Interval")
    parser.add_argument("--calibration_file", type=str, help="Path to camera calibration NPZ file")
    parser.add_argument("--real_time_vis", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--end_vis", action="store_false", help="Disable result visualization on Open3D")
    parser.add_argument("--model", type=str, choices=['midas', 'zoe', 'metric3d'], default='zoe', help="Choose the point cloud generation model")
    parser.add_argument("--save_point_clouds", action="store_true", help="Save point clouds as txt files without running SLAM")
    parser.add_argument("--distance_threshold", type=int, default=5, help="Minimum brightness for point cloud filtering")
    parser.add_argument("--rotation_threshold", type=float, default=0.5, help="Minimum brightness for point cloud filtering")
    parser.add_argument("--output_pcd", type=str, default="global_pcd.pcd", help="Filename for the final global point cloud (default: global_pcd.pcd)")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size for point cloud downsampling")
    parser.add_argument("--min_depth", type=float, default=2.0, help="Minimum depth for point cloud filtering")
    parser.add_argument("--max_depth", type=float, default=10.0, help="Maximum depth for point cloud filtering")
    parser.add_argument("--min_brightness", type=int, default=10, help="Minimum brightness for point cloud filtering")
    parser.add_argument("--orb_features", type=int, default=5000, help="Number of ORB features to detect")
    parser.add_argument("--feature_match_ratio", type=float, default=0.75, help="Ratio test threshold for feature matching")
    parser.add_argument("--essential_matrix_threshold", type=float, default=1.0, help="RANSAC threshold for essential matrix estimation"),
    parser.add_argument("--output_format", type=str, default="pcd", choices=["pcd", "txt"], help="Output format: pcd or txt.")
    

    args = parser.parse_args()

    slam = SLAMSystem(args.video_path,args.calibration_file,args.orb_features, args.frame_interval, args.real_time_vis, args.end_vis, args.model, args.save_point_clouds, args.distance_threshold, args.rotation_threshold, args.output_pcd, args.voxel_size, args.min_depth, args.max_depth, args.min_brightness,args.feature_match_ratio,args.essential_matrix_threshold, args.output_format)
    slam.run()
if __name__ == "__main__":
    main()
