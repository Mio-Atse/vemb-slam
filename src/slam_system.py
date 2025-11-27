import os
import cv2
import numpy as np
import open3d as o3d
import glob
from datetime import datetime

from .utils import Logger, load_camera_parameters, create_timestamped_log_folder
from .pose_estimation import PoseEstimator
from .depth_estimation import create_depth_estimator
from .visualization import MatplotlibVisualizer
from .motion_estimation import MotionEstimator
from .segmentation import SegmentationProcessor

class SLAMSystem:
    def __init__(self, config):
        self.config = config
        
        # Setup Logging
        self.base_folder = create_timestamped_log_folder()
        self.logger = Logger(os.path.join(self.base_folder, 'log'))
        self.logger.log_parameters(config)
        
        # Output paths
        self.save_pcd_folder = os.path.join(self.base_folder, 'point_cloud_file')
        self.trajectory_folder = os.path.join(self.base_folder, 'camera_trajectory')
        self.frame_pcd_folder = os.path.join(self.base_folder, 'frame_point_cloud_file')
        
        # Load Camera Parameters
        self.camera_matrix, self.dist_coeffs = load_camera_parameters(config.get('calibration_file'))
        
        # Initialize Depth Estimator
        self.depth_estimator = create_depth_estimator(
            config.get('model', 'zoe'),
            config.get('calibration_file'),
            voxel_size=config.get('voxel_size', 0.02),
            min_depth=config.get('min_depth', 0.1),
            max_depth=config.get('max_depth', 10.0),
            min_brightness=config.get('min_brightness', 10)
        )
        
        # Initialize Pose Estimator
        focal_length = (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2
        principal_point = (self.camera_matrix[0, 2], self.camera_matrix[1, 2])
        self.pose_estimator = PoseEstimator(
            focal_length, 
            principal_point, 
            config.get('orb_features', 5000),
            config.get('feature_match_ratio', 0.75),
            config.get('essential_matrix_threshold', 1.0),
            config.get('frame_interval', 1)
        )
        
        # Initialize Motion Estimator (Optional)
        self.motion_estimator = None
        if config.get('use_pixhawk', False):
            self.motion_estimator = MotionEstimator(
                distance_threshold=config.get('motion_distance_threshold', 5.0),
                rotation_threshold=config.get('motion_rotation_threshold', 0.5)
            )
            if not self.motion_estimator.connect_mavlink():
                self.logger.log("Warning: Failed to connect to MAVLink. Falling back to frame interval.")
                self.motion_estimator = None

        # Initialize Segmentation (Optional)
        self.segmentation_processor = None
        if config.get('enable_segmentation', False):
            self.segmentation_processor = SegmentationProcessor(
                config.get('segmentation_model_path'),
                config.get('segmentation_model_name'),
                mode='nano' if config.get('nano_segmentation', False) else 'standard'
            )

        # SLAM State
        self.global_map = o3d.geometry.PointCloud()
        self.global_map_segmented = o3d.geometry.PointCloud() if self.segmentation_processor else None
        self.pose = np.eye(4)
        self.camera_positions = []
        self.prev_frame = None
        self.depth_prev = None
        
        # Coordinate Transforms
        self.camera_to_world = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        self.world_to_camera = np.linalg.inv(self.camera_to_world)
        
        # Visualization
        self.visualizer = None
        if config.get('real_time_vis', False):
            self.visualizer = MatplotlibVisualizer(enable_segmentation=bool(self.segmentation_processor))

    def run(self):
        try:
            if self.config.get('use_dataset', False):
                self._run_dataset()
            else:
                self._run_camera()
        except Exception as e:
            self.logger.log(f"An error occurred during SLAM processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _run_camera(self):
        # Support both video file and camera input
        video_source = self.config.get('video_path')
        if video_source is None:
            video_source = self.config.get('camera_id', 0)
            self.logger.log(f"Using camera ID: {video_source}")
        else:
            self.logger.log(f"Using video file: {video_source}")
            
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            self.logger.log(f"Error: Could not open video source: {video_source}")
            return
            
        # Set resolution only for camera (not video files)
        if isinstance(video_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        processed_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            should_process = False
            if self.motion_estimator:
                dist_met, rot_met = self.motion_estimator.update_motion_estimation()
                if dist_met or rot_met or self.prev_frame is None:
                    should_process = True
            elif frame_count % self.config.get('frame_interval', 1) == 0:
                should_process = True
                
            if should_process:
                self._process_frame(frame, frame_count)
                processed_frame_count += 1
                
            frame_count += 1
            
        cap.release()
        self.logger.log(f"Processed {processed_frame_count} out of {frame_count} frames")

    def _run_dataset(self):
        dataset_path = self.config.get('dataset_path')
        image_files = sorted(glob.glob(os.path.join(dataset_path, "*.png"))) + \
                      sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
        
        for i, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is None:
                continue
                
            self._process_frame(frame, i)

    def _process_frame(self, frame, frame_index):
        self.logger.log(f"Processing frame {frame_index}")
        
        # 1. Depth Estimation & Point Cloud Generation
        point_cloud_np, depth_map = self.depth_estimator.process_frame_with_depth(frame)
        
        if self.config.get('save_point_clouds_only', False):
            self._save_point_cloud(point_cloud_np, frame_index)
            return

        # 2. Pose Estimation
        current_pose = self._estimate_pose(frame, depth_map, frame_index)
        
        # 3. Update Global Map
        self._update_map(point_cloud_np, current_pose)
        
        # 4. Segmentation (Optional)
        if self.segmentation_processor:
            self._process_segmentation(point_cloud_np, frame, current_pose)

        # 5. Visualization
        if self.visualizer:
            self._update_visualization()

        # Update state
        self.prev_frame = frame.copy()
        self.depth_prev = depth_map.copy()

    def _estimate_pose(self, frame, depth_map, frame_index):
        if self.prev_frame is None:
            self.pose = np.eye(4)
            self.camera_positions.append(self.pose[:3, 3])
            self.pose_estimator.update_pose_history(self.pose)
            return self.pose

        # Feature Matching
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp1, desc1 = self.pose_estimator.detect_and_compute(gray_prev)
        kp2, desc2 = self.pose_estimator.detect_and_compute(gray_curr)
        
        matches = self.pose_estimator.match_features(desc1, desc2)
        
        if len(matches) < 8:
            self.logger.log(f"Frame {frame_index}: Insufficient matches. Skipping pose update.")
            return self.pose

        # 3D-2D Correspondence
        pts1_3d = []
        pts2_2d = []
        
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
        
        for i, (u, v) in enumerate(pts1):
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < self.depth_prev.shape[1] and 0 <= v_int < self.depth_prev.shape[0]:
                d = self.depth_prev[v_int, u_int]
                if d > 0:
                    # Back-project to 3D in camera frame
                    x = (u - self.camera_matrix[0, 2]) * d / self.camera_matrix[0, 0]
                    y = (v - self.camera_matrix[1, 2]) * d / self.camera_matrix[1, 1]
                    z = d
                    
                    # Transform to World Frame (using previous pose)
                    # Keep the legacy axis alignment used elsewhere in the pipeline while building the 3D-2D correspondences.
                    
                    # Original: 
                    # pt_cam = [x, y, z, 1]
                    # pt_world = self.camera_to_world @ pt_cam (This seems to be a fixed coordinate rotation, not the actual world pose)
                    
                    pt_cam = np.array([x, y, z, 1.0])
                    # This 'camera_to_world' is just a coordinate axis swap (Z-up vs Y-down)
                    pt_world_aligned = self.camera_to_world @ pt_cam 
                    pts1_3d.append(pt_world_aligned[:3])
                    pts2_2d.append(pts2[i])

        pts1_3d = np.array(pts1_3d).reshape(-1, 3)
        pts2_2d = np.array(pts2_2d).reshape(-1, 2)

        if len(pts1_3d) < 6:
            self.logger.log(f"Frame {frame_index}: Not enough 3D points. Skipping.")
            return self.pose

        # Convert back to camera coordinates for PnP (because PnP expects points in object frame, which here is our "world aligned" frame)
        pts1_3d_cam = []
        for pt in pts1_3d:
            pt_w = np.append(pt, 1.0)
            pt_c = self.world_to_camera @ pt_w
            pts1_3d_cam.append(pt_c[:3])
        pts1_3d_cam = np.array(pts1_3d_cam)

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts1_3d_cam, pts2_2d, self.camera_matrix, self.dist_coeffs
        )

        if retval:
            R_cam, _ = cv2.Rodrigues(rvec)
            T_cam = np.eye(4)
            T_cam[:3, :3] = R_cam
            T_cam[:3, 3] = tvec.flatten()
            
            # T_cam is the transform from "World Aligned" to Current Camera
            # We want the Global Pose.
            # The original code's logic:
            # T_world = self.camera_to_world @ T_cam @ self.world_to_camera
            # self.pose = self.pose @ np.linalg.inv(T_world)
            
            T_world = self.camera_to_world @ T_cam @ self.world_to_camera
            
            # Apply ICP refinement if enabled
            if self.config.get('enable_icp', False):
                 # Placeholder for ICP logic if we want to add it back
                 # For now, relying on PnP
                 pass

            self.pose = self.pose @ np.linalg.inv(T_world)
            self.pose_estimator.update_pose_history(self.pose)
            self.camera_positions.append(self.pose[:3, 3])
        else:
            self.logger.log(f"Frame {frame_index}: PnP Failed.")

        return self.pose

    def _update_map(self, point_cloud_np, pose):
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])
        current_pcd.colors = o3d.utility.Vector3dVector(point_cloud_np[:, 3:] / 255.0)
        
        # Remove outliers
        cl, ind = current_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        current_pcd = current_pcd.select_by_index(ind)
        
        # Transform to global frame
        current_pcd.transform(pose)
        
        # Downsample
        current_pcd.voxel_down_sample(voxel_size=self.config.get('voxel_size', 0.02))
        
        self.global_map += current_pcd
        # Periodic downsampling of global map to prevent memory explosion
        if len(self.global_map.points) > 1000000:
             self.global_map = self.global_map.voxel_down_sample(voxel_size=self.config.get('voxel_size', 0.02))

    def _process_segmentation(self, point_cloud_np, frame, pose):
        if not self.segmentation_processor or not getattr(self.segmentation_processor, "enabled", False):
            return

        xyz = point_cloud_np[:, :3]
        rgb = point_cloud_np[:, 3:]

        if xyz.size == 0 or rgb.size == 0:
            return
        
        # Segment
        segmented_points, labels = self.segmentation_processor.segment_frame(xyz, rgb)
        
        if segmented_points is not None and len(segmented_points) > 0:
            seg_pcd = o3d.geometry.PointCloud()
            seg_pcd.points = o3d.utility.Vector3dVector(segmented_points[:, :3])
            seg_pcd.colors = o3d.utility.Vector3dVector(segmented_points[:, 3:] / 255.0)
            
            seg_pcd.transform(pose)
            seg_pcd.voxel_down_sample(voxel_size=self.config.get('voxel_size', 0.02))
            
            self.global_map_segmented += seg_pcd

    def _update_visualization(self):
        points = np.asarray(self.global_map.points)
        colors = np.asarray(self.global_map.colors)
        
        if self.segmentation_processor:
            self.visualizer.update_segmented(
                np.asarray(self.global_map_segmented.points),
                np.asarray(self.global_map_segmented.colors)
            )
            
        self.visualizer.update(points, colors, self.pose[:3, 3])

    def _save_point_cloud(self, point_cloud_np, index):
        filename = f"frame_{index:04d}.pcd"
        filepath = os.path.join(self.frame_pcd_folder, filename)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud_np[:, 3:] / 255.0)
        o3d.io.write_point_cloud(filepath, pcd)
        self.logger.log(f"Saved {filepath}")

    def _cleanup(self):
        # Save Global Map
        output_name = self.config.get('output_pcd', 'global_map.pcd')
        if len(self.global_map.points) > 0:
            o3d.io.write_point_cloud(os.path.join(self.base_folder, output_name), self.global_map)
            self.logger.log(f"Saved global map to {output_name}")
        else:
            self.logger.log("Global map is empty; skipping save.")
        
        if self.global_map_segmented and len(self.global_map_segmented.points) > 0:
            o3d.io.write_point_cloud(os.path.join(self.base_folder, 'global_map_segmented.pcd'), self.global_map_segmented)
            self.logger.log("Saved segmented global map.")

        # Save Trajectory
        if self.camera_positions:
            traj_path = os.path.join(self.trajectory_folder, 'trajectory.txt')
            np.savetxt(traj_path, np.array(self.camera_positions), delimiter=',')
            self.logger.log(f"Saved trajectory to {traj_path}")

        if self.visualizer:
            self.visualizer.close()
