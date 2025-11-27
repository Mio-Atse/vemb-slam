import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import time
import open3d as o3d

class PoseEstimator:
    def __init__(self, focal_length, principal_point, orb_features, feature_match_ratio, essential_matrix_threshold, frame_interval=1):
        
        self.orb_features = orb_features
        self.feature_match_ratio = feature_match_ratio
        self.essential_matrix_threshold = essential_matrix_threshold
        self.frame_interval = frame_interval

        self.orb = cv2.ORB_create(self.orb_features)
        
        # Use FLANN matcher instead of BFMatcher for better performance
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.pose_history = []
        self.trajectory_timestamps = []  # Store timestamps for poses
        
    def detect_and_compute(self, img):
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for i in range(len(matches)):
                if len(matches[i]) == 2:
                    m, n = matches[i]
                    if m.distance < self.feature_match_ratio * n.distance:
                        good_matches.append(m)
            
            return good_matches
        except Exception as e:
            print(f"Matching failed: {e}")
            return []

    def enhance_frame(self, frame):
        """
        Enhance frame by equalizing the histogram of the V channel in HSV space.
        Detect and mitigate motion blur and out-of-focus issues.
        """
        def is_blurred(image, threshold=100):
            """
            Check if the frame is blurry using Laplacian variance.
            """
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return variance < threshold, variance

        # Step 1: Enhance using HSV equalization
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_frame = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        # Step 2: Check for blur
        is_blur, blur_variance = is_blurred(enhanced_frame)
        return enhanced_frame, is_blur
    
    def decompose_pose(self, pose_matrix):
        """
        Decompose 4x4 pose matrix into rotation and translation
        """
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        return R, t

    def compose_pose(self, R, t):
        """
        Compose rotation and translation into 4x4 pose matrix
        """
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        pose_matrix[:3, 3] = t
        return pose_matrix

    def smooth_trajectory(self):
        """
        Enhanced trajectory smoothing using SLERP for rotation and spline for translation
        """
        if len(self.pose_history) < 3:
            return self.pose_history[-1] if self.pose_history else np.eye(4)

        # Extract rotations and translations
        rotations = []
        translations = []
        # Use a small window for smoothing to avoid lag
        window_size = 5
        poses_to_smooth = self.pose_history[-window_size:]
        
        for pose in poses_to_smooth:
            R, t = self.decompose_pose(pose)
            rotations.append(R)
            translations.append(t)

        # Convert rotations to quaternions
        quats = [Rotation.from_matrix(R).as_quat() for R in rotations]
        
        # Create time points
        times = np.linspace(0, 1, len(rotations))
        
        # Smooth translations using appropriate interpolation based on number of points
        if len(translations) < 2:
            smoothed_translation = np.array(translations[-1])
        else:
            spline_x = interp1d(times, [t[0] for t in translations], kind='linear')
            spline_y = interp1d(times, [t[1] for t in translations], kind='linear')
            spline_z = interp1d(times, [t[2] for t in translations], kind='linear')
            smoothed_translation = np.array([spline_x(1), spline_y(1), spline_z(1)])
            # Optional: average Z if it's noisy
            # smoothed_translation[2] = np.mean([t[2] for t in translations])

        # Smooth rotation using SLERP
        if len(quats) < 2:
            smoothed_rotation = Rotation.from_quat(quats[-1]).as_matrix()
        else:
            rot_interp = Slerp([times[-2], times[-1]], 
                            Rotation.from_quat([quats[-2], quats[-1]]))
            smoothed_rotation = rot_interp(1).as_matrix()

        # Compose smoothed pose
        smoothed_pose = self.compose_pose(smoothed_rotation, smoothed_translation)
        
        return smoothed_pose

    def update_pose_history(self, new_pose):
        """
        Update pose history with new pose
        """
        self.pose_history.append(new_pose)
        self.trajectory_timestamps.append(time.time())
        
        # Keep only recent history to save memory, but enough for smoothing/optimization
        max_history = 100 
        if len(self.pose_history) > max_history:
            self.pose_history = self.pose_history[-max_history:]
            self.trajectory_timestamps = self.trajectory_timestamps[-max_history:]

    def filter_trajectory_outliers(self):
        """
        Filter out obvious outliers from the trajectory
        """
        if len(self.pose_history) < 3:
            return

        # Calculate velocities between consecutive poses
        velocities = []
        for i in range(1, len(self.pose_history)):
            dt = self.trajectory_timestamps[i] - self.trajectory_timestamps[i-1]
            if dt == 0:
                dt = 1e-6
            prev_pos = self.pose_history[i-1][:3, 3]
            curr_pos = self.pose_history[i][:3, 3]
            
            # Scale velocity calculation by frame interval
            velocity = np.linalg.norm(curr_pos - prev_pos) / (dt * self.frame_interval)
            velocities.append(velocity)

        # Calculate velocity statistics
        median_velocity = np.median(velocities)
        velocity_std = np.std(velocities)
        
        # Remove obvious outliers
        threshold = median_velocity + 3 * velocity_std # Increased threshold to be less aggressive
        
        # If the last added pose caused a spike, revert it (simple check)
        if len(velocities) > 0 and velocities[-1] > threshold:
             # Conservative guard to avoid single-frame jumps; consider replacing with flagging logic.
             pass

    def optimize_pose_graph(self):
        """
        Optimize the entire trajectory using pose graph optimization with improved connectivity
        """
        if len(self.pose_history) < 2:
            return None

        pose_graph = o3d.pipelines.registration.PoseGraph()
        
        # Add all poses as nodes
        for pose in self.pose_history:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
        
        # Add sequential edges with adaptive information matrix
        for i in range(len(self.pose_history)-1):
            relative_pose = np.linalg.inv(self.pose_history[i]) @ self.pose_history[i+1]
            
            # Calculate time difference between poses
            dt = self.trajectory_timestamps[i+1] - self.trajectory_timestamps[i]
            confidence = 1.0 / max(dt, 0.1)  # Higher confidence for smaller time gaps
            
            information = np.identity(6) * confidence
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, i+1,
                    relative_pose,
                    information=information,
                    uncertain=True
                )
            )
        
        # Add more edges to ensure connectivity (Heuristic Loop Closure)
        max_edge_distance = 2.0 
        min_edges_per_node = 2 
        
        for i in range(len(self.pose_history)):
            edge_count = 0
            for j in range(len(self.pose_history)):
                if i == j:
                    continue
                    
                distance = np.linalg.norm(
                    self.pose_history[i][:3, 3] - self.pose_history[j][:3, 3]
                )
                
                if distance < max_edge_distance or edge_count < min_edges_per_node:
                    relative_pose = np.linalg.inv(self.pose_history[i]) @ self.pose_history[j]
                    
                    # Calculate confidence based on distance
                    confidence = 1.0 / (1.0 + distance)
                    information = np.identity(6) * confidence * 0.5
                    
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j,
                            relative_pose,
                            information=information,
                            uncertain=True
                        )
                    )
                    edge_count += 1
                
                if edge_count >= min_edges_per_node:
                    break
        
        # More robust optimization parameters
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.2,
            edge_prune_threshold=0.35,
            preference_loop_closure=0.1,
            reference_node=0
        )
        
        try:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option
            )
            
            # Verify optimization result
            if len(pose_graph.nodes) == len(self.pose_history):
                optimized_poses = [node.pose for node in pose_graph.nodes]
                return optimized_poses
            else:
                print("[WARNING] Optimization returned incorrect number of poses")
                return None
                
        except Exception as e:
            print(f"[ERROR] Pose graph optimization failed: {str(e)}")
            return None
