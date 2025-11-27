import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import open3d as o3d
from abc import ABC, abstractmethod

class BaseDepthEstimator(ABC):
    def __init__(self, calibration_path, transformation=None, voxel_size=0.02, 
                 min_brightness=10, max_brightness=200, min_depth=0.1, max_depth=10.0, depth_scale=1.0):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.voxel_size = voxel_size
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale

        self.coordinate_transform = np.array([
            [0, 0, 1, 0],  # Old Z becomes new X (forward)
            [-1, 0, 0, 0], # Old X becomes negative Y (right to left)
            [0, -1, 0, 0], # Old Y becomes negative Z (up to down)
            [0, 0, 0, 1]
        ])
        
        self.transformation = np.eye(4) if transformation is None else transformation

        if calibration_path:
            self.load_camera_parameters(calibration_path)
        else:
            self.set_default_camera_parameters()

    def set_default_camera_parameters(self):
        self.FX = 525.0
        self.FY = 525.0
        self.CX = 320.0
        self.CY = 240.0
        self.dist_coeffs = np.zeros(5)
        print("Using default camera parameters")

    def load_camera_parameters(self, calibration_path):
        try:
            calibration_data = np.load(calibration_path)
            if 'Camera_matrix' in calibration_data:
                camera_matrix = calibration_data['Camera_matrix']
                self.FX = camera_matrix[0, 0]
                self.FY = camera_matrix[1, 1]
                self.CX = camera_matrix[0, 2]
                self.CY = camera_matrix[1, 2]
                print(f"Camera parameters loaded: FX: {self.FX}, FY: {self.FY}, CX: {self.CX}, CY: {self.CY}")
            else:
                raise KeyError("Camera matrix not found in calibration file")

            if 'distCoeff' in calibration_data:
                self.dist_coeffs = calibration_data['distCoeff'].ravel()
                print(f"Distortion coefficients loaded: {self.dist_coeffs}")
            else:
                self.dist_coeffs = np.zeros(5)

        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            self.set_default_camera_parameters()

    @abstractmethod
    def estimate_depth(self, image):
        """
        Estimate depth from an RGB image.
        Returns a depth map (numpy array) in meters.
        """
        pass

    def process_frame_with_depth(self, frame):
        h, w = frame.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            np.array([[self.FX, 0, self.CX], [0, self.FY, self.CY], [0, 0, 1]]),
            self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted_frame = cv2.undistort(frame,
                                          np.array([[self.FX, 0, self.CX], [0, self.FY, self.CY], [0, 0, 1]]),
                                          self.dist_coeffs,
                                          None,
                                          new_camera_matrix)

        # Estimate depth
        depth_map = self.estimate_depth(undistorted_frame)
        
        # Resize depth map to match frame if necessary
        if depth_map.shape[:2] != (h, w):
             depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Scale depth map
        scaled_depth_map = depth_map * self.depth_scale

        # Create 3D points
        rows, cols = scaled_depth_map.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.zeros((rows * cols, 3), dtype=np.float32)
        points[:, 0] = (c.flatten() - self.CX) * scaled_depth_map.flatten() / self.FX
        points[:, 1] = (r.flatten() - self.CY) * scaled_depth_map.flatten() / self.FY
        points[:, 2] = scaled_depth_map.flatten()

        # Filter points
        color_image = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        brightness = np.mean(color_image.reshape(-1, 3), axis=1)
        
        mask = (scaled_depth_map.flatten() > self.min_depth) & \
               (scaled_depth_map.flatten() < self.max_depth) & \
               (brightness > self.min_brightness) & \
               (brightness < self.max_brightness)

        valid_points = points[mask]
        valid_colors = color_image.reshape(-1, 3)[mask]

        # Apply transformations
        if len(valid_points) > 0:
            homogeneous_points = np.hstack((valid_points, np.ones((valid_points.shape[0], 1))))
            transformed_points = np.dot(homogeneous_points, self.coordinate_transform.T)
            transformed_points = np.dot(transformed_points, self.transformation.T)[:, :3]
            
            # Downsample
            if self.voxel_size > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(transformed_points)
                pcd.colors = o3d.utility.Vector3dVector(valid_colors.astype(np.float64) / 255.0)
                downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                transformed_points = np.asarray(downsampled_pcd.points)
                valid_colors = np.asarray(downsampled_pcd.colors) * 255

            point_cloud = np.hstack((transformed_points, valid_colors.astype(np.uint8)))
        else:
            point_cloud = np.empty((0, 6))

        return point_cloud, scaled_depth_map

class MiDaSGenerator(BaseDepthEstimator):
    def __init__(self, model_type="MiDaS_small", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.default_transform

    def estimate_depth(self, image):
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(color_image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()

        # Normalize and invert for MiDaS (relative depth)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map = 1 - depth_map
        return depth_map

class ZoeDepthGenerator(BaseDepthEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import pipeline
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device=0 if torch.cuda.is_available() else -1)

    def estimate_depth(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        depth_map = self.pipe(pil_image)["depth"]
        return np.array(depth_map)

class Metric3DGenerator(BaseDepthEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric3d_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
        self.metric3d_model.to(self.device)
        self.metric3d_model.eval()

    def estimate_depth(self, image):
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_depth, confidence, _ = self.metric3d_model.inference({'input': image_tensor})
            depth_map = pred_depth.squeeze().cpu().numpy()
            
        return depth_map

def create_depth_estimator(model_type, calibration_file, **kwargs):
    if model_type == 'midas':
        # Default scale for MiDaS as per original code
        kwargs.setdefault('depth_scale', 5.0)
        return MiDaSGenerator(calibration_path=calibration_file, **kwargs)
    elif model_type == 'metric3d':
        # Default scale for Metric3D as per original code
        kwargs.setdefault('depth_scale', 0.322)
        return Metric3DGenerator(calibration_path=calibration_file, **kwargs)
    else: # zoe
        # Default scale for Zoe as per original code
        kwargs.setdefault('depth_scale', 1.0)
        return ZoeDepthGenerator(calibration_path=calibration_file, **kwargs)
