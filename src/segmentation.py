import torch
import numpy as np
import importlib
import torch.nn.functional as F
import sys
import os
from torch.serialization import add_safe_globals

# Add the directory containing your model to the Python path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SegmentationProcessor:
    def __init__(self, model_path, model_name, mode='standard', num_classes=13):
        self.mode = mode
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Segmentation using device: {self.device} in {self.mode} mode")
        
        self.classifier = self.load_classifier(model_path, model_name)
        self.enabled = self.classifier is not None
        if not self.enabled:
            print("Segmentation disabled: classifier failed to load.")
        
        self.color_map = {
            'ceiling': [0, 255, 0],
            'floor': [0, 0, 255],
            'wall': [0, 255, 255],
            'beam': [255, 255, 0],
            'column': [255, 0, 255],
            'window': [100, 100, 255],
            'door': [200, 200, 100],
            'table': [170, 120, 200],
            'chair': [255, 0, 0],
            'sofa': [200, 100, 100],
            'bookcase': [10, 200, 100],
            'board': [200, 200, 200],
            'clutter': [50, 50, 50]
        }
        self.label_to_names = {i: name for i, name in enumerate(self.color_map.keys())}
        
        # Nano specific constants
        if self.mode == 'nano':
            self.batch_size = 2
            self.num_point = 2048
            self.block_size = 1.0
            self.stride = 0.75
            self.color_map_np = np.array(list(self.color_map.values()), dtype=np.uint8)
        else:
            self.batch_size = 32
            self.num_point = 4096
            self.block_size = 1.0
            self.stride = 0.5

    def _load_checkpoint(self, model_path, map_location):
        """
        Handle PyTorch 2.6+ weights_only default and allowlist numpy scalars used in older checkpoints.
        """
        last_error = None

        # Preferred: explicitly disable weights_only to allow legacy checkpoints that include numpy scalars.
        try:
            return torch.load(model_path, map_location=map_location, weights_only=False)
        except TypeError as e:
            last_error = e  # Torch <2.6 without weights_only arg
        except Exception as e:
            last_error = e
            if "Unsupported global" in str(e):
                try:
                    add_safe_globals([np.core.multiarray.scalar])
                    return torch.load(model_path, map_location=map_location, weights_only=False)
                except Exception as inner:
                    last_error = inner

        # Fallback: legacy torch.load signature.
        try:
            return torch.load(model_path, map_location=map_location)
        except Exception as e:
            last_error = e

        raise last_error

    def load_classifier(self, model_path, model_name):
        try:
            # Simply import the module - works with src.models.pointnet2_sem_seg format
            MODEL = importlib.import_module(model_name)
            classifier = MODEL.get_model(self.num_classes)
            
            map_location = 'cpu' if self.mode == 'nano' else self.device
            checkpoint = self._load_checkpoint(model_path, map_location=map_location)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            classifier.load_state_dict(new_state_dict)
            
            classifier = classifier.to(self.device)
            classifier.eval()
            
            if self.mode == 'nano':
                torch.cuda.empty_cache()
                
            return classifier
        except Exception as e:
            print(f"Failed to load segmentation model: {e}")
            return None

    def segment_frame(self, xyz, rgb):
        if self.classifier is None or not self.enabled:
            return None, None

        # Gracefully skip if point cloud is empty
        if xyz is None or rgb is None:
            return None, None
        if isinstance(xyz, np.ndarray):
            if xyz.size == 0:
                return None, None
        elif hasattr(xyz, "numel") and xyz.numel() == 0:
            return None, None
        if isinstance(rgb, np.ndarray) and rgb.size == 0:
            return None, None
        if hasattr(rgb, "numel") and rgb.numel() == 0:
            return None, None

        if self.mode == 'nano':
            return self._segment_frame_nano(xyz, rgb)
        else:
            return self._segment_frame_standard(xyz, rgb)

    def _segment_frame_standard(self, xyz, rgb):
        # Convert inputs to PyTorch tensors and move to GPU immediately
        xyz_t = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        rgb_t = torch.tensor(rgb, dtype=torch.float32, device=self.device)
        
        # Perform segmentation
        labels = self._segment_point_cloud_with_voting_standard(xyz_t, rgb_t)
        
        # Efficient color mapping using pre-computed color tensor
        color_map_tensor = torch.tensor(list(self.color_map.values()), device=self.device)
        segmented_colors = color_map_tensor[labels]
        
        # Combine results efficiently
        segmented_points = torch.cat((xyz_t, segmented_colors), dim=1)
        
        return segmented_points.cpu().numpy(), labels.cpu().numpy()

    def _segment_point_cloud_with_voting_standard(self, xyz, rgb, num_votes=3):
        if xyz.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        num_points = xyz.size(0)
        vote_label_pool = torch.zeros((num_points, self.num_classes), dtype=torch.float32, device=self.device)
        
        random_offsets = torch.rand((num_votes, *xyz.shape), device=self.device) * 0.02 - 0.01
        
        for vote in range(num_votes):
            xyz_vote = xyz + random_offsets[vote]
            blocks = self._split_point_cloud_into_blocks_standard(xyz_vote, rgb)
            
            batch_blocks = []
            batch_indices = []
            current_batch = []
            current_indices = []
            
            for block_points, block_point_indices in blocks:
                current_batch.append(block_points)
                current_indices.append(block_point_indices)
                
                if len(current_batch) == self.batch_size:
                    batch_blocks.append((torch.stack(current_batch), current_indices))
                    current_batch = []
                    current_indices = []
            
            if current_batch:
                batch_blocks.append((torch.stack(current_batch), current_indices))
            
            for batch_points, batch_indices in batch_blocks:
                batch_points = batch_points.transpose(2, 1)
                
                with torch.no_grad():
                    seg_pred, _ = self.classifier(batch_points)
                    batch_pred_label = seg_pred.argmax(dim=2)
                
                for b in range(len(batch_indices)):
                    one_hot = F.one_hot(batch_pred_label[b], num_classes=self.num_classes).float()
                    vote_label_pool.index_add_(0, batch_indices[b], one_hot)
        
        final_pred_labels = vote_label_pool.argmax(dim=1)
        return final_pred_labels

    def _split_point_cloud_into_blocks_standard(self, xyz, rgb):
        if xyz.numel() == 0:
            return []

        coord_min = xyz.min(dim=0)[0]
        coord_max = xyz.max(dim=0)[0]
        
        grid_x = int(torch.ceil((coord_max[0] - coord_min[0] - self.block_size) / self.stride)) + 1
        grid_y = int(torch.ceil((coord_max[1] - coord_min[1] - self.block_size) / self.stride)) + 1
        
        x_starts = coord_min[0] + torch.arange(grid_x, device=self.device) * self.stride
        y_starts = coord_min[1] + torch.arange(grid_y, device=self.device) * self.stride
        
        x_starts, y_starts = torch.meshgrid(x_starts, y_starts, indexing='ij')
        block_centers = torch.stack([x_starts.flatten(), y_starts.flatten()], dim=1)
        
        blocks = []
        batch_size = 64
        for i in range(0, len(block_centers), batch_size):
            batch_centers = block_centers[i:i + batch_size]
            
            s_x = batch_centers[:, 0].unsqueeze(1)
            s_y = batch_centers[:, 1].unsqueeze(1)
            e_x = s_x + self.block_size
            e_y = s_y + self.block_size
            
            x_valid = (xyz[:, 0].unsqueeze(0) >= s_x) & (xyz[:, 0].unsqueeze(0) <= e_x)
            y_valid = (xyz[:, 1].unsqueeze(0) >= s_y) & (xyz[:, 1].unsqueeze(0) <= e_y)
            valid_points = x_valid & y_valid
            
            for b in range(len(batch_centers)):
                block_mask = valid_points[b]
                if not block_mask.any():
                    continue
                    
                block_point_indices = torch.where(block_mask)[0]
                block_xyz = xyz[block_point_indices]
                block_rgb = rgb[block_point_indices]
                
                block_center = torch.tensor([s_x[b].item() + self.block_size/2, s_y[b].item() + self.block_size/2, 0], device=self.device)
                block_xyz_centered = block_xyz - block_center
                block_xyz_normalized = block_xyz / coord_max
                block_rgb_normalized = block_rgb / 255.0
                
                block_points = torch.cat((block_xyz_centered, block_rgb_normalized, block_xyz_normalized), dim=1)
                
                if block_points.size(0) >= self.num_point:
                    idx = torch.randperm(block_points.size(0), device=self.device)[:self.num_point]
                    block_points = block_points[idx]
                    block_point_indices = block_point_indices[idx]
                else:
                    idx = torch.randint(block_points.size(0), (self.num_point - block_points.size(0),), device=self.device)
                    block_points = torch.cat((block_points, block_points[idx]), dim=0)
                    block_point_indices = torch.cat((block_point_indices, block_point_indices[idx]))
                
                blocks.append((block_points, block_point_indices))
        
        return blocks

    def _segment_frame_nano(self, xyz, rgb):
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.float32)
        
        labels = self._segment_point_cloud_with_voting_nano(xyz, rgb)
        
        segmented_colors = self.color_map_np[labels]
        segmented_points = np.concatenate((xyz, segmented_colors), axis=1)
        
        return segmented_points, labels.numpy() if isinstance(labels, torch.Tensor) else labels

    def _segment_point_cloud_with_voting_nano(self, xyz, rgb, num_votes=2):
        num_points = len(xyz)
        vote_label_pool = torch.zeros((num_points, self.num_classes), dtype=torch.float32)
        
        for vote in range(num_votes):
            xyz_vote = xyz + (np.random.rand(*xyz.shape) * 0.02 - 0.01)
            blocks = self._split_point_cloud_into_blocks_nano(xyz_vote, rgb)
            
            for i in range(0, len(blocks), self.batch_size):
                batch_blocks = blocks[i:i + self.batch_size]
                batch_points = torch.stack([b[0] for b in batch_blocks]).transpose(2, 1)
                batch_indices = [b[1] for b in batch_blocks]
                
                batch_points = batch_points.to(self.device)
                
                with torch.no_grad():
                    seg_pred, _ = self.classifier(batch_points)
                    batch_pred_label = seg_pred.argmax(dim=2)
                
                batch_pred_label = batch_pred_label.cpu()
                for b in range(len(batch_indices)):
                    one_hot = F.one_hot(batch_pred_label[b], num_classes=self.num_classes).float()
                    vote_label_pool[batch_indices[b]] += one_hot
                
                del batch_points, seg_pred
                torch.cuda.empty_cache()
        
        final_pred_labels = vote_label_pool.argmax(dim=1)
        return final_pred_labels

    def _split_point_cloud_into_blocks_nano(self, xyz, rgb):
        xyz_cpu = xyz.cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
        rgb_cpu = rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
        
        coord_min = np.min(xyz_cpu, axis=0)
        coord_max = np.max(xyz_cpu, axis=0)
        
        grid_x = int(np.ceil((coord_max[0] - coord_min[0] - self.block_size) / self.stride)) + 1
        grid_y = int(np.ceil((coord_max[1] - coord_min[1] - self.block_size) / self.stride)) + 1
        
        blocks = []
        for x in range(grid_x):
            for y in range(grid_y):
                s_x = coord_min[0] + x * self.stride
                s_y = coord_min[1] + y * self.stride
                e_x = s_x + self.block_size
                e_y = s_y + self.block_size
                
                block_mask = (xyz_cpu[:, 0] >= s_x) & (xyz_cpu[:, 0] <= e_x) & \
                            (xyz_cpu[:, 1] >= s_y) & (xyz_cpu[:, 1] <= e_y)
                
                if not np.any(block_mask):
                    continue
                
                block_xyz = xyz_cpu[block_mask]
                block_rgb = rgb_cpu[block_mask]
                
                block_center = np.array([s_x + self.block_size/2, s_y + self.block_size/2, 0])
                block_xyz_centered = block_xyz - block_center
                block_xyz_normalized = block_xyz / coord_max
                block_rgb_normalized = block_rgb / 255.0
                
                block_points = np.concatenate((block_xyz_centered, block_rgb_normalized, block_xyz_normalized), axis=1)
                
                if len(block_points) >= self.num_point:
                    idx = np.random.choice(len(block_points), self.num_point, replace=False)
                else:
                    idx = np.random.choice(len(block_points), self.num_point, replace=True)
                
                block_points = block_points[idx]
                block_points = torch.from_numpy(block_points).float()
                
                blocks.append((block_points, torch.from_numpy(block_mask[idx]).long()))
                
                del block_xyz, block_rgb, block_points
                
        return blocks
