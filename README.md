# vEMB-SLAM

A modular dense visual SLAM system using deep learning for depth estimation and ORB-SLAM concepts for tracking.

## Features
- **Depth Estimation:** Supports ZoeDepth, MiDaS, and Metric3D models
- **Visual Odometry:** ORB feature tracking + PnP pose estimation
- **Mapping:** Dense point cloud generation with optional semantic segmentation
- **Visualization:** Real-time 3D visualization with matplotlib
- **Motion Estimation:** Optional integration with Pixhawk (MAVLink) or AirSim simulator
- **Segmentation:** PointNet++ based semantic segmentation of point clouds

## Project Structure
```
vEMB-SLAM/
├── config/                  # Configuration files
│   └── default_config.yaml  # Main configuration
├── data/                    # Data files
│   └── calibration/         # Camera calibration files (.npz)
├── envs/                    # Conda environment configs
│   ├── environment.yml      # Full environment
│   └── jetson-env.yml      # Jetson Nano environment
├── src/                     # Source code
│   ├── depth_estimation.py  # Depth model wrappers
│   ├── pose_estimation.py   # ORB + PnP tracking
│   ├── segmentation.py      # Point cloud segmentation
│   ├── slam_system.py       # Main SLAM system
│   ├── motion_estimation.py # Pixhawk/MAVLink integration
│   ├── visualization.py     # Real-time visualization
│   ├── utils.py            # Utilities
│   ├── models/             # PointNet++ model definitions
│   │   ├── pointnet2_sem_seg.py
│   │   ├── pointnet_utils.py
│   │   └── ...
│   └── zoedepth/           # ZoeDepth library
├── scripts/                 # Executable scripts
│   ├── run_slam.py         # Main entry point
│   ├── airsim_slam.py      # AirSim simulator version
│   └── visualize_trajectory.py  # Trajectory visualization
├── weights/                 # Trained model weights
│   └── segmentation/        # PointNet++ weights
│       ├── pointnet2_sem_seg.pth
│       └── ...
├── logs/                    # Runtime logs (auto-generated)
├── requirements.txt         # Python dependencies
├── run_scripts.sh          # Parameter sweep script
└── README.md
```

## Installation

### 1. Create Conda Environment
```bash
conda env create -f envs/environment.yml
conda activate slam
```

For Jetson Nano:
```bash
conda env create -f envs/jetson-env.yml
```

### 2. Install Additional Dependencies
```bash
pip install -r requirements.txt
```

**Note:** You may need to install PyTorch and Open3D separately based on your CUDA version.

## Usage

### Basic Usage
Run SLAM with default configuration:
```bash
python scripts/run_slam.py --config config/default_config.yaml
```

### Command Line Options
Override configuration parameters:

**Change Depth Model:**
```bash
python scripts/run_slam.py --model midas
# Options: zoe, midas, metric3d
```

**Enable Real-time Visualization:**
```bash
python scripts/run_slam.py --real_time_vis
```

**Disable Visualization:**
```bash
python scripts/run_slam.py --no_vis
```

**Enable Semantic Segmentation:**
```bash
python scripts/run_slam.py --enable_segmentation
```

**Use Pixhawk Motion Estimation:**
```bash
python scripts/run_slam.py --use_pixhawk
```

**Process Dataset:**
```bash
python scripts/run_slam.py --dataset_path /path/to/image/folder
```

### AirSim Integration
For simulation with AirSim:
```bash
python scripts/airsim_slam.py --calibration_file data/calibration/CalibrationMatrix_college_cpt.npz
```

### Trajectory Visualization
Visualize saved trajectory:
```bash
python scripts/visualize_trajectory.py logs/log_YYYYMMDD_HHMMSS/camera_trajectory/trajectory.txt
```

### Parameter Sweep
Run experiments with different models:
```bash
bash run_scripts.sh
```

## Configuration

Edit `config/default_config.yaml` to customize:

### Camera Settings
- `camera_id`: Camera device ID (default: 0)
- `calibration_file`: Path to calibration file

### Depth Model Settings
- `model`: Depth estimation model (zoe/midas/metric3d)
- `min_depth`, `max_depth`: Depth range in meters
- `min_brightness`: Brightness threshold
- `voxel_size`: Point cloud downsampling size

### SLAM Parameters
- `frame_interval`: Process every Nth frame
- `orb_features`: Number of ORB features
- `feature_match_ratio`: Feature matching threshold
- `essential_matrix_threshold`: RANSAC threshold

### Segmentation Settings
- `enable_segmentation`: Enable/disable segmentation
- `nano_segmentation`: Use Jetson-optimized mode
- `segmentation_model_path`: Path to PointNet++ weights
- `segmentation_model_name`: Model architecture to use

## Camera Calibration

Calibration files are stored in `data/calibration/`. Each `.npz` file should contain:
- `Camera_matrix`: 3x3 intrinsic matrix
- `distCoeff`: Distortion coefficients

## Model Weights

Pre-trained PointNet++ weights are in `weights/segmentation/`:
- `pointnet2_sem_seg.pth` - Semantic segmentation (recommended)
- `pointnet2_cls_*.pth` - Classification variants
- `pointnet2_part_seg.pth` - Part segmentation

## Output

SLAM outputs are saved in `logs/log_YYYYMMDD_HHMMSS/`:
- `point_cloud_file/global_map.pcd` - Final 3D map
- `camera_trajectory/trajectory.txt` - Camera poses (X,Y,Z)
- `log/slam_log.txt` - Processing log

## Troubleshooting

**Import Errors:** Make sure you're in the conda environment:
```bash
conda activate slam
```

**CUDA Out of Memory:** Reduce `voxel_size` or use `nano_segmentation: true`

**No Camera Found:** Check `camera_id` in config

**Depth Model Download:** First run will download models (~2GB each) via torch.hub

## Citation

If you use this code, please cite the relevant papers:
- ZoeDepth: [arXiv:2302.12288](https://arxiv.org/abs/2302.12288)
- Metric3D: [arXiv:2307.10984](https://arxiv.org/abs/2307.10984)
- PointNet++: [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)

## License

See individual model licenses in their respective directories.
