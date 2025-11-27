# Models Directory

This directory contains the **model definitions** (architecture code) for the segmentation models used in the RGB-SLAM system.

## Contents

- `pointnet2_sem_seg.py` - PointNet++ for semantic segmentation
- `pointnet2_sem_seg_msg.py` - Multi-scale grouped PointNet++ for semantic segmentation
- `pointnet_sem_seg.py` - Original PointNet for semantic segmentation
- `pointnet2_cls_msg.py` - PointNet++ for classification (multi-scale)
- `pointnet2_cls_ssg.py` - PointNet++ for classification (single-scale)
- `pointnet2_part_seg_msg.py` - PointNet++ for part segmentation (multi-scale)
- `pointnet2_part_seg_ssg.py` - PointNet++ for part segmentation (single-scale)
- `pointnet_cls.py` - Original PointNet for classification
- `pointnet_part_seg.py` - Original PointNet for part segmentation
- `pointnet2_utils.py` - Utility functions for PointNet++
- `pointnet_utils.py` - Utility functions for PointNet

## Model Weights

The trained weights for these models are stored in `weights/segmentation/`.

## Usage

These model definitions are imported by `src/segmentation.py` when loading the segmentation classifier. The config file (`config/default_config.yaml`) specifies which model to use via the `segmentation_model_name` parameter.

Example:
```yaml
segmentation_model_name: "src.models.pointnet2_sem_seg"
segmentation_model_path: "weights/segmentation/pointnet2_sem_seg.pth"
```
