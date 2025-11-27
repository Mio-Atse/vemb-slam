import argparse
import yaml
import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.slam_system import SLAMSystem

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="RGB SLAM System")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    
    # Allow overriding key config parameters via CLI
    parser.add_argument("--video_path", type=str, help="Path to video file (overrides camera_id)")
    parser.add_argument("--model", type=str, choices=['zoe', 'midas', 'metric3d'], help="Depth model")
    parser.add_argument("--use_pixhawk", action="store_true", help="Enable Pixhawk motion estimation")
    parser.add_argument("--enable_segmentation", action="store_true", help="Enable semantic segmentation")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset (enables dataset mode)")
    parser.add_argument("--real_time_vis", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--no_vis", action="store_false", dest="real_time_vis", help="Disable real-time visualization")
    parser.set_defaults(real_time_vis=None) # Default to config if not specified

    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.video_path:
        config['video_path'] = args.video_path
        config['camera_id'] = None
    if args.model:
        config['model'] = args.model
    if args.use_pixhawk:
        config['use_pixhawk'] = True
    if args.enable_segmentation:
        config['enable_segmentation'] = True
    if args.dataset_path:
        config['use_dataset'] = True
        config['dataset_path'] = args.dataset_path
    if args.real_time_vis is not None:
        config['real_time_vis'] = args.real_time_vis

    print("Loaded Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    slam = SLAMSystem(config)
    slam.run()

if __name__ == "__main__":
    main()
