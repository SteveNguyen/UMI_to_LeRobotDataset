#!/usr/bin/env python3
"""
Convert UMI (Universal Manipulation Interface) datasets from Zarr format
to LeRobot Dataset v3 format.

UMI format reference: https://umi-data.github.io/
LeRobot v3 format reference: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
"""

import argparse
from pathlib import Path
import numpy as np
import zarr

# Register imagecodecs with numcodecs
try:
    import imagecodecs.numcodecs
    imagecodecs.numcodecs.register_codecs()
except ImportError:
    print("Warning: imagecodecs not available, JpegXL compressed images may not work")

from PIL import Image
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_umi_zarr(zarr_path: Path) -> tuple[zarr.Group, list[int]]:
    """
    Load UMI Zarr dataset and extract episode boundaries.

    Args:
        zarr_path: Path to the .zarr directory or zip file

    Returns:
        Tuple of (zarr_root, episode_ends)
    """
    print(f"Loading UMI dataset from {zarr_path}")

    # Open the zarr dataset
    root = zarr.open(str(zarr_path), mode='r')

    # Get episode boundaries
    episode_ends = root['meta']['episode_ends'][:]
    print(f"Found {len(episode_ends)} episodes")
    print(f"Total frames: {episode_ends[-1]}")

    return root, episode_ends.tolist()


def get_umi_data_info(root: zarr.Group) -> dict:
    """
    Extract information about available data arrays in the UMI dataset.

    Args:
        root: Zarr root group

    Returns:
        Dictionary with data array names and shapes
    """
    data_info = {}
    data_group = root['data']

    for key in data_group.keys():
        array = data_group[key]
        data_info[key] = {
            'shape': array.shape,
            'dtype': array.dtype,
            'chunks': array.chunks if hasattr(array, 'chunks') else None
        }

    print("\nAvailable data arrays:")
    for key, info in data_info.items():
        print(f"  {key}: shape={info['shape']}, dtype={info['dtype']}")

    return data_info


def create_lerobot_features(data_info: dict) -> tuple[dict, list[str]]:
    """
    Create LeRobot feature specification from UMI data info.

    Args:
        data_info: Dictionary with UMI data array information

    Returns:
        Tuple of (features dict, list of state field names in order)
    """
    features = {}
    state_fields = []
    state_dims = []

    # Map UMI data fields to LeRobot features
    for key, info in data_info.items():
        shape = info['shape']

        if 'rgb' in key.lower() or 'image' in key.lower():
            # Camera data: stored as video in LeRobot
            # UMI shape: (frames, height, width, channels)
            # LeRobot expects: (channels, height, width)
            if len(shape) == 4:
                _, h, w, c = shape
                # Use observation.images prefix for camera data
                camera_name = key.replace('_rgb', '').replace('_', '.')
                features[f"observation.images.{camera_name}"] = {
                    "dtype": "video",
                    "shape": (c, h, w),
                    "names": ["channel", "height", "width"]
                }
        else:
            # State data: we'll concatenate all robot state into observation.state
            if 'robot' in key and len(shape) >= 2:
                feature_shape = shape[1:]
                flat_size = int(np.prod(feature_shape))
                state_fields.append(key)
                state_dims.append(flat_size)

    # Create single observation.state feature with concatenated state
    if state_fields:
        total_state_dim = sum(state_dims)
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (total_state_dim,),
            "names": None  # Can be None for state vectors
        }

    # Add action feature (required by LeRobot)
    # Use eef_pos + eef_rot + gripper as action (common pattern)
    action_dims = 0
    for field in state_fields:
        if any(x in field for x in ['eef_pos', 'eef_rot', 'gripper']):
            idx = state_fields.index(field)
            action_dims += state_dims[idx]

    if action_dims > 0:
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dims,),
            "names": None
        }

    print("\nLeRobot features:")
    for key, spec in features.items():
        print(f"  {key}: dtype={spec['dtype']}, shape={spec['shape']}")

    print(f"\nState fields (in order): {state_fields}")

    return features, state_fields


def convert_umi_to_lerobot(
    zarr_path: Path,
    output_dir: Path,
    repo_id: str,
    fps: int = 60,
    task_description: str = "UMI manipulation task",
    push_to_hub: bool = False
):
    """
    Convert UMI dataset to LeRobot Dataset v3 format.

    Args:
        zarr_path: Path to UMI .zarr dataset
        output_dir: Output directory for LeRobot dataset
        repo_id: Repository ID for the dataset (e.g., "username/dataset-name")
        fps: Frame rate (UMI default is 60Hz)
        task_description: Description of the task
        push_to_hub: Whether to push to HuggingFace Hub after conversion
    """
    # Load UMI dataset
    root, episode_ends = load_umi_zarr(zarr_path)
    data_info = get_umi_data_info(root)

    # Create LeRobot feature specification
    features, state_fields = create_lerobot_features(data_info)

    # Create LeRobot dataset
    print(f"\nCreating LeRobot dataset at {output_dir}")
    print(f"Repository ID: {repo_id}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(output_dir),
        fps=fps,
        features=features,
        use_videos=True,  # Use video compression for images
    )

    # Convert episodes
    data_group = root['data']
    episode_starts = [0] + episode_ends[:-1]

    print(f"\nConverting {len(episode_ends)} episodes...")

    for ep_idx, (start_idx, end_idx) in enumerate(tqdm(
        zip(episode_starts, episode_ends),
        total=len(episode_ends),
        desc="Converting episodes"
    )):
        # Process each frame in the episode
        for frame_idx in range(start_idx, end_idx):
            frame_data = {}
            state_components = []
            action_components = []

            # Extract data for this frame
            for key in data_group.keys():
                array = data_group[key]

                if 'rgb' in key.lower() or 'image' in key.lower():
                    # Handle camera data
                    # UMI stores as (frames, H, W, C), convert to (C, H, W)
                    img_data = array[frame_idx]

                    # Convert to PIL Image and then to numpy array in correct format
                    if img_data.dtype == np.uint8:
                        # Already uint8, convert to (C, H, W)
                        img_data = np.transpose(img_data, (2, 0, 1))

                    camera_name = key.replace('_rgb', '').replace('_', '.')
                    frame_data[f"observation.images.{camera_name}"] = img_data

                elif key in state_fields:
                    # Handle state data - extract and add to state vector
                    data = array[frame_idx].flatten().astype(np.float32)
                    state_components.append(data)

                    # Also use eef_pos, eef_rot, and gripper for action
                    if any(x in key for x in ['eef_pos', 'eef_rot', 'gripper']):
                        action_components.append(data)

            # Concatenate all state components into single observation.state
            if state_components:
                frame_data["observation.state"] = np.concatenate(state_components)

            # Create action from relevant components
            if action_components:
                frame_data["action"] = np.concatenate(action_components)

            # Add task description
            frame_data["task"] = task_description

            # Add frame to dataset
            dataset.add_frame(frame_data)

        # Save episode
        dataset.save_episode()

    # Finalize dataset (required for v3)
    print("\nFinalizing dataset...")
    dataset.finalize()

    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Episodes: {len(episode_ends)}")
    print(f"  Total frames: {episode_ends[-1]}")

    # Push to Hub if requested
    if push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        dataset.push_to_hub()
        print(f"✓ Pushed to hub: https://huggingface.co/datasets/{repo_id}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert UMI datasets to LeRobot Dataset v3 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert local UMI dataset
  python umi_to_lerobot.py \\
    --zarr-path data/dataset.zarr \\
    --output-dir lerobot_data \\
    --repo-id username/umi-dataset

  # Convert and push to HuggingFace Hub
  python umi_to_lerobot.py \\
    --zarr-path data/dataset.zarr \\
    --output-dir lerobot_data \\
    --repo-id username/umi-dataset \\
    --push-to-hub

  # Specify custom FPS and task description
  python umi_to_lerobot.py \\
    --zarr-path data/dataset.zarr \\
    --output-dir lerobot_data \\
    --repo-id username/umi-dataset \\
    --fps 30 \\
    --task "Pick and place task"
        """
    )

    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to UMI .zarr dataset directory"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lerobot_data"),
        help="Output directory for LeRobot dataset (default: lerobot_data)"
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID for the dataset (e.g., 'username/dataset-name')"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frame rate (default: 60, matching UMI's recording rate)"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="UMI manipulation task",
        help="Task description (default: 'UMI manipulation task')"
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub after conversion"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.zarr_path.exists():
        parser.error(f"Zarr path does not exist: {args.zarr_path}")

    # Note: Output directory will be created by LeRobotDataset.create()
    # Do not create it here as LeRobotDataset requires it not to exist

    # Run conversion
    convert_umi_to_lerobot(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        task_description=args.task,
        push_to_hub=args.push_to_hub
    )


if __name__ == "__main__":
    main()
