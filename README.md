# UMI to LeRobot Dataset Converter

Convert datasets from the [UMI (Universal Manipulation Interface)](https://github.com/real-stanford/universal_manipulation_interface) project's Zarr format to [LeRobot Dataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) format.

## Features

- ✅ Converts UMI Zarr datasets to LeRobot Dataset v3 format
- ✅ Preserves all robot state and camera observations
- ✅ Handles episode boundaries automatically
- ✅ Supports video compression for efficient storage
- ✅ Compatible with LeRobot visualization and training tools
- ✅ Optional HuggingFace Hub upload

## Installation

This project uses `uv` for package management:

```bash
# Install dependencies
uv sync

# Or install manually with pip
pip install -e .
```

## Usage

### Basic Conversion

```bash
python umi_to_lerobot.py \
  --zarr-path data/dataset.zarr \
  --output-dir lerobot_data \
  --repo-id username/umi-dataset
```

### With Custom Options

```bash
python umi_to_lerobot.py \
  --zarr-path data/dataset.zarr \
  --output-dir lerobot_data \
  --repo-id username/umi-dataset \
  --fps 30 \
  --task "Pick and place red cube"
```

### Push to HuggingFace Hub

```bash
python umi_to_lerobot.py \
  --zarr-path data/dataset.zarr \
  --output-dir lerobot_data \
  --repo-id username/umi-dataset \
  --push-to-hub
```

### Command-line Arguments

- `--zarr-path`: Path to UMI `.zarr` dataset directory (required)
- `--output-dir`: Output directory for LeRobot dataset (default: `lerobot_data`)
- `--repo-id`: Repository ID for the dataset, e.g., `username/dataset-name` (required)
- `--fps`: Frame rate (default: 60, matching UMI's recording rate)
- `--task`: Task description (default: `UMI manipulation task`)
- `--push-to-hub`: Push dataset to HuggingFace Hub after conversion

## Visualizing Converted Datasets

After conversion, you can visualize the dataset using LeRobot's visualization script:

```bash
# Clone LeRobot if you haven't already
git clone https://github.com/huggingface/lerobot.git

# Visualize the converted dataset
python lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
  --repo-id username/umi-dataset \
  --root lerobot_data \
  --episode-index 0
```

Or use the LeRobot CLI (if installed):

```bash
lerobot-dataset-viz \
  --repo-id username/umi-dataset \
  --root lerobot_data \
  --episode-index 0
```

## UMI Dataset Format

The UMI dataset uses Zarr format with the following structure:

```
dataset.zarr/
├── data/
│   ├── camera0_rgb/          # RGB camera observations (H×W×3, uint8)
│   ├── robot0_eef_pos/       # End-effector position
│   ├── robot0_eef_rot_axis_angle/  # End-effector rotation
│   ├── robot0_gripper_width/ # Gripper width
│   ├── robot0_demo_start_pose/  # Demo start pose
│   └── robot0_demo_end_pose/    # Demo end pose
└── meta/
    └── episode_ends/         # Episode boundary indices
```

Data is stored at 60Hz frame rate with all frames concatenated along the time dimension.

## LeRobot Dataset v3 Format

The converter produces a LeRobot Dataset v3 with the following structure:

```
lerobot_data/
├── meta/
│   ├── info.json            # Dataset schema and metadata
│   ├── stats.json           # Feature statistics
│   ├── tasks.jsonl          # Task descriptions
│   └── episodes/            # Episode metadata (Parquet)
├── data/                    # Frame-by-frame tabular data (Parquet)
└── videos/                  # Compressed camera observations (MP4)
```

### Feature Mapping

UMI fields are mapped to LeRobot features as follows:

| UMI Field | LeRobot Feature | Type |
|-----------|----------------|------|
| `camera0_rgb` | `observation.images.camera0` | video (C×H×W) |
| `robot0_eef_pos` | `observation.state.eef_pos` | float32 |
| `robot0_eef_rot_axis_angle` | `observation.state.eef_rot` | float32 |
| `robot0_gripper_width` | `observation.state.gripper` | float32 |
| `robot0_demo_start_pose` | `observation.state.demo_start_pose` | float32 |
| `robot0_demo_end_pose` | `observation.state.demo_end_pose` | float32 |
| (derived) | `action` | float32 (eef_pos + eef_rot + gripper) |

## Requirements

- Python ≥ 3.12
- zarr ≥ 2.18.0
- numpy ≥ 1.26.0
- lerobot (from git)
- tqdm ≥ 4.66.0
- pillow ≥ 10.0.0

## Troubleshooting

### Missing zarr module

If you get `ModuleNotFoundError: No module named 'zarr'`, install dependencies:

```bash
uv sync
# or
pip install zarr numpy
```

### LeRobot not found

Install LeRobot from the main branch:

```bash
pip install git+https://github.com/huggingface/lerobot.git
```

### Video encoding errors

Make sure you have ffmpeg installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## References

- [UMI Dataset Format](https://umi-data.github.io/)
- [LeRobot Dataset v3.0 Documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot)
- [UMI GitHub Repository](https://github.com/real-stanford/universal_manipulation_interface)

## License

This converter tool is provided as-is for converting between open-source dataset formats. Please refer to the original UMI and LeRobot repositories for their respective licenses.
