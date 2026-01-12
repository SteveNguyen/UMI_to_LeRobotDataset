#!/bin/bash
# Example script to visualize the converted LeRobot dataset
# Usage: ./visualize_dataset.sh [episode_index]

EPISODE_INDEX="${1:-0}"

echo "Visualizing episode ${EPISODE_INDEX} from converted dataset..."
uv run python vendors/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
  --repo-id test/umi-converted \
  --root lerobot_output \
  --episode-index ${EPISODE_INDEX} \
  --display-compressed-images true
