import os.path
import re
from collections import defaultdict
import torch

def extract_case_and_frame(filename):
    match = re.match(r'frame_(\d+)(\.\d+)?\_(\d+).png', filename)
    if match:
        case_num = int(match.group(1))
        frame_num = int(match.group(3))
        return case_num, frame_num
    else:
        print(f'Cannot match {filename}')
        return None, None

def get_min_frame_dataset(original_dataset, subset):
    # Collect frame number for each case from the original dataset
    case_to_frames = defaultdict(list)
    for idx in subset.indices:  # Use the indices of the Subset
        path, _ = original_dataset.samples[idx]  # Access the original dataset's samples
        filename = os.path.basename(path)
        case_num, frame_num = extract_case_and_frame(filename)
        if case_num is not None:
            case_to_frames[case_num].append((frame_num, idx))

    # Find the index of the image with the lowest frame number for each case
    min_frame_indices = []
    for case_num, frames in case_to_frames.items():
        min_frame = min(frames, key=lambda x: x[0])
        min_frame_indices.append(min_frame[1])

    # Create new dataset as a subset of the original dataset
    min_frame_dataset = torch.utils.data.Subset(original_dataset, min_frame_indices)
    return min_frame_dataset

# Usage:
# Assuming `original_dataset` is the dataset that was passed to create the subsets
# and `test_subset` is a Subset object created from the original_dataset