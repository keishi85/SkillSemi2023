from collections import Counter
import torch


# Modify get_class_distribution function to handle Subset
def get_class_distribution(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        # Access labels from the original dataset using subset indices
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        # Dataset is not a Subset, use the existing method
        labels = dataset.targets
    return Counter(labels)

# Calculate class weight
def get_class_weights(dataset):
    class_counts = get_class_distribution(dataset)
    total_samples = sum(class_counts.values())

    # key : label, value : number of label
    class_weights = {class_id: total_samples / count for class_id, count in class_counts.items()}
    return class_weights