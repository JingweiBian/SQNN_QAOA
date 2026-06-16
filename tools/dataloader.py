# -*- coding: utf-8 -*-

"""MNIST data loading helpers for the original SQNN classification task."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _downsample_shape(num_classes):
    if num_classes in (2, 3):
        return (4, 4)
    if num_classes in (4, 5):
        return (8, 8)
    print(
        f"[data] class count {num_classes} is outside 2-5; "
        "falling back to 4x4 downsampling"
    )
    return (4, 4)


def _preprocess_transform(downsample_size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.interpolate(
                    x.unsqueeze(0),
                    size=downsample_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            ),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )


class TensorLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def filter_dataset(dataset, digits):
    """Filter a torchvision dataset to the selected digits and remap labels."""

    digit_to_index = {digit: index for index, digit in enumerate(sorted(digits))}
    digit_set = set(digits)
    filtered_data = []
    filtered_labels = []

    for index in range(len(dataset)):
        data, label = dataset[index]
        if label in digit_set:
            filtered_data.append(data)
            filtered_labels.append(digit_to_index[label])

    if filtered_data:
        data_tensor = torch.stack(filtered_data)
    else:
        sample_shape = dataset[0][0].shape if len(dataset) else (0,)
        data_tensor = torch.empty((0, *sample_shape))
    return TensorLabelDataset(data_tensor, filtered_labels)


def create_mnist_loaders(config, download=True):
    """Create train/test DataLoader objects for a selected MNIST digit subset."""

    data_dir = config.get("data_dir", "./data")
    batch_size = int(config.get("batch_size", 32))
    digits = list(config.get("digits", [0, 1]))
    downsample_size = _downsample_shape(len(digits))
    input_dim = downsample_size[0] * downsample_size[1]

    print("[data] dataset=MNIST")
    print(f"[data] digits={digits}")
    print(f"[data] downsample={downsample_size}, input_dim={input_dim}")

    preprocess = _preprocess_transform(downsample_size)
    train_set = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=preprocess,
    )
    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=preprocess,
    )

    train_set = filter_dataset(train_set, digits)
    test_set = filter_dataset(test_set, digits)

    debug_mode = config.get("debug_mode", False)
    if debug_mode:
        debug_samples = int(config.get("debug_samples", 8))
        print(f"[data] debug mode: using {debug_samples} train samples")
        train_set = Subset(train_set, list(range(min(debug_samples, len(train_set)))))
        test_count = min(max(1, debug_samples // 3), len(test_set))
        test_set = Subset(test_set, list(range(test_count)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"[data] train_samples={len(train_set)}, test_samples={len(test_set)}")
    print(f"[data] batch_size={batch_size}")
    return train_loader, test_loader
