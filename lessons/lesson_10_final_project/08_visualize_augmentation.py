from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_utils import (
    CLASS_NAMES,
    build_fashion_mnist_transform,
)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def unnormalize_image(image):
    mean = 0.2860
    std = 0.3530

    image = image * std + mean
    image = image.clamp(0, 1)

    return image


def save_augmentation_grid(original_dataset, augmented_dataset, output_path):
    sample_indices = [0, 1, 2, 3, 4, 5]

    rows = len(sample_indices)
    cols = 5

    plt.figure(figsize=(12, 14))

    for row, index in enumerate(sample_indices):
        original_image, label = original_dataset[index]

        plt.subplot(rows, cols, row * cols + 1)
        plt.imshow(original_image.squeeze(0), cmap="gray")
        plt.title(f"Original\n{CLASS_NAMES[label]}")
        plt.axis("off")

        for aug_id in range(1, cols):
            augmented_image, _ = augmented_dataset[index]

            augmented_image = unnormalize_image(augmented_image)

            plt.subplot(rows, cols, row * cols + aug_id + 1)
            plt.imshow(augmented_image.squeeze(0), cmap="gray")
            plt.title(f"Aug {aug_id}")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    torch.manual_seed(42)

    section("1. Build transforms")

    original_transform = build_fashion_mnist_transform(
        use_normalization=False,
        use_augmentation=False,
    )

    augmented_transform = build_fashion_mnist_transform(
        use_normalization=True,
        use_augmentation=True,
    )

    print("Original transform:")
    print(original_transform)

    print("\nAugmented transform:")
    print(augmented_transform)

    section("2. Load FashionMNIST with different transforms")

    original_dataset = datasets.FashionMNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=original_transform,
    )

    augmented_dataset = datasets.FashionMNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=augmented_transform,
    )

    print("Original dataset size :", len(original_dataset))
    print("Augmented dataset size:", len(augmented_dataset))

    section("3. Save augmentation visualization")

    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "lesson_10_augmentation_visualization.png"

    save_augmentation_grid(
        original_dataset=original_dataset,
        augmented_dataset=augmented_dataset,
        output_path=output_path,
    )

    print(f"Augmentation visualization saved to: {output_path}")

    section("4. What to check")

    print("Open the saved image and check:")
    print("1. Are augmented images still recognizable?")
    print("2. Are rotations too strong?")
    print("3. Are translations too large?")
    print("4. Do labels still make sense?")
    print("5. Are some classes harmed by horizontal flip?")


if __name__ == "__main__":
    main()