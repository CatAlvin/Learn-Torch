from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models import ImprovedFashionMNISTCNN
from src.data_utils import CLASS_NAMES, get_fashion_mnist_datasets
from src.train_utils import predict_image


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Load normalized test dataset")

    _, _, _, test_dataset_full = get_fashion_mnist_datasets(
        train_subset_size=None,
        test_subset_size=None,
        use_normalization=True,
    )

    print("Full test dataset size:", len(test_dataset_full))

    section("3. Create improved model and load best checkpoint")

    model = ImprovedFashionMNISTCNN().to(device)

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "lesson_10_improved_fashion_mnist_cnn_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            "Please run 03_train_improved_cnn.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best test accuracy: {checkpoint['best_test_acc']:.4f}")

    section("4. Predict test images")

    for index in range(10):
        image, label = test_dataset_full[index]

        logits, probabilities, predicted_class = predict_image(
            model=model,
            image=image,
            device=device,
        )

        pred_id = predicted_class.item()
        confidence = probabilities[0, pred_id].item()

        print(
            f"image={index:02d} | "
            f"true={CLASS_NAMES[label]:12s} | "
            f"pred={CLASS_NAMES[pred_id]:12s} | "
            f"confidence={confidence:.4f}"
        )


if __name__ == "__main__":
    main()