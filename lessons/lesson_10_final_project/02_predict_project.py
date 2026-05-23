from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models import FashionMNISTCNN
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

    section("2. Load test dataset")

    _, _, _, test_dataset_full = get_fashion_mnist_datasets(
        train_subset_size=5000,
        test_subset_size=1000,
    )

    print("Full test dataset size:", len(test_dataset_full))

    section("3. Create model and load weights")

    model = FashionMNISTCNN().to(device)

    model_path = PROJECT_ROOT / "outputs" / "models" / "lesson_10_fashion_mnist_cnn.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please run 01_train_project.py first."
        )

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.eval()

    print(f"Loaded model from: {model_path}")

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