from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models import ImprovedFashionMNISTCNN
from src.data_utils import CLASS_NAMES, get_fashion_mnist_datasets
from src.train_utils import predict_image


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def collect_predictions(model, dataset, device):
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for image, label in dataset:
            image_batch = image.unsqueeze(dim=0).to(device)

            logits = model(image_batch)
            pred = torch.argmax(logits, dim=1)

            all_true_labels.append(label)
            all_pred_labels.append(pred.item())

    return all_true_labels, all_pred_labels


def calculate_per_class_accuracy(true_labels, pred_labels, num_classes):
    correct_per_class = [0 for _ in range(num_classes)]
    total_per_class = [0 for _ in range(num_classes)]

    for true_label, pred_label in zip(true_labels, pred_labels):
        total_per_class[true_label] += 1

        if true_label == pred_label:
            correct_per_class[true_label] += 1

    per_class_accuracy = []

    for class_id in range(num_classes):
        if total_per_class[class_id] == 0:
            accuracy = 0.0
        else:
            accuracy = correct_per_class[class_id] / total_per_class[class_id]

        per_class_accuracy.append(accuracy)

    return correct_per_class, total_per_class, per_class_accuracy


def save_confusion_matrix(true_labels, pred_labels, output_path):
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm)
    plt.title("FashionMNIST Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_positions = list(range(len(CLASS_NAMES)))
    plt.xticks(tick_positions, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_positions, CLASS_NAMES)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_misclassified_examples(model, dataset, device, output_path, max_examples=12):
    model.eval()

    examples = []

    for index in range(len(dataset)):
        image, label = dataset[index]

        logits, probabilities, predicted_class = predict_image(
            model=model,
            image=image,
            device=device,
        )

        pred_id = predicted_class.item()

        if pred_id != label:
            confidence = probabilities[0, pred_id].item()

            examples.append(
                {
                    "image": image,
                    "true_label": label,
                    "pred_label": pred_id,
                    "confidence": confidence,
                }
            )

        if len(examples) >= max_examples:
            break

    if len(examples) == 0:
        print("No misclassified examples found.")
        return

    rows = 3
    cols = 4

    plt.figure(figsize=(12, 8))

    for i, example in enumerate(examples):
        image = example["image"]
        true_label = example["true_label"]
        pred_label = example["pred_label"]
        confidence = example["confidence"]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image.squeeze(0), cmap="gray")
        plt.title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[pred_label]}\n"
            f"Conf: {confidence:.2f}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {len(examples)} misclassified examples to: {output_path}")


def main():
    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Load normalized full test dataset")

    _, _, _, test_dataset_full = get_fashion_mnist_datasets(
        train_subset_size=None,
        test_subset_size=None,
        use_normalization=True,
    )

    print("Full test dataset size:", len(test_dataset_full))

    section("3. Load best improved CNN checkpoint")

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
    print(f"Best test accuracy in checkpoint: {checkpoint['best_test_acc']:.4f}")

    section("4. Collect predictions")

    true_labels, pred_labels = collect_predictions(
        model=model,
        dataset=test_dataset_full,
        device=device,
    )

    total = len(true_labels)
    correct = sum(
        true_label == pred_label
        for true_label, pred_label in zip(true_labels, pred_labels)
    )

    overall_accuracy = correct / total

    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    section("5. Per-class accuracy")

    correct_per_class, total_per_class, per_class_accuracy = (
        calculate_per_class_accuracy(
            true_labels=true_labels,
            pred_labels=pred_labels,
            num_classes=len(CLASS_NAMES),
        )
    )

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_id:02d} | "
            f"{class_name:12s} | "
            f"correct={correct_per_class[class_id]:4d} / "
            f"total={total_per_class[class_id]:4d} | "
            f"accuracy={per_class_accuracy[class_id]:.4f}"
        )

    section("6. Classification report")

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=CLASS_NAMES,
    )

    print(report)

    section("7. Save confusion matrix and error examples")

    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    confusion_matrix_path = figure_dir / "lesson_10_improved_confusion_matrix.png"
    misclassified_path = figure_dir / "lesson_10_improved_misclassified_examples.png"

    save_confusion_matrix(
        true_labels=true_labels,
        pred_labels=pred_labels,
        output_path=confusion_matrix_path,
    )

    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    save_misclassified_examples(
        model=model,
        dataset=test_dataset_full,
        device=device,
        output_path=misclassified_path,
        max_examples=12,
    )


if __name__ == "__main__":
    main()