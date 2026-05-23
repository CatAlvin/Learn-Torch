from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models import FashionMNISTCNN
from src.data_utils import get_fashion_mnist_loaders
from src.train_utils import train_one_epoch, evaluate


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_training_curves(train_loss, test_loss, train_acc, test_acc, output_dir):
    loss_path = output_dir / "lesson_10_project_loss.png"
    acc_path = output_dir / "lesson_10_project_accuracy.png"

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FashionMNIST Project Loss")
    plt.legend()
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("FashionMNIST Project Accuracy")
    plt.legend()
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {loss_path}")
    print(f"Accuracy curve saved to: {acc_path}")


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Load DataLoaders")

    train_loader, test_loader, _, _ = get_fashion_mnist_loaders(
        batch_size=64,
        train_subset_size=5000,
        test_subset_size=1000,
        num_workers=0,
    )

    print("Train batches:", len(train_loader))
    print("Test batches :", len(test_loader))

    section("3. Create model")

    model = FashionMNISTCNN().to(device)

    print(model)

    section("4. Create loss function and optimizer")

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("Loss function:", loss_fn)
    print("Optimizer:", optimizer)

    section("5. Train model")

    num_epochs = 5

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(
            f"epoch={epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"test_loss={test_loss:.6f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    section("6. Save model")

    model_dir = PROJECT_ROOT / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lesson_10_fashion_mnist_cnn.pth"

    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")

    section("7. Save training curves")

    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    save_training_curves(
        train_loss=train_loss_history,
        test_loss=test_loss_history,
        train_acc=train_acc_history,
        test_acc=test_acc_history,
        output_dir=figure_dir,
    )


if __name__ == "__main__":
    main()