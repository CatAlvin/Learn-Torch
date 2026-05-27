from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models import ImprovedFashionMNISTCNN
from src.data_utils import get_fashion_mnist_loaders
from src.train_utils import train_one_epoch, evaluate


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def save_training_curves(
    train_loss,
    test_loss,
    train_acc,
    test_acc,
    learning_rates,
    output_dir,
):
    loss_path = output_dir / "lesson_10_augmentation_loss.png"
    acc_path = output_dir / "lesson_10_augmentation_accuracy.png"
    lr_path = output_dir / "lesson_10_augmentation_lr.png"

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training with Data Augmentation - Loss")
    plt.legend()
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training with Data Augmentation - Accuracy")
    plt.legend()
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(learning_rates)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Training with Data Augmentation - Learning Rate")
    plt.savefig(lr_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {loss_path}")
    print(f"Accuracy curve saved to: {acc_path}")
    print(f"Learning rate curve saved to: {lr_path}")


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Load FashionMNIST with train-time augmentation")

    train_loader, test_loader, _, _ = get_fashion_mnist_loaders(
        batch_size=128,
        train_subset_size=10000,
        test_subset_size=2000,
        num_workers=0,
        use_normalization=True,
        use_augmentation=True,
    )

    print("Train batches:", len(train_loader))
    print("Test batches :", len(test_loader))

    section("3. Create model, loss function, optimizer, scheduler")

    model = ImprovedFashionMNISTCNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print(model)
    print("Optimizer:", optimizer)
    print("Scheduler:", scheduler)

    section("4. Train with augmentation")

    max_epochs = 30

    early_stopping_patience = 7
    epochs_without_improvement = 0

    best_test_acc = 0.0
    best_test_loss = float("inf")

    model_dir = PROJECT_ROOT / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "lesson_10_augmentation_best.pth"

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    learning_rate_history = []

    for epoch in range(max_epochs + 1):
        current_lr = get_current_lr(optimizer)

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

        scheduler.step(test_loss)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        learning_rate_history.append(current_lr)

        print(
            f"epoch={epoch:02d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.6f} | "
            f"test_loss={test_loss:.6f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

        improved = test_acc > best_test_acc

        if improved:
            best_test_acc = test_acc
            best_test_loss = test_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_test_acc": best_test_acc,
                    "best_test_loss": best_test_loss,
                    "use_normalization": True,
                    "use_augmentation": True,
                },
                best_model_path,
            )

            print(
                f"New best checkpoint saved: "
                f"best_test_acc={best_test_acc:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "\nEarly stopping triggered. "
                f"No test accuracy improvement for "
                f"{early_stopping_patience} epochs."
            )
            break

    section("5. Training summary")

    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Best test loss    : {best_test_loss:.6f}")
    print(f"Best checkpoint   : {best_model_path}")

    section("6. Save curves")

    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    save_training_curves(
        train_loss=train_loss_history,
        test_loss=test_loss_history,
        train_acc=train_acc_history,
        test_acc=test_acc_history,
        learning_rates=learning_rate_history,
        output_dir=figure_dir,
    )


if __name__ == "__main__":
    main()