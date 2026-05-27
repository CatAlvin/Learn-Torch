from pathlib import Path
import sys
import csv

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


def save_metrics_csv(metrics, output_path):
    fieldnames = [
        "experiment_name",
        "epoch",
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
        "learning_rate",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in metrics:
            writer.writerow(row)

    print(f"Metrics CSV saved to: {output_path}")


def save_comparison_plot(all_metrics, output_path):
    plt.figure()

    for experiment_name, metrics in all_metrics.items():
        epochs = [row["epoch"] for row in metrics]
        test_acc = [row["test_acc"] for row in metrics]

        plt.plot(
            epochs,
            test_acc,
            label=experiment_name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Experiment Comparison - Test Accuracy")
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def run_experiment(
    experiment_name,
    use_augmentation,
    device,
    max_epochs=10,
):
    section(f"Run experiment: {experiment_name}")

    train_loader, test_loader, _, _ = get_fashion_mnist_loaders(
        batch_size=128,
        train_subset_size=10000,
        test_subset_size=2000,
        num_workers=0,
        use_normalization=True,
        use_augmentation=use_augmentation,
    )

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

    metrics = []

    best_test_acc = 0.0

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

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        row = {
            "experiment_name": experiment_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "learning_rate": current_lr,
        }

        metrics.append(row)

        print(
            f"{experiment_name} | "
            f"epoch={epoch:02d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.6f} | "
            f"test_loss={test_loss:.6f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"best_test_acc={best_test_acc:.4f}"
        )

    return metrics, best_test_acc


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Prepare output directories")

    logs_dir = PROJECT_ROOT / "outputs" / "logs"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"

    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    section("3. Run experiments")

    experiment_configs = [
        {
            "experiment_name": "improved_cnn_no_augmentation",
            "use_augmentation": False,
        },
        {
            "experiment_name": "improved_cnn_with_augmentation",
            "use_augmentation": True,
        },
    ]

    all_metrics = {}
    summary = []

    for config in experiment_configs:
        experiment_name = config["experiment_name"]

        metrics, best_test_acc = run_experiment(
            experiment_name=experiment_name,
            use_augmentation=config["use_augmentation"],
            device=device,
            max_epochs=10,
        )

        all_metrics[experiment_name] = metrics

        summary.append(
            {
                "experiment_name": experiment_name,
                "best_test_acc": best_test_acc,
                "final_test_acc": metrics[-1]["test_acc"],
                "final_test_loss": metrics[-1]["test_loss"],
            }
        )

        csv_path = logs_dir / f"{experiment_name}.csv"

        save_metrics_csv(
            metrics=metrics,
            output_path=csv_path,
        )

    section("4. Save summary CSV")

    summary_path = logs_dir / "experiment_summary.csv"

    with open(summary_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "experiment_name",
            "best_test_acc",
            "final_test_acc",
            "final_test_loss",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary:
            writer.writerow(row)

    print(f"Experiment summary saved to: {summary_path}")

    print("\nExperiment summary:")
    for row in summary:
        print(
            f"{row['experiment_name']} | "
            f"best_test_acc={row['best_test_acc']:.4f} | "
            f"final_test_acc={row['final_test_acc']:.4f} | "
            f"final_test_loss={row['final_test_loss']:.6f}"
        )

    section("5. Save comparison plot")

    comparison_path = figures_dir / "lesson_10_experiment_comparison.png"

    save_comparison_plot(
        all_metrics=all_metrics,
        output_path=comparison_path,
    )


if __name__ == "__main__":
    main()