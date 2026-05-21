from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_tensor(name, tensor):
    print(f"{name}:")
    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)
    print("device:", tensor.device)


class FashionMNISTCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=16 * 7 * 7, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def calculate_accuracy(logits, labels):
    predicted_classes = torch.argmax(logits, dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(logits, labels)

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, labels)

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy


def save_sample_images(dataset, output_path):
    plt.figure(figsize=(8, 4))

    for i in range(10):
        image, label = dataset[i]

        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(0), cmap="gray")
        plt.title(CLASS_NAMES[label])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves(train_loss, test_loss, train_acc, test_acc, output_dir):
    loss_path = output_dir / "lesson_09_fashion_mnist_loss.png"
    acc_path = output_dir / "lesson_09_fashion_mnist_accuracy.png"

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FashionMNIST CNN Loss")
    plt.legend()
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("FashionMNIST CNN Accuracy")
    plt.legend()
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {loss_path}")
    print(f"Accuracy curve saved to: {acc_path}")


def predict_some_images(model, dataset, device, count=10):
    model.eval()

    images = []
    labels = []

    for i in range(count):
        image, label = dataset[i]
        images.append(image)
        labels.append(label)

    image_batch = torch.stack(images, dim=0).to(device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(logits, dim=1)

    print("\nPrediction examples:")

    for i in range(count):
        true_label = label_tensor[i].item()
        pred_label = predicted_classes[i].item()
        confidence = probabilities[i, pred_label].item()

        print(
            f"image {i:02d} | "
            f"true={CLASS_NAMES[true_label]:12s} | "
            f"pred={CLASS_NAMES[pred_label]:12s} | "
            f"confidence={confidence:.4f}"
        )


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Load FashionMNIST dataset")

    transform = transforms.ToTensor()

    train_dataset_full = datasets.FashionMNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset_full = datasets.FashionMNIST(
        root="data/raw",
        train=False,
        download=True,
        transform=transform,
    )

    print("Full train dataset size:", len(train_dataset_full))
    print("Full test dataset size :", len(test_dataset_full))

    # To make this lesson faster, use a smaller subset.
    # Later you can train on the full dataset.
    train_indices = list(range(5000))
    test_indices = list(range(1000))

    train_dataset = Subset(train_dataset_full, train_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    print("Used train subset size:", len(train_dataset))
    print("Used test subset size :", len(test_dataset))

    section("3. Check one sample")

    image, label = train_dataset[0]

    show_tensor("one image", image)
    print("label:", label)
    print("class name:", CLASS_NAMES[label])

    print("\nMeaning:")
    print("FashionMNIST image shape is [1, 28, 28].")
    print("1 means grayscale channel.")
    print("28 and 28 are image height and width.")

    section("4. Create DataLoaders")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    print("Train batches:", len(train_loader))
    print("Test batches :", len(test_loader))

    batch_images, batch_labels = next(iter(train_loader))

    show_tensor("batch_images", batch_images)
    show_tensor("batch_labels", batch_labels)

    section("5. Save sample images")

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "lesson_09_fashion_mnist_samples.png"
    save_sample_images(train_dataset_full, sample_path)

    print(f"Sample images saved to: {sample_path}")

    section("6. Create CNN model")

    model = FashionMNISTCNN().to(device)

    print(model)

    section("7. Check model output shape")

    batch_images = batch_images.to(device)

    logits = model(batch_images)

    show_tensor("logits", logits)

    print("\nExpected:")
    print("batch_images shape: [64, 1, 28, 28]")
    print("logits shape      : [64, 10]")
    print("Because this is a 10-class classification task.")

    section("8. Train model")

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

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

    section("9. Predict some test images")

    predict_some_images(
        model=model,
        dataset=test_dataset_full,
        device=device,
        count=10,
    )

    section("10. Save training curves")

    save_training_curves(
        train_loss=train_loss_history,
        test_loss=test_loss_history,
        train_acc=train_acc_history,
        test_acc=test_acc_history,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()