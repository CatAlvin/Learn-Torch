from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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
    return correct / total


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


def predict_one_image(model, image, device):
    model.eval()

    # Original image shape: [1, 28, 28]
    # Model expects batch shape: [N, C, H, W]
    # So we add a batch dimension:
    # [1, 28, 28] -> [1, 1, 28, 28]
    image_batch = image.unsqueeze(dim=0).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    return logits, probabilities, predicted_class


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Load FashionMNIST")

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

    train_dataset = Subset(train_dataset_full, list(range(5000)))
    test_dataset = Subset(test_dataset_full, list(range(1000)))

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

    print("Train subset size:", len(train_dataset))
    print("Test subset size :", len(test_dataset))

    section("3. Create and train model")

    model = FashionMNISTCNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    num_epochs = 3

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

        print(
            f"epoch={epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"test_loss={test_loss:.6f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    section("4. Save model state_dict")

    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "fashion_mnist_cnn_state_dict.pth"

    torch.save(model.state_dict(), model_path)

    print(f"Model state_dict saved to: {model_path}")

    section("5. Load model into a new model object")

    loaded_model = FashionMNISTCNN().to(device)

    loaded_state_dict = torch.load(
        model_path,
        map_location=device,
    )

    loaded_model.load_state_dict(loaded_state_dict)

    loaded_model.eval()

    print("Loaded model successfully.")

    section("6. Compare original model and loaded model")

    image, label = test_dataset_full[0]

    show_tensor("one test image", image)
    print("true label:", label)
    print("true class:", CLASS_NAMES[label])

    original_logits, original_probs, original_pred = predict_one_image(
        model=model,
        image=image,
        device=device,
    )

    loaded_logits, loaded_probs, loaded_pred = predict_one_image(
        model=loaded_model,
        image=image,
        device=device,
    )

    print("\nOriginal model prediction:")
    print("pred class id:", original_pred.item())
    print("pred class   :", CLASS_NAMES[original_pred.item()])
    print("confidence   :", original_probs[0, original_pred.item()].item())

    print("\nLoaded model prediction:")
    print("pred class id:", loaded_pred.item())
    print("pred class   :", CLASS_NAMES[loaded_pred.item()])
    print("confidence   :", loaded_probs[0, loaded_pred.item()].item())

    print("\nAre logits close?")
    print(torch.allclose(original_logits, loaded_logits))

    section("7. Predict several images with loaded model")

    for index in range(10):
        image, label = test_dataset_full[index]

        logits, probabilities, predicted_class = predict_one_image(
            model=loaded_model,
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