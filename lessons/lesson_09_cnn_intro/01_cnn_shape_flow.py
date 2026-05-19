import torch


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_tensor(name, tensor):
    print(f"{name}:")
    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)
    print("device:", tensor.device)


class SimpleCNN(torch.nn.Module):
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
            torch.nn.Linear(in_features=16 * 7 * 7, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Create fake image batch")

    # Fake image batch:
    # 8 grayscale images, each image is 28 x 28.
    images = torch.rand(8, 1, 28, 28, device=device)

    show_tensor("images", images)

    print("\nMeaning of shape [8, 1, 28, 28]:")
    print("8  = batch size")
    print("1  = grayscale channel")
    print("28 = image height")
    print("28 = image width")

    section("3. First Conv2d layer")

    conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=8,
        kernel_size=3,
        padding=1,
    ).to(device)

    x = conv1(images)

    show_tensor("after conv1", x)

    print("\nMeaning:")
    print("Input shape : [8, 1, 28, 28]")
    print("Output shape: [8, 8, 28, 28]")
    print("The number of channels changes from 1 to 8.")
    print("Height and width stay 28 because padding=1.")

    section("4. ReLU activation")

    relu = torch.nn.ReLU()

    x = relu(x)

    show_tensor("after ReLU", x)

    print("\nReLU does not change shape.")
    print("It only changes values: negative values become 0.")

    section("5. MaxPool2d layer")

    pool = torch.nn.MaxPool2d(kernel_size=2)

    x = pool(x)

    show_tensor("after max pooling", x)

    print("\nMeaning:")
    print("Input shape : [8, 8, 28, 28]")
    print("Output shape: [8, 8, 14, 14]")
    print("MaxPool2d(kernel_size=2) halves height and width.")

    section("6. Second Conv2d + ReLU + MaxPool2d")

    conv2 = torch.nn.Conv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
    ).to(device)

    x = conv2(x)
    show_tensor("after conv2", x)

    x = relu(x)
    show_tensor("after second ReLU", x)

    x = pool(x)
    show_tensor("after second max pooling", x)

    print("\nShape flow:")
    print("[8, 1, 28, 28]")
    print("-> Conv2d(1, 8)      -> [8, 8, 28, 28]")
    print("-> MaxPool2d(2)      -> [8, 8, 14, 14]")
    print("-> Conv2d(8, 16)     -> [8, 16, 14, 14]")
    print("-> MaxPool2d(2)      -> [8, 16, 7, 7]")

    section("7. Flatten before Linear layer")

    flatten = torch.nn.Flatten()

    flat_x = flatten(x)

    show_tensor("flat_x", flat_x)

    print("\nMeaning:")
    print("Before flatten: [8, 16, 7, 7]")
    print("After flatten : [8, 784]")
    print("Because 16 * 7 * 7 = 784.")

    section("8. Linear classifier")

    linear = torch.nn.Linear(in_features=16 * 7 * 7, out_features=10).to(device)

    logits = linear(flat_x)

    show_tensor("logits", logits)

    print("\nMeaning:")
    print("Each image gets 10 class scores.")
    print("logits shape is [batch_size, num_classes].")
    print("Here: [8, 10].")

    section("9. Full SimpleCNN model")

    model = SimpleCNN().to(device)

    print(model)

    logits = model(images)

    show_tensor("model output logits", logits)

    section("10. Fake labels and CrossEntropyLoss")

    # Suppose we have 10 classes: 0 to 9.
    labels = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.long,
        device=device,
    )

    show_tensor("labels", labels)

    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(logits, labels)

    print("loss:", loss.item())

    section("11. Predictions")

    probabilities = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(logits, dim=1)

    show_tensor("probabilities", probabilities)
    show_tensor("predicted_classes", predicted_classes)

    print("\nImportant:")
    print("For training: use raw logits with CrossEntropyLoss.")
    print("For display : use softmax to see probabilities.")


if __name__ == "__main__":
    main()