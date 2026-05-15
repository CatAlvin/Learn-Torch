from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_tensor(name, tensor):
    print(f"{name}:")
    print(tensor)
    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)
    print("device:", tensor.device)


class StudentPassDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y


class StudentPassClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=2),
        )

    def forward(self, x):
        logits = self.network(x)
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

    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_features)
        loss = loss_fn(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(logits, batch_labels)

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, batch_labels)

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy


def predict_one_student(model, student, feature_mean, feature_std, device):
    model.eval()

    normalized_student = (student - feature_mean) / feature_std
    normalized_student = normalized_student.to(device)

    with torch.no_grad():
        logits = model(normalized_student)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    return logits, probabilities, predicted_class


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Prepare raw data")

    # Features:
    # study_hours, attendance_rate, homework_score, quiz_score
    student_features = torch.tensor(
        [
            [2.0, 0.70, 65.0, 60.0],
            [5.0, 0.90, 88.0, 84.0],
            [1.0, 0.50, 40.0, 45.0],
            [4.0, 0.85, 78.0, 80.0],
            [3.0, 0.75, 72.0, 70.0],
            [6.0, 0.95, 92.0, 90.0],
            [2.5, 0.60, 58.0, 62.0],
            [4.5, 0.88, 85.0, 82.0],
            [3.5, 0.78, 74.0, 73.0],
            [5.5, 0.92, 90.0, 88.0],
            [1.5, 0.55, 48.0, 50.0],
            [4.2, 0.83, 80.0, 79.0],
            [2.2, 0.65, 62.0, 59.0],
            [5.8, 0.96, 94.0, 91.0],
            [3.8, 0.81, 77.0, 76.0],
            [1.2, 0.52, 44.0, 48.0],
            [4.8, 0.89, 87.0, 85.0],
            [3.2, 0.74, 70.0, 69.0],
            [5.2, 0.91, 89.0, 87.0],
            [2.8, 0.68, 64.0, 66.0],
        ],
        dtype=torch.float32,
    )

    # Labels:
    # 0 = likely to fail
    # 1 = likely to pass
    #
    # Important:
    # For CrossEntropyLoss, labels must be class indices with dtype torch.long.
    pass_labels = torch.tensor(
        [
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
        ],
        dtype=torch.long,
    )

    show_tensor("student_features", student_features)
    show_tensor("pass_labels", pass_labels)

    section("3. Normalize features")

    feature_mean = student_features.mean(dim=0, keepdim=True)
    feature_std = student_features.std(dim=0, keepdim=True)

    normalized_features = (student_features - feature_mean) / feature_std

    show_tensor("feature_mean", feature_mean)
    show_tensor("feature_std", feature_std)
    show_tensor("normalized_features", normalized_features)

    section("4. Create Dataset and split")

    dataset = StudentPassDataset(
        features=normalized_features,
        labels=pass_labels,
    )

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print("Total dataset size:", len(dataset))
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    section("5. Create DataLoaders")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    section("6. Create model, loss function, and optimizer")

    model = StudentPassClassifier().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    print(model)
    print("Loss function:", loss_fn)
    print("Optimizer:", optimizer)

    section("7. Check one batch before training")

    sample_features, sample_labels = next(iter(train_loader))
    sample_features = sample_features.to(device)
    sample_labels = sample_labels.to(device)

    sample_logits = model(sample_features)
    sample_loss = loss_fn(sample_logits, sample_labels)
    sample_probs = torch.softmax(sample_logits, dim=1)
    sample_predicted_classes = torch.argmax(sample_logits, dim=1)

    show_tensor("sample_features", sample_features)
    show_tensor("sample_labels", sample_labels)
    show_tensor("sample_logits", sample_logits)
    show_tensor("sample_probs", sample_probs)
    show_tensor("sample_predicted_classes", sample_predicted_classes)

    print("sample_loss:", sample_loss.item())

    section("8. Train and validate")

    num_epochs = 500

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if epoch % 50 == 0:
            print(
                f"epoch={epoch:04d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

    section("9. Predict new students")

    new_students = torch.tensor(
        [
            [4.0, 0.80, 80.0, 78.0],
            [1.0, 0.50, 42.0, 45.0],
            [5.5, 0.93, 91.0, 89.0],
        ],
        dtype=torch.float32,
    )

    logits, probabilities, predicted_classes = predict_one_student(
        model=model,
        student=new_students,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )

    show_tensor("new_students", new_students)
    show_tensor("logits", logits)
    show_tensor("probabilities", probabilities)
    show_tensor("predicted_classes", predicted_classes)

    print("\nClass meaning:")
    print("0 = fail")
    print("1 = pass")

    section("10. Save curves")

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_fig_path = output_dir / "lesson_08_classification_loss.png"
    acc_fig_path = output_dir / "lesson_08_classification_accuracy.png"

    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classification Loss")
    plt.legend()
    plt.savefig(loss_fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy")
    plt.legend()
    plt.savefig(acc_fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {loss_fig_path}")
    print(f"Accuracy curve saved to: {acc_fig_path}")


if __name__ == "__main__":
    main()