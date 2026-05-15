from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
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


class StudentScoreDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y


class StudentScoreModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=1),
        )

    def forward(self, x):
        return self.network(x)


def train_with_dataloader(model, dataloader, device, num_epochs=500, learning_rate=0.001):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(num_epochs + 1):
        model.train()

        epoch_loss = 0.0

        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            pred = model(batch_features)
            loss = loss_fn(pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(average_epoch_loss)

        if epoch % 100 == 0:
            print(f"epoch={epoch:04d} | loss={average_epoch_loss:.6f}")

    return loss_history


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Prepare raw Tensor data")

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
        ],
        dtype=torch.float32,
    )

    final_scores = torch.tensor(
        [
            [68.0],
            [90.0],
            [45.0],
            [82.0],
            [74.0],
            [95.0],
            [60.0],
            [86.0],
            [76.0],
            [92.0],
            [52.0],
            [83.0],
        ],
        dtype=torch.float32,
    )

    show_tensor("student_features", student_features)
    show_tensor("final_scores", final_scores)

    section("3. Normalize features")

    feature_mean = student_features.mean(dim=0, keepdim=True)
    feature_std = student_features.std(dim=0, keepdim=True)

    normalized_features = (student_features - feature_mean) / feature_std

    show_tensor("feature_mean", feature_mean)
    show_tensor("feature_std", feature_std)
    show_tensor("normalized_features", normalized_features)

    section("4. Create Dataset")

    dataset = StudentScoreDataset(
        features=normalized_features,
        labels=final_scores,
    )

    print("Dataset length:", len(dataset))

    first_x, first_y = dataset[0]

    show_tensor("First sample features", first_x)
    show_tensor("First sample label", first_y)

    section("5. Create DataLoader")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
    )

    print("Number of batches:", len(dataloader))

    for batch_index, (batch_features, batch_labels) in enumerate(dataloader):
        print(f"\nBatch {batch_index}")
        show_tensor("batch_features", batch_features)
        show_tensor("batch_labels", batch_labels)

    section("6. Train model with DataLoader")

    model = StudentScoreModel().to(device)

    print("Model structure:")
    print(model)

    loss_history = train_with_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        num_epochs=500,
        learning_rate=0.001,
    )

    section("7. Predict a new student")

    new_student = torch.tensor(
        [[4.0, 0.80, 80.0, 78.0]],
        dtype=torch.float32,
    )

    normalized_new_student = (new_student - feature_mean) / feature_std
    normalized_new_student = normalized_new_student.to(device)

    model.eval()

    with torch.no_grad():
        predicted_score = model(normalized_new_student)

    show_tensor("new_student", new_student)
    show_tensor("normalized_new_student", normalized_new_student)
    show_tensor("predicted_score", predicted_score)

    section("8. Save loss curve")

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "lesson_06_dataloader_training_loss.png"

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training with DataLoader")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {fig_path}")


if __name__ == "__main__":
    main()