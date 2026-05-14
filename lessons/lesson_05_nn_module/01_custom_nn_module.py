from pathlib import Path

import torch
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


class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        output = self.linear(x)
        return output


class StudentScoreModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=1),
        )

    def forward(self, x):
        output = self.network(x)
        return output


def train_model(model, x, y, num_epochs=1000, learning_rate=0.01):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(num_epochs + 1):
        model.train()

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 200 == 0:
            print(f"epoch={epoch:04d} | loss={loss.item():.6f}")

    return loss_history


def main():
    torch.manual_seed(42)

    section("1. Select device")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    section("2. Simple custom model: y = wx + b")

    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        device=device,
    )

    y = torch.tensor(
        [[5.0], [8.0], [11.0], [14.0], [17.0]],
        device=device,
    )

    show_tensor("x", x)
    show_tensor("y", y)

    simple_model = SimpleLinearModel().to(device)

    print("\nSimpleLinearModel structure:")
    print(simple_model)

    print("\nSimpleLinearModel parameters before training:")
    for name, param in simple_model.named_parameters():
        print(name, param.data)

    simple_loss_history = train_model(
        model=simple_model,
        x=x,
        y=y,
        num_epochs=500,
        learning_rate=0.01,
    )

    print("\nSimpleLinearModel parameters after training:")
    for name, param in simple_model.named_parameters():
        print(name, param.data)

    section("3. Test simple custom model")

    test_x = torch.tensor(
        [[6.0], [7.0], [8.0]],
        device=device,
    )

    simple_model.eval()

    with torch.no_grad():
        test_pred = simple_model(test_x)

    show_tensor("test_x", test_x)
    show_tensor("test_pred", test_pred)

    print("\nExpected values:")
    print("x = 6 -> y = 20")
    print("x = 7 -> y = 23")
    print("x = 8 -> y = 26")

    section("4. Multi-feature student score model")

    # Each row is one student.
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
        ],
        dtype=torch.float32,
        device=device,
    )

    # Target: final exam score
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
        ],
        dtype=torch.float32,
        device=device,
    )

    show_tensor("student_features", student_features)
    show_tensor("final_scores", final_scores)

    section("5. Feature normalization")

    feature_mean = student_features.mean(dim=0, keepdim=True)
    feature_std = student_features.std(dim=0, keepdim=True)

    normalized_features = (student_features - feature_mean) / feature_std

    show_tensor("feature_mean", feature_mean)
    show_tensor("feature_std", feature_std)
    show_tensor("normalized_features", normalized_features)

    section("6. Create and train StudentScoreModel")

    student_model = StudentScoreModel().to(device)

    print("\nStudentScoreModel structure:")
    print(student_model)

    print("\nStudentScoreModel parameters:")
    for name, param in student_model.named_parameters():
        print(name, param.shape)

    student_loss_history = train_model(
        model=student_model,
        x=normalized_features,
        y=final_scores,
        num_epochs=2000,
        learning_rate=0.001,
    )

    section("7. Predict a new student")

    new_student = torch.tensor(
        [[4.0, 0.80, 80.0, 78.0]],
        dtype=torch.float32,
        device=device,
    )

    normalized_new_student = (new_student - feature_mean) / feature_std

    student_model.eval()

    with torch.no_grad():
        predicted_score = student_model(normalized_new_student)

    show_tensor("new_student", new_student)
    show_tensor("normalized_new_student", normalized_new_student)
    show_tensor("predicted_score", predicted_score)

    section("8. Save loss curves")

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_fig_path = output_dir / "lesson_05_simple_custom_model_loss.png"
    student_fig_path = output_dir / "lesson_05_student_score_model_loss.png"

    plt.figure()
    plt.plot(simple_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Simple Custom Model Loss")
    plt.savefig(simple_fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(student_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Student Score Model Loss")
    plt.savefig(student_fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Simple model loss curve saved to: {simple_fig_path}")
    print(f"Student model loss curve saved to: {student_fig_path}")


if __name__ == "__main__":
    main()