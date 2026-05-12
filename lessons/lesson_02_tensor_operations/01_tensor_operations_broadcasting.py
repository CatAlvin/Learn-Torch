import torch


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(name, tensor):
    print(f"{name}:")
    print(tensor)
    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)
    print("device:", tensor.device)


def main():
    torch.manual_seed(42)

    section("1. Element-wise operations")

    a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    b = torch.tensor([10, 20, 30, 40], dtype=torch.float32)

    show("a", a)
    show("b", b)

    show("a + b", a + b)
    show("b - a", b - a)
    show("a * b", a * b)
    show("b / a", b / a)
    show("a ** 2", a ** 2)

    section("2. Tensor and scalar operations")

    scores = torch.tensor([60, 70, 80, 90], dtype=torch.float32)

    show("scores", scores)
    show("scores + 5", scores + 5)
    show("scores * 1.1", scores * 1.1)
    show("scores / 100", scores / 100)

    section("3. Aggregation operations")

    score_table = torch.tensor([
        [80, 85, 90],
        [70, 88, 95],
        [92, 76, 89],
        [60, 72, 68],
    ], dtype=torch.float32)

    show("score_table", score_table)

    print("Total sum:")
    print(score_table.sum())

    print("\nOverall mean:")
    print(score_table.mean())

    print("\nMax value:")
    print(score_table.max())

    print("\nMin value:")
    print(score_table.min())

    print("\nMean of each student, dim=1:")
    student_mean = score_table.mean(dim=1)
    show("student_mean", student_mean)

    print("\nMean of each subject, dim=0:")
    subject_mean = score_table.mean(dim=0)
    show("subject_mean", subject_mean)

    section("4. keepdim=True")

    student_mean_keepdim = score_table.mean(dim=1, keepdim=True)
    subject_mean_keepdim = score_table.mean(dim=0, keepdim=True)

    show("student_mean_keepdim", student_mean_keepdim)
    show("subject_mean_keepdim", subject_mean_keepdim)

    print("\nSubtract each student's own average:")
    centered_by_student = score_table - student_mean_keepdim
    show("centered_by_student", centered_by_student)

    section("5. Broadcasting: vector plus matrix")

    x = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=torch.float32)

    bias = torch.tensor([10, 20, 30], dtype=torch.float32)

    show("x", x)
    show("bias", bias)

    y = x + bias

    show("x + bias", y)

    print("\nMeaning:")
    print("x shape    :", x.shape)
    print("bias shape :", bias.shape)
    print("result     :", y.shape)
    print("bias is automatically expanded from [3] to [2, 3].")

    section("6. Broadcasting in student score adjustment")

    # Suppose different subjects have different bonus points.
    # math + 5, english + 3, programming + 10
    subject_bonus = torch.tensor([5, 3, 10], dtype=torch.float32)

    adjusted_scores = score_table + subject_bonus

    show("subject_bonus", subject_bonus)
    show("adjusted_scores", adjusted_scores)

    section("7. Normalization example")

    features = torch.tensor([
        [2.0, 0.70, 65.0, 60.0],
        [5.0, 0.90, 88.0, 84.0],
        [1.0, 0.50, 40.0, 45.0],
        [4.0, 0.85, 78.0, 80.0],
        [3.0, 0.75, 72.0, 70.0],
    ], dtype=torch.float32)

    show("features", features)

    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, keepdim=True)

    show("feature_mean", feature_mean)
    show("feature_std", feature_std)

    normalized_features = (features - feature_mean) / feature_std

    show("normalized_features", normalized_features)

    print("\nCheck normalized mean:")
    print(normalized_features.mean(dim=0))

    print("\nCheck normalized std:")
    print(normalized_features.std(dim=0))

    section("8. Important difference: element-wise multiply vs matrix multiply")

    m1 = torch.tensor([
        [1, 2],
        [3, 4],
    ], dtype=torch.float32)

    m2 = torch.tensor([
        [10, 20],
        [30, 40],
    ], dtype=torch.float32)

    show("m1", m1)
    show("m2", m2)

    elementwise_product = m1 * m2
    matrix_product = m1 @ m2

    show("m1 * m2", elementwise_product)
    show("m1 @ m2", matrix_product)

    print("\nImportant:")
    print("* means element-wise multiplication.")
    print("@ means matrix multiplication.")

    section("9. Mini neural-network-like calculation")

    # 5 students, 4 features each
    x = features

    # 4 input features -> 1 output score
    w = torch.tensor([
        [0.30],
        [10.00],
        [0.20],
        [0.25],
    ], dtype=torch.float32)

    b = torch.tensor([5.0], dtype=torch.float32)

    show("x", x)
    show("w", w)
    show("b", b)

    output = x @ w + b

    show("output = x @ w + b", output)

    print("\nMeaning:")
    print("x shape:", x.shape)
    print("w shape:", w.shape)
    print("b shape:", b.shape)
    print("output shape:", output.shape)
    print("This is similar to one Linear layer.")


if __name__ == "__main__":
    main()