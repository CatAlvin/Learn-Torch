import torch


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def describe_tensor(name, tensor):
    print(f"{name}:")
    print(tensor)
    print(f"shape : {tensor.shape}")
    print(f"dtype : {tensor.dtype}")
    print(f"device: {tensor.device}")


def main():
    torch.manual_seed(42)

    section("1. Shape: scalar, vector, matrix, image batch")

    scalar = torch.tensor(3.14)
    vector = torch.tensor([1, 2, 3, 4])
    matrix = torch.tensor([
        [80, 85, 90],
        [70, 88, 95],
    ])

    # A fake batch of 32 RGB images.
    # Shape meaning: batch_size, channels, height, width.
    image_batch = torch.rand(32, 3, 224, 224)

    describe_tensor("scalar", scalar)
    describe_tensor("vector", vector)
    describe_tensor("matrix", matrix)

    print("image_batch:")
    print(f"shape : {image_batch.shape}")
    print(f"dtype : {image_batch.dtype}")
    print(f"device: {image_batch.device}")

    section("2. Dtype: integer Tensor vs float Tensor")

    int_scores = torch.tensor([80, 90, 75, 88])
    float_scores = torch.tensor([80, 90, 75, 88], dtype=torch.float32)

    describe_tensor("int_scores", int_scores)
    describe_tensor("float_scores", float_scores)

    print("\nMean of int_scores:")
    print(int_scores.float().mean())

    print("\nMean of float_scores:")
    print(float_scores.mean())

    section("3. Dtype in classification labels")

    # Suppose we have 4 samples and 3 classes:
    # 0 = cat, 1 = dog, 2 = bird
    labels = torch.tensor([0, 2, 1, 0], dtype=torch.long)

    describe_tensor("labels", labels)

    print("\nFor classification tasks, labels are usually torch.long / torch.int64.")
    print("labels dtype:", labels.dtype)

    section("4. Device: CPU and GPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Selected device:", device)

    x_cpu = torch.rand(2, 3)
    describe_tensor("x_cpu", x_cpu)

    x_device = x_cpu.to(device)
    describe_tensor("x_device", x_device)

    section("5. Device mismatch example")

    a = torch.rand(2, 3).to(device)
    b = torch.rand(2, 3)

    describe_tensor("a", a)
    describe_tensor("b", b)

    try:
        c = a + b
        describe_tensor("c", c)
    except RuntimeError as error:
        print("This operation failed because tensors are on different devices.")
        print("Error message:")
        print(error)

    print("\nFix: move b to the same device as a.")

    b = b.to(device)
    c = a + b

    describe_tensor("fixed c", c)

    section("6. Mini case: student features and labels")

    # Each row represents one student.
    # Columns:
    # study_hours, attendance_rate, homework_score, quiz_score
    student_features = torch.tensor([
        [2.0, 0.70, 65.0, 60.0],
        [5.0, 0.90, 88.0, 84.0],
        [1.0, 0.50, 40.0, 45.0],
        [4.0, 0.85, 78.0, 80.0],
        [3.0, 0.75, 72.0, 70.0],
    ], dtype=torch.float32)

    # 0 = likely to fail
    # 1 = likely to pass
    student_labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)

    describe_tensor("student_features", student_features)
    describe_tensor("student_labels", student_labels)

    print("\nMeaning of student_features.shape:")
    print(f"{student_features.shape[0]} students")
    print(f"{student_features.shape[1]} features per student")

    section("7. Move mini case data to selected device")

    student_features = student_features.to(device)
    student_labels = student_labels.to(device)

    describe_tensor("student_features on device", student_features)
    describe_tensor("student_labels on device", student_labels)


if __name__ == "__main__":
    main()