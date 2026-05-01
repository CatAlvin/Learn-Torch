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

    section("1. Indexing a 1D Tensor")

    scores = torch.tensor([80, 90, 75, 88, 92])

    show("scores", scores)

    print("\nFirst score:")
    print(scores[0])

    print("\nLast score:")
    print(scores[-1])

    print("\nScores from index 1 to index 3:")
    print(scores[1:4])

    section("2. Indexing a 2D Tensor")

    # rows: students
    # columns: math, english, programming
    score_table = torch.tensor([
        [80, 85, 90],
        [70, 88, 95],
        [92, 76, 89],
        [60, 72, 68],
    ])

    show("score_table", score_table)

    print("\nFirst student:")
    print(score_table[0])

    print("\nFirst student's programming score:")
    print(score_table[0, 2])

    print("\nAll students' math scores:")
    print(score_table[:, 0])

    print("\nAll students' programming scores:")
    print(score_table[:, 2])

    print("\nFirst two students, all subjects:")
    print(score_table[:2, :])

    print("\nAll students, english and programming:")
    print(score_table[:, 1:])

    section("3. Boolean indexing")

    student_average = score_table.float().mean(dim=1)

    show("student_average", student_average)

    high_average_mask = student_average >= 85

    show("high_average_mask", high_average_mask)

    print("\nStudents whose average score is >= 85:")
    print(score_table[high_average_mask])

    section("4. reshape: change Tensor shape")

    x = torch.arange(12)

    show("x", x)

    x_3_by_4 = x.reshape(3, 4)
    show("x_3_by_4", x_3_by_4)

    x_2_by_6 = x.reshape(2, 6)
    show("x_2_by_6", x_2_by_6)

    section("5. reshape with -1")

    y = torch.arange(24)

    show("y", y)

    y_4_by_unknown = y.reshape(4, -1)
    show("y_4_by_unknown", y_4_by_unknown)

    y_unknown_by_3 = y.reshape(-1, 3)
    show("y_unknown_by_3", y_unknown_by_3)

    section("6. view: similar to reshape")

    z = torch.arange(12)
    z_view = z.view(3, 4)

    show("z", z)
    show("z_view", z_view)

    section("7. unsqueeze: add one dimension")

    one_student = torch.tensor([2.0, 0.70, 65.0, 60.0])

    show("one_student", one_student)

    one_student_batch = one_student.unsqueeze(dim=0)

    show("one_student_batch", one_student_batch)

    print("\nMeaning:")
    print("Before unsqueeze: one sample with 4 features")
    print("After unsqueeze : one batch with 1 sample and 4 features")

    section("8. squeeze: remove dimensions with size 1")

    prediction = torch.tensor([[0.82]])

    show("prediction", prediction)

    squeezed_prediction = prediction.squeeze()

    show("squeezed_prediction", squeezed_prediction)

    batch_prediction = torch.tensor([[0.82], [0.35], [0.91]])

    show("batch_prediction", batch_prediction)

    squeezed_batch_prediction = batch_prediction.squeeze(dim=1)

    show("squeezed_batch_prediction", squeezed_batch_prediction)

    section("9. Mini case: prepare one sample for a model")

    # Suppose a model expects input shape [batch_size, num_features].
    # But we only have one student's features with shape [num_features].
    single_student = torch.tensor([5.0, 0.90, 88.0, 84.0], dtype=torch.float32)

    show("single_student", single_student)

    model_input = single_student.unsqueeze(dim=0)

    show("model_input", model_input)

    print("\nNow the shape is [1, 4], which means:")
    print("1 sample in this batch")
    print("4 features for this sample")

    section("10. Mini case: flatten image batch")

    # Fake image batch:
    # 8 grayscale images, each image is 28 x 28.
    images = torch.rand(8, 1, 28, 28)

    show("images", images)

    flattened_images = images.reshape(8, -1)

    show("flattened_images", flattened_images)

    print("\nMeaning:")
    print("Before flattening: [8, 1, 28, 28]")
    print("After flattening : [8, 784]")
    print("Each image becomes a vector with 1 * 28 * 28 = 784 features.")


if __name__ == "__main__":
    main()