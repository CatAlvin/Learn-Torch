import torch
import numpy as np


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("1. Create Tensor from Python list")

    scores = torch.tensor([80, 90, 75, 88])
    print("scores:")
    print(scores)
    print("shape:", scores.shape)
    print("dtype:", scores.dtype)
    print("device:", scores.device)

    section("2. Create 2D Tensor: a small score table")

    # rows: students
    # columns: math, english, programming
    score_table = torch.tensor([
        [80, 85, 90],
        [70, 88, 95],
        [92, 76, 89],
        [60, 72, 68],
    ])

    print("score_table:")
    print(score_table)
    print("shape:", score_table.shape)
    print("dtype:", score_table.dtype)

    section("3. Create Tensor with specific dtype")

    float_scores = torch.tensor([80, 90, 75, 88], dtype=torch.float32)
    print("float_scores:")
    print(float_scores)
    print("dtype:", float_scores.dtype)

    section("4. Common Tensor creation functions")

    zeros_tensor = torch.zeros(2, 3)
    ones_tensor = torch.ones(2, 3)
    random_tensor = torch.rand(2, 3)
    normal_tensor = torch.randn(2, 3)
    range_tensor = torch.arange(0, 10, 2)
    line_tensor = torch.linspace(0, 1, 5)

    print("zeros_tensor:")
    print(zeros_tensor)

    print("\nones_tensor:")
    print(ones_tensor)

    print("\nrandom_tensor, values are in [0, 1):")
    print(random_tensor)

    print("\nnormal_tensor, values follow normal distribution:")
    print(normal_tensor)

    print("\nrange_tensor:")
    print(range_tensor)

    print("\nline_tensor:")
    print(line_tensor)

    section("5. Convert NumPy array to Tensor")

    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_numpy = torch.from_numpy(np_array)

    print("np_array:")
    print(np_array)
    print(type(np_array))

    print("\ntensor_from_numpy:")
    print(tensor_from_numpy)
    print(type(tensor_from_numpy))

    section("6. Convert Tensor back to NumPy array")

    back_to_numpy = tensor_from_numpy.numpy()

    print("back_to_numpy:")
    print(back_to_numpy)
    print(type(back_to_numpy))

    section("7. A small practice: calculate average score")

    score_table_float = score_table.float()

    student_average = score_table_float.mean(dim=1)
    subject_average = score_table_float.mean(dim=0)

    print("score_table_float:")
    print(score_table_float)

    print("\nstudent_average:")
    print(student_average)

    print("\nsubject_average:")
    print(subject_average)

    section("8. Check CUDA device")

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        gpu_tensor = score_table_float.to("cuda")
        print("gpu_tensor:")
        print(gpu_tensor)
        print("device:", gpu_tensor.device)

        cpu_tensor = gpu_tensor.to("cpu")
        print("\ncpu_tensor:")
        print(cpu_tensor)
        print("device:", cpu_tensor.device)


if __name__ == "__main__":
    main()