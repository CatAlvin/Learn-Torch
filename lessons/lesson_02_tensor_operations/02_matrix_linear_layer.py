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

    section("1. Matrix multiplication shape rule")

    a = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=torch.float32)

    b = torch.tensor([
        [10, 20],
        [30, 40],
        [50, 60],
    ], dtype=torch.float32)

    show("a", a)
    show("b", b)

    result = a @ b

    show("a @ b", result)

    print("\nShape rule:")
    print("[2, 3] @ [3, 2] = [2, 2]")

    section("2. One student, one prediction")

    # One student has 4 features:
    # study_hours, attendance_rate, homework_score, quiz_score
    student = torch.tensor([
        [5.0, 0.90, 88.0, 84.0]
    ], dtype=torch.float32)

    # 4 input features -> 1 output
    w = torch.tensor([
        [2.0],
        [20.0],
        [0.30],
        [0.40],
    ], dtype=torch.float32)

    b = torch.tensor([5.0], dtype=torch.float32)

    show("student", student)
    show("w", w)
    show("b", b)

    prediction = student @ w + b

    show("prediction = student @ w + b", prediction)

    print("\nManual explanation:")
    print("prediction = 5.0*2.0 + 0.90*20.0 + 88.0*0.30 + 84.0*0.40 + 5.0")

    section("3. Multiple students, one prediction per student")

    students = torch.tensor([
        [2.0, 0.70, 65.0, 60.0],
        [5.0, 0.90, 88.0, 84.0],
        [1.0, 0.50, 40.0, 45.0],
        [4.0, 0.85, 78.0, 80.0],
        [3.0, 0.75, 72.0, 70.0],
    ], dtype=torch.float32)

    show("students", students)

    predictions = students @ w + b

    show("predictions = students @ w + b", predictions)

    print("\nMeaning:")
    print("students shape   :", students.shape)
    print("w shape          :", w.shape)
    print("b shape          :", b.shape)
    print("predictions shape:", predictions.shape)

    section("4. Multiple outputs: pass score and risk score")

    # 4 input features -> 2 outputs
    # output 0: pass score
    # output 1: risk score
    w_multi = torch.tensor([
        [2.0, -1.0],
        [20.0, -10.0],
        [0.30, -0.20],
        [0.40, -0.30],
    ], dtype=torch.float32)

    b_multi = torch.tensor([5.0, 3.0], dtype=torch.float32)

    show("w_multi", w_multi)
    show("b_multi", b_multi)

    multi_outputs = students @ w_multi + b_multi

    show("multi_outputs = students @ w_multi + b_multi", multi_outputs)

    print("\nMeaning:")
    print("students shape    :", students.shape)
    print("w_multi shape     :", w_multi.shape)
    print("b_multi shape     :", b_multi.shape)
    print("multi_outputs shape:", multi_outputs.shape)
    print("Each student now has 2 output scores.")

    section("5. Connect to torch.nn.Linear")

    linear = torch.nn.Linear(in_features=4, out_features=2)

    show("linear.weight", linear.weight)
    show("linear.bias", linear.bias)

    linear_outputs = linear(students)

    show("linear_outputs = linear(students)", linear_outputs)

    print("\nImportant detail:")
    print("nn.Linear stores weight shape as [out_features, in_features].")
    print("But mathematically it still represents a linear transformation.")

    section("6. Compare manual linear calculation with nn.Linear")

    manual_w = linear.weight.T
    manual_b = linear.bias

    manual_outputs = students @ manual_w + manual_b

    show("manual_w = linear.weight.T", manual_w)
    show("manual_b = linear.bias", manual_b)
    show("manual_outputs = students @ manual_w + manual_b", manual_outputs)

    print("\nAre linear_outputs and manual_outputs close?")
    print(torch.allclose(linear_outputs, manual_outputs))

    section("7. Mini case: game character power score")

    # columns:
    # level, attack, defense, speed, magic_power
    characters = torch.tensor([
        [10.0, 80.0, 60.0, 55.0, 30.0],
        [12.0, 95.0, 70.0, 45.0, 25.0],
        [8.0, 60.0, 50.0, 80.0, 40.0],
        [15.0, 75.0, 90.0, 35.0, 20.0],
        [9.0, 50.0, 45.0, 90.0, 85.0],
        [11.0, 70.0, 65.0, 70.0, 60.0],
    ], dtype=torch.float32)

    # 5 attributes -> 1 power score
    power_w = torch.tensor([
        [1.5],
        [0.8],
        [0.6],
        [0.4],
        [0.7],
    ], dtype=torch.float32)

    power_b = torch.tensor([10.0], dtype=torch.float32)

    show("characters", characters)
    show("power_w", power_w)
    show("power_b", power_b)

    power_score = characters @ power_w + power_b

    show("power_score", power_score)

    print("\nShape check:")
    print("[6, 5] @ [5, 1] + [1] = [6, 1]")


if __name__ == "__main__":
    main()