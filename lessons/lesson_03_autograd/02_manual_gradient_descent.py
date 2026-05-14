from pathlib import Path

import torch
import matplotlib.pyplot as plt


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_param(step, w, b, loss):
    print(
        f"step={step:03d} | "
        f"w={w.item():.4f} | "
        f"b={b.item():.4f} | "
        f"loss={loss.item():.6f}"
    )


def main():
    torch.manual_seed(42)

    section("1. Prepare training data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # x: study hours
    # y: exam score
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], device=device)
    y = torch.tensor([[5.0], [8.0], [11.0], [14.0], [17.0]], device=device)

    print("x:")
    print(x)
    print("x shape:", x.shape)

    print("\ny:")
    print(y)
    print("y shape:", y.shape)

    section("2. Initialize learnable parameters")

    # We want the model to learn:
    # y = w * x + b
    #
    # The true values are:
    # w = 3
    # b = 2
    #
    # But the model does not know them at the beginning.

    w = torch.randn(1, 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)

    print("Initial w:", w.item())
    print("Initial b:", b.item())

    section("3. Train with manual gradient descent")

    learning_rate = 0.01
    num_steps = 300

    loss_history = []

    for step in range(num_steps + 1):
        # Forward pass
        pred = x @ w + b

        # Compute mean squared error
        loss = ((pred - y) ** 2).mean()

        # Clear old gradients
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()

        # Backward pass
        loss.backward()

        # Update parameters
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        loss_history.append(loss.item())

        if step % 50 == 0:
            show_param(step, w, b, loss)

    section("4. Final learned parameters")

    print(f"Learned w: {w.item():.4f}")
    print(f"Learned b: {b.item():.4f}")

    print("\nTrue relationship:")
    print("y = 3 * x + 2")

    print("\nLearned relationship:")
    print(f"y = {w.item():.4f} * x + {b.item():.4f}")

    section("5. Test the trained model")

    test_x = torch.tensor([[6.0], [7.0], [8.0]], device=device)

    with torch.no_grad():
        test_pred = test_x @ w + b

    print("test_x:")
    print(test_x)

    print("\ntest_pred:")
    print(test_pred)

    print("\nExpected values:")
    print("x = 6 -> y = 20")
    print("x = 7 -> y = 23")
    print("x = 8 -> y = 26")

    section("6. Save loss curve")

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "lesson_03_manual_gradient_descent_loss.png"

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Manual Gradient Descent Loss Curve")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {fig_path}")


if __name__ == "__main__":
    main()