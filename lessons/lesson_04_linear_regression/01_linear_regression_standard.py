from pathlib import Path

import torch
import matplotlib.pyplot as plt


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_model_parameters(model):
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(param)
        print("shape:", param.shape)
        print("requires_grad:", param.requires_grad)


def main():
    torch.manual_seed(42)

    section("1. Prepare training data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # True relationship:
    # y = 3x + 2
    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        device=device,
    )

    y = torch.tensor(
        [[5.0], [8.0], [11.0], [14.0], [17.0]],
        device=device,
    )

    print("x:")
    print(x)
    print("x shape:", x.shape)

    print("\ny:")
    print(y)
    print("y shape:", y.shape)

    section("2. Create model")

    # nn.Linear(1, 1) means:
    # input feature count = 1
    # output feature count = 1
    model = torch.nn.Linear(in_features=1, out_features=1)

    # Move model parameters to CPU or GPU
    model = model.to(device)

    show_model_parameters(model)

    section("3. Create loss function and optimizer")

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    print("Loss function:", loss_fn)
    print("Optimizer:", optimizer)

    section("4. Train model")

    num_epochs = 500
    loss_history = []

    for epoch in range(num_epochs + 1):
        # Forward pass
        pred = model(x)

        # Compute loss
        loss = loss_fn(pred, y)

        # Clear old gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 100 == 0:
            weight = model.weight.item()
            bias = model.bias.item()
            print(
                f"epoch={epoch:03d} | "
                f"weight={weight:.4f} | "
                f"bias={bias:.4f} | "
                f"loss={loss.item():.6f}"
            )

    section("5. Final learned model")

    learned_w = model.weight.item()
    learned_b = model.bias.item()

    print(f"Learned weight: {learned_w:.4f}")
    print(f"Learned bias  : {learned_b:.4f}")

    print("\nTrue relationship:")
    print("y = 3 * x + 2")

    print("\nLearned relationship:")
    print(f"y = {learned_w:.4f} * x + {learned_b:.4f}")

    section("6. Test model")

    test_x = torch.tensor(
        [[6.0], [7.0], [8.0]],
        device=device,
    )

    model.eval()

    with torch.no_grad():
        test_pred = model(test_x)

    print("test_x:")
    print(test_x)

    print("\ntest_pred:")
    print(test_pred)

    print("\nExpected:")
    print("x = 6 -> y = 20")
    print("x = 7 -> y = 23")
    print("x = 8 -> y = 26")

    section("7. Save loss curve")

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "lesson_04_linear_regression_standard_loss.png"

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Linear Regression Training Loss")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {fig_path}")


if __name__ == "__main__":
    main()