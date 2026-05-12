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
    print("requires_grad:", tensor.requires_grad)
    print("grad_fn:", tensor.grad_fn)


def main():
    torch.manual_seed(42)

    section("1. Basic autograd example")

    x = torch.tensor(2.0, requires_grad=True)

    y = x ** 2 + 3 * x + 1

    show("x", x)
    show("y", y)

    y.backward()

    print("\nx.grad:")
    print(x.grad)

    print("\nManual check:")
    print("y = x^2 + 3x + 1")
    print("dy/dx = 2x + 3")
    print("when x = 2, dy/dx = 7")

    section("2. Computation graph intuition")

    a = torch.tensor(3.0, requires_grad=True)

    b = a * 2
    c = b + 5
    d = c ** 2

    show("a", a)
    show("b = a * 2", b)
    show("c = b + 5", c)
    show("d = c ** 2", d)

    d.backward()

    print("\na.grad:")
    print(a.grad)

    print("\nManual check:")
    print("b = 2a")
    print("c = b + 5 = 2a + 5")
    print("d = c^2 = (2a + 5)^2")
    print("dd/da = 2 * (2a + 5) * 2")
    print("when a = 3, dd/da = 2 * 11 * 2 = 44")

    section("3. Gradient for multiple variables")

    w = torch.tensor(4.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x_input = torch.tensor(3.0)

    pred = w * x_input + b

    target = torch.tensor(15.0)

    loss = (pred - target) ** 2

    show("w", w)
    show("b", b)
    show("x_input", x_input)
    show("pred", pred)
    show("target", target)
    show("loss", loss)

    loss.backward()

    print("\nw.grad:")
    print(w.grad)

    print("\nb.grad:")
    print(b.grad)

    print("\nMeaning:")
    print("w.grad tells us how loss changes when w changes.")
    print("b.grad tells us how loss changes when b changes.")

    section("4. Gradient accumulation")

    p = torch.tensor(2.0, requires_grad=True)

    y1 = p ** 2
    y1.backward()

    print("After first backward, p.grad:")
    print(p.grad)

    y2 = p ** 2
    y2.backward()

    print("\nAfter second backward, p.grad:")
    print(p.grad)

    print("\nImportant:")
    print("PyTorch accumulates gradients by default.")
    print("So we need to clear gradients before the next backward in training.")

    p.grad.zero_()

    print("\nAfter p.grad.zero_(), p.grad:")
    print(p.grad)

    section("5. Why zero_grad is needed in training")

    weight = torch.tensor(1.0, requires_grad=True)

    learning_rate = 0.1

    for step in range(3):
        prediction = weight * 2
        target = torch.tensor(10.0)
        loss = (prediction - target) ** 2

        loss.backward()

        print(f"\nStep {step}")
        print("weight before update:", weight.item())
        print("loss:", loss.item())
        print("weight.grad:", weight.grad.item())

        with torch.no_grad():
            weight -= learning_rate * weight.grad

        weight.grad.zero_()

        print("weight after update:", weight.item())

    section("6. torch.no_grad")

    q = torch.tensor(5.0, requires_grad=True)

    r = q * 2
    show("r = q * 2", r)

    with torch.no_grad():
        s = q * 2

    show("s = q * 2 inside no_grad", s)

    print("\nMeaning:")
    print("torch.no_grad() disables gradient tracking.")
    print("It is often used during model evaluation or manual parameter updates.")

    section("7. Mini case: one-step linear model update")

    # Goal:
    # We want a simple model y = w*x + b.
    # Given x = 2, target y = 10.
    # Let PyTorch tell us how w and b should move.

    x = torch.tensor(2.0)
    target = torch.tensor(10.0)

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    pred = w * x + b
    loss = (pred - target) ** 2

    show("pred", pred)
    show("loss", loss)

    loss.backward()

    print("\nw.grad:")
    print(w.grad)

    print("\nb.grad:")
    print(b.grad)

    learning_rate = 0.1

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    print("\nAfter one update:")
    print("w:", w)
    print("b:", b)

    new_pred = w * x + b

    print("\nnew_pred:")
    print(new_pred)

    print("\nOld prediction was 2.0.")
    print("Target is 10.0.")
    print("After one update, prediction moves closer to target.")


if __name__ == "__main__":
    main()