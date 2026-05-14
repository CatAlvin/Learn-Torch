# Lesson 04 - Linear Regression with PyTorch Standard Components

## Core idea

In the previous lesson, we manually created parameters and updated them.

In this lesson, we use PyTorch standard components:

- `torch.nn.Linear`
- `torch.nn.MSELoss`
- `torch.optim.SGD`

## Model

A linear regression model can be written as:

```text
y_hat = wx + b
```

In PyTorch:

```python
model = torch.nn.Linear(in_features=1, out_features=1)
```

This layer automatically creates:

```text
weight
bias
```

## Loss function

For regression, we often use mean squared error.

Manual version:

```python
loss = ((pred - y) ** 2).mean()
```

PyTorch version:

```python
loss_fn = torch.nn.MSELoss()
loss = loss_fn(pred, y)
```

## Optimizer

Manual update:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

PyTorch optimizer:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
```

## Standard training loop

```python
for epoch in range(num_epochs):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Training loop meaning

```text
1. model(x): make predictions
2. loss_fn(pred, y): compute loss
3. optimizer.zero_grad(): clear old gradients
4. loss.backward(): compute new gradients
5. optimizer.step(): update parameters
```

## train mode and eval mode

Training mode:

```python
model.train()
```

Evaluation mode:

```python
model.eval()
```

During prediction, usually use:

```python
with torch.no_grad():
    pred = model(test_x)
```

## Key takeaway

The standard PyTorch training loop is a cleaner version of manual gradient descent.