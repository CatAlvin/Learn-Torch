# Lesson 03 - Autograd Basics

## Core idea

Autograd is PyTorch's automatic differentiation engine.

It records Tensor operations and automatically computes gradients.

## requires_grad

If a Tensor needs gradients, set:

```python
x = torch.tensor(2.0, requires_grad=True)
```

This tells PyTorch to track operations involving `x`.

## Basic example

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()

print(x.grad)
```

Math:

```text
y = x^2 + 3x + 1
dy/dx = 2x + 3
when x = 2, dy/dx = 7
```

## backward

`backward()` starts backpropagation from the output Tensor.

```python
y.backward()
```

The gradient is stored in:

```python
x.grad
```

## Gradient meaning

A gradient tells us how the output changes when a variable changes slightly.

For model training:

```text
gradient tells us how loss changes when a parameter changes
```

## Gradient descent

A simple update rule:

```python
w = w - learning_rate * w.grad
```

Meaning:

```text
Move the parameter in the direction that reduces loss.
```

## Gradient accumulation

PyTorch accumulates gradients by default.

So before the next backward pass, clear old gradients:

```python
w.grad.zero_()
```

In real training loops, we usually write:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## torch.no_grad

When manually updating parameters, use:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
```

This prevents PyTorch from tracking the update operation.

## Key training flow

```text
1. forward
2. compute loss
3. backward
4. update parameters
5. clear gradients
```

## Manual gradient descent

### Goal

We want the model to learn a simple linear relationship:

```text
y = 3x + 2
```

But the model starts with random parameters.

### Model

```python
pred = x @ w + b
```

Shape example:

```text
x    : [5, 1]
w    : [1, 1]
b    : [1]
pred : [5, 1]
```

### Loss

We use mean squared error:

```python
loss = ((pred - y) ** 2).mean()
```

Meaning:

```text
The loss measures how far predictions are from true values.
```

### Manual training loop

```python
for step in range(num_steps):
    pred = x @ w + b
    loss = ((pred - y) ** 2).mean()

    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
```

### Training flow

```text
1. forward
2. compute loss
3. clear old gradients
4. backward
5. update parameters
6. repeat
```

### Key idea

The model learns by repeatedly adjusting `w` and `b` to reduce loss.

### Learning rate

The learning rate controls the step size of parameter updates.

```text
Too small: training is very slow.
Too large: loss may oscillate or diverge.
Reasonable: loss decreases steadily.
```