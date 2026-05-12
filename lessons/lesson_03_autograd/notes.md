# Lesson 03 - Autograd Basics

## Core idea

Autograd is PyTorch's automatic differentiation engine.

It records Tensor operations and automatically computes gradients.

## requires_grad

If a Tensor needs gradients, set:

!!!python
x = torch.tensor(2.0, requires_grad=True)
!!!

This tells PyTorch to track operations involving `x`.

## Basic example

!!!python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()

print(x.grad)
!!!

Math:

!!!text
y = x^2 + 3x + 1
dy/dx = 2x + 3
when x = 2, dy/dx = 7
!!!

## backward

`backward()` starts backpropagation from the output Tensor.

!!!python
y.backward()
!!!

The gradient is stored in:

!!!python
x.grad
!!!

## Gradient meaning

A gradient tells us how the output changes when a variable changes slightly.

For model training:

!!!text
gradient tells us how loss changes when a parameter changes
!!!

## Gradient descent

A simple update rule:

!!!python
w = w - learning_rate * w.grad
!!!

Meaning:

!!!text
Move the parameter in the direction that reduces loss.
!!!

## Gradient accumulation

PyTorch accumulates gradients by default.

So before the next backward pass, clear old gradients:

!!!python
w.grad.zero_()
!!!

In real training loops, we usually write:

!!!python
optimizer.zero_grad()
loss.backward()
optimizer.step()
!!!

## torch.no_grad

When manually updating parameters, use:

!!!python
with torch.no_grad():
    w -= learning_rate * w.grad
!!!

This prevents PyTorch from tracking the update operation.

## Key training flow

!!!text
1. forward
2. compute loss
3. backward
4. update parameters
5. clear gradients
!!!