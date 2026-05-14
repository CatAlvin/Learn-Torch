# Lesson 05 - Custom Models with nn.Module

## Core idea

`torch.nn.Module` is the base class for PyTorch models.

A custom model usually looks like this:

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        output = self.layer(x)
        return output
```

## Why use nn.Module?

Using `nn.Module` allows PyTorch to manage:

- layers
- parameters
- gradients
- model saving and loading
- train and eval modes

## super().__init__()

Inside `__init__`, usually write:

```python
super().__init__()
```

This initializes the parent `nn.Module` class.

Without it, PyTorch may not properly manage the model.

## Registering layers

When we write:

```python
self.linear = torch.nn.Linear(1, 1)
```

PyTorch automatically registers this layer and its parameters.

So these parameters appear in:

```python
model.parameters()
```

## forward

`forward` defines how input data flows through the model.

```python
def forward(self, x):
    output = self.linear(x)
    return output
```

When we call:

```python
pred = model(x)
```

PyTorch internally calls the model's forward logic.

## Sequential

`torch.nn.Sequential` connects layers in order.

Example:

```python
self.network = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
)
```

Meaning:

```text
4 input features
-> 8 hidden features
-> ReLU activation
-> 1 output value
```

## ReLU

ReLU is an activation function.

```text
ReLU(x) = max(0, x)
```

It adds non-linearity to the model.

Without activation functions, stacking multiple Linear layers is still just a linear transformation.

## named_parameters

We can inspect model parameters:

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

Example:

```text
network.0.weight
network.0.bias
network.2.weight
network.2.bias
```

ReLU does not have trainable parameters.

## Training function

We can wrap the training loop in a function:

```python
def train_model(model, x, y, num_epochs=1000, learning_rate=0.01):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Key takeaway

A custom `nn.Module` lets us build larger and more flexible neural networks.