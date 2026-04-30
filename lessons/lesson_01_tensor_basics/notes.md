# Lesson 01 - Tensor Basics

## Core idea

A Tensor is the basic data structure in PyTorch.

It is similar to a NumPy array, but it can also be used for GPU computation and automatic differentiation.

## Why Tensor matters

In PyTorch, almost everything is represented as Tensor:

- input data
- model output
- model parameters
- loss value
- gradients

## Common Tensor shapes

| Data | Shape example |
|---|---|
| scalar | `torch.Size([])` |
| vector | `torch.Size([4])` |
| table | `torch.Size([4, 3])` |
| grayscale image | `torch.Size([28, 28])` |
| RGB image | `torch.Size([3, 224, 224])` |
| batch of RGB images | `torch.Size([32, 3, 224, 224])` |

## Important rule about dim

`dim` means the dimension to reduce.

Example:

```python
score_table = torch.tensor([
    [80, 85, 90],
    [70, 88, 95],
    [92, 76, 89],
    [60, 72, 68],
])
```

The shape is:

```python
torch.Size([4, 3])
```

`mean(dim=1)` calculates the average of each row.

`mean(dim=0)` calculates the average of each column.

## Device

A Tensor can be stored on CPU or GPU.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Move Tensor to GPU:

```python
x = x.to("cuda")
```

Move Tensor back to CPU:

```python
x = x.to("cpu")
```

## Key takeaway

Before training any neural network, always check:

- shape
- dtype
- device

## Shape, dtype, and device

When debugging PyTorch code, always check three things:

- `shape`
- `dtype`
- `device`

### Shape

Shape tells us the structure of the data.

```python
x = torch.rand(32, 3, 224, 224)
print(x.shape)
```

This usually means:

```text
32 images
3 color channels
224 height
224 width
```

### Dtype

Dtype tells us the data type.

Common rules:

```text
features: torch.float32
classification labels: torch.long / torch.int64
```

### Device

Device tells us where the Tensor is stored.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

Important rule:

```text
The model and data must be on the same device.
```

Otherwise, PyTorch will raise a device mismatch error.