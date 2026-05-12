# Lesson 02 - Tensor Operations and Broadcasting

## Core idea

Neural networks are built from Tensor operations.

Common operations include:

- element-wise operations
- aggregation operations
- broadcasting
- matrix multiplication

## Element-wise operations

Element-wise operations calculate values position by position.

```python
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([10, 20, 30], dtype=torch.float32)

a + b
a * b
```

Important:

```text
* means element-wise multiplication.
@ means matrix multiplication.
```

## Aggregation

Aggregation operations reduce Tensor values.

```python
x.sum()
x.mean()
x.max()
x.min()
```

For a score table:

```python
score_table.mean(dim=1)
score_table.mean(dim=0)
```

Meaning:

```text
mean(dim=1): average of each row
mean(dim=0): average of each column
```

## keepdim

`keepdim=True` keeps the reduced dimension.

```python
student_mean = score_table.mean(dim=1, keepdim=True)
```

Shape example:

```text
Without keepdim: [4]
With keepdim   : [4, 1]
```

This is useful for broadcasting.

## Broadcasting

Broadcasting allows PyTorch to automatically expand compatible Tensor shapes.

Example:

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
], dtype=torch.float32)

bias = torch.tensor([10, 20, 30], dtype=torch.float32)

y = x + bias
```

Shape:

```text
x    : [2, 3]
bias : [3]
y    : [2, 3]
```

## Broadcasting rule

A practical rule:

```text
Compare shapes from right to left.
Dimensions are compatible if:
1. they are equal, or
2. one of them is 1, or
3. one dimension is missing.
```

## Normalization

Feature normalization is common in machine learning.

```python
feature_mean = features.mean(dim=0, keepdim=True)
feature_std = features.std(dim=0, keepdim=True)

normalized_features = (features - feature_mean) / feature_std
```

After normalization:

```text
Each feature column has mean close to 0.
Each feature column has standard deviation close to 1.
```

## Matrix multiplication

```python
output = x @ w + b
```

This is similar to one linear layer.

Shape example:

```text
x      : [5, 4]
w      : [4, 1]
b      : [1]
output : [5, 1]
```