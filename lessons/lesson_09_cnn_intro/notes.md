# Lesson 09 - CNN Introduction

## Core idea

CNN stands for Convolutional Neural Network.

It is commonly used for image tasks.

## Image Tensor shape

PyTorch CNNs usually use:

```text
[N, C, H, W]
```

Meaning:

```text
N = batch size
C = channels
H = height
W = width
```

Example:

```text
[8, 1, 28, 28]
```

Means:

```text
8 grayscale images
1 channel
28 height
28 width
```

## Conv2d

A convolution layer can extract local image features.

Example:

```python
conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1,
)
```

Shape example:

```text
[8, 1, 28, 28] -> [8, 8, 28, 28]
```

Meaning:

```text
batch size stays 8
channels change from 1 to 8
height and width stay 28 because padding=1
```

## ReLU

ReLU does not change shape.

```text
negative values -> 0
positive values -> unchanged
```

## MaxPool2d

Max pooling reduces spatial size.

```python
pool = torch.nn.MaxPool2d(kernel_size=2)
```

Shape example:

```text
[8, 8, 28, 28] -> [8, 8, 14, 14]
```

Meaning:

```text
height and width are halved
```

## CNN shape flow

Example CNN:

```text
[8, 1, 28, 28]
-> Conv2d(1, 8)  -> [8, 8, 28, 28]
-> MaxPool2d(2)  -> [8, 8, 14, 14]
-> Conv2d(8, 16) -> [8, 16, 14, 14]
-> MaxPool2d(2)  -> [8, 16, 7, 7]
```

## Flatten

Flatten keeps batch size and flattens the remaining dimensions.

```python
flatten = torch.nn.Flatten()
```

Shape example:

```text
[8, 16, 7, 7] -> [8, 784]
```

Because:

```text
16 * 7 * 7 = 784
```

## Linear classifier

After feature extraction, use Linear layers for classification.

```python
classifier = torch.nn.Linear(16 * 7 * 7, 10)
```

Shape example:

```text
[8, 784] -> [8, 10]
```

Meaning:

```text
8 images
10 class logits per image
```

## CrossEntropyLoss for CNN classification

For classification:

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

Important:

```text
logits shape: [batch_size, num_classes]
labels shape: [batch_size]
labels dtype: torch.long
```

## Key takeaway

A simple CNN often follows this pattern:

```text
Conv2d -> ReLU -> Pool -> Conv2d -> ReLU -> Pool -> Flatten -> Linear
```