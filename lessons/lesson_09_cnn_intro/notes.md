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

## FashionMNIST CNN training

### Dataset

FashionMNIST is an image classification dataset.

It has 10 classes:

```text
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
```

Each image is grayscale:

```text
[1, 28, 28]
```

A batch of images has shape:

```text
[batch_size, 1, 28, 28]
```

### ToTensor

`transforms.ToTensor()` converts images into PyTorch Tensors.

```python
transform = transforms.ToTensor()
```

### DataLoader

Example:

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
```

For Windows learning environment, `num_workers=0` is the safest option.

### CNN model

The model structure:

```text
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
Flatten
Linear -> ReLU -> Linear
```

Shape flow:

```text
[N, 1, 28, 28]
-> [N, 8, 28, 28]
-> [N, 8, 14, 14]
-> [N, 16, 14, 14]
-> [N, 16, 7, 7]
-> [N, 784]
-> [N, 64]
-> [N, 10]
```

### Output

FashionMNIST has 10 classes, so the final output shape is:

```text
[batch_size, 10]
```

Each row contains 10 logits.

### Loss

For this multi-class classification task:

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

Important:

```text
Use raw logits for CrossEntropyLoss.
Do not apply softmax before the loss function.
```

### Accuracy

Prediction:

```python
predicted_classes = torch.argmax(logits, dim=1)
```

Accuracy:

```python
correct = (predicted_classes == labels).sum().item()
accuracy = correct / labels.size(0)
```

### Key takeaway

CNN training still follows the same PyTorch training loop:

```text
model(images)
loss_fn(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Saving and loading models

### Why save models?

After training, the learned parameters are stored in memory.

If we close the program without saving, the trained model is lost.

So we save model weights to a file.

### state_dict

A model's `state_dict` stores its learned parameters.

Example:

```python
torch.save(model.state_dict(), "model.pth")
```

It usually contains:

```text
layer weights
layer biases
```

### Recommended save style

Recommended:

```python
torch.save(model.state_dict(), model_path)
```

Less recommended for long-term projects:

```python
torch.save(model, model_path)
```

Reason:

```text
Saving state_dict is lighter and easier to maintain.
```

### Loading a model

To load a model, first create the same model structure.

```python
loaded_model = FashionMNISTCNN().to(device)
```

Then load the saved weights:

```python
state_dict = torch.load(model_path, map_location=device)
loaded_model.load_state_dict(state_dict)
```

### Inference mode

Before prediction, use:

```python
loaded_model.eval()
```

During prediction, use:

```python
with torch.no_grad():
    logits = loaded_model(x)
```

### Single image prediction

A single FashionMNIST image has shape:

```text
[1, 28, 28]
```

But CNN expects:

```text
[N, C, H, W]
```

So we add a batch dimension:

```python
image_batch = image.unsqueeze(dim=0)
```

Shape change:

```text
[1, 28, 28] -> [1, 1, 28, 28]
```

### Key takeaway

The real project workflow is:

```text
train model
save state_dict
create same model structure
load state_dict
use model.eval()
predict with torch.no_grad()
```