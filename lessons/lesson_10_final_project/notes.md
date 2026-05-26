# Lesson 10 - Final Mini Project

## Core idea

This lesson turns previous lesson code into a small project structure.

The goal is to separate:

- model definition
- data loading
- training utilities
- training script
- prediction script

## Project structure

```text
src/models.py
src/data_utils.py
src/train_utils.py

lessons/lesson_10_final_project/01_train_project.py
lessons/lesson_10_final_project/02_predict_project.py
```

## models.py

`src/models.py` stores model classes.

Example:

```python
class FashionMNISTCNN(torch.nn.Module):
    ...
```

## data_utils.py

`src/data_utils.py` stores dataset and DataLoader functions.

Example:

```python
train_loader, test_loader, train_dataset_full, test_dataset_full = (
    get_fashion_mnist_loaders()
)
```

## train_utils.py

`src/train_utils.py` stores reusable training functions:

```text
calculate_accuracy
train_one_epoch
evaluate
predict_image
```

## Training script

`01_train_project.py` does:

```text
load data
create model
train model
evaluate model
save model state_dict
save loss and accuracy curves
```

## Prediction script

`02_predict_project.py` does:

```text
load test dataset
create same model structure
load saved state_dict
predict test images
print predicted classes and confidence
```

## Important rule

`state_dict` saves weights, not model structure.

So prediction code must create the same model class before loading weights.

## Key takeaway

A real PyTorch project should not put everything into one file.

A cleaner structure is:

```text
src/ for reusable code
lessons/ or scripts/ for runnable experiments
data/ for datasets
outputs/ for models, figures, and logs
```


## Part 2 - Improved CNN

### Goal

In this part, we improve the FashionMNIST CNN project.

Main improvements:

```text
Normalize input images
Use a deeper CNN
Add BatchNorm
Add Dropout
Use AdamW
Save best checkpoint
```

### Normalize

For FashionMNIST, we can use:

```python
transforms.Normalize(
    mean=(0.2860,),
    std=(0.3530,),
)
```

Meaning:

```text
x_normalized = (x - mean) / std
```

### BatchNorm

BatchNorm helps stabilize training.

In CNNs, we usually use:

```python
torch.nn.BatchNorm2d(num_features=16)
```

Common pattern:

```text
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
```

### Dropout

Dropout randomly disables some neurons during training.

Example:

```python
torch.nn.Dropout(p=0.3)
```

Meaning:

```text
During training, 30% of the activations are randomly set to zero.
During evaluation, Dropout is disabled.
```

This helps reduce overfitting.

### AdamW

AdamW is an optimizer often used in deep learning projects.

Example:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)
```

`weight_decay` is a regularization technique.

It can help reduce overfitting.

### Improved CNN structure

```text
Input: [N, 1, 28, 28]

Conv2d(1, 16)
BatchNorm2d(16)
ReLU
MaxPool2d

Conv2d(16, 32)
BatchNorm2d(32)
ReLU
MaxPool2d

Conv2d(32, 64)
BatchNorm2d(64)
ReLU

Flatten
Linear(64 * 7 * 7, 128)
ReLU
Dropout(0.3)
Linear(128, 10)

Output: [N, 10]
```

### Checkpoint

A checkpoint can store more than model weights.

Example:

```python
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_test_acc": best_test_acc,
    },
    checkpoint_path,
)
```

### Loading checkpoint

```python
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Best model

During training, save the model whenever test accuracy improves.

```python
if test_acc > best_test_acc:
    best_test_acc = test_acc
    torch.save(checkpoint, best_model_path)
```

### Key takeaway

A stronger training workflow includes:

```text
better model
better preprocessing
regularization
best checkpoint saving
separate train and predict scripts
```