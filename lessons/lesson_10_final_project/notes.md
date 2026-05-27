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

## Part 3 - Model evaluation

### Goal

In this part, we evaluate the improved CNN more carefully.

We check:

```text
overall accuracy
per-class accuracy
confusion matrix
classification report
misclassified examples
```

### Overall accuracy

Overall accuracy measures total correct predictions.

```python
accuracy = correct_predictions / total_samples
```

### Per-class accuracy

Per-class accuracy checks each class separately.

Example:

```text
T-shirt/top accuracy
Trouser accuracy
Shirt accuracy
Sneaker accuracy
```

This helps us find weak classes.

### Confusion matrix

A confusion matrix shows how classes are confused.

```text
Rows    = true labels
Columns = predicted labels
```

Diagonal values mean correct predictions.

Off-diagonal values mean wrong predictions.

### Classification report

A classification report includes:

```text
precision
recall
f1-score
support
```

Basic meaning:

```text
precision: how reliable predictions of this class are
recall: how many real samples of this class were found
f1-score: balance between precision and recall
support: number of samples in this class
```

### Misclassified examples

Saving wrong predictions helps us inspect model errors visually.

Example:

```text
True: Shirt
Pred: T-shirt/top
Confidence: 0.71
```

### Key takeaway

A good evaluation does not only report one accuracy number.

It should also answer:

```text
Which classes are weak?
Which classes are confused?
What do wrong examples look like?
Is the model confidently wrong?
```

## Part 4 - Scheduler and Early Stopping

### Goal

In this part, we improve the training process.

We add:

```text
Learning rate scheduler
Early stopping
Best checkpoint based on test accuracy
Learning rate curve
```

### Why scheduler?

A fixed learning rate may not be ideal for the whole training process.

Common idea:

```text
Early training: larger learning rate for faster learning
Later training: smaller learning rate for fine adjustment
```

### ReduceLROnPlateau

We use:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)
```

Meaning:

```text
mode="min": monitored value should decrease
factor=0.5: reduce learning rate by half
patience=2: wait 2 epochs before reducing learning rate
```

After each epoch:

```python
scheduler.step(test_loss)
```

### Early stopping

Early stopping stops training when performance does not improve for several epochs.

Example:

```text
If test accuracy does not improve for 6 epochs, stop training.
```

Basic logic:

```python
if test_acc > best_test_acc:
    best_test_acc = test_acc
    epochs_without_improvement = 0
    save_checkpoint()
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= patience:
    stop_training()
```

### Why save best checkpoint?

The last epoch is not always the best epoch.

A model may overfit after reaching its best validation or test performance.

So we save the model whenever the monitored metric improves.

### Key takeaway

A stronger training workflow should include:

```text
optimizer
scheduler
early stopping
best checkpoint
training curves
learning rate curve
```

## Part 5 - Data Augmentation

### Goal

In this part, we add train-time data augmentation.

Data augmentation helps improve generalization.

### What is data augmentation?

Data augmentation randomly changes training images.

Examples:

```text
random horizontal flip
small rotation
small translation
```

The goal is to help the model learn more robust features.

### Train transform vs test transform

Important rule:

```text
Use augmentation for training data.
Do not use random augmentation for test data.
```

Reason:

```text
Training augmentation improves robustness.
Test data should remain stable and fair for evaluation.
```

### Transform pipeline

Training transform:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=10)
transforms.RandomAffine(
    degrees=0,
    translate=(0.08, 0.08),
)
transforms.ToTensor()
transforms.Normalize(mean=(0.2860,), std=(0.3530,))
```

Test transform:

```python
transforms.ToTensor()
transforms.Normalize(mean=(0.2860,), std=(0.3530,))
```

### Why train accuracy may decrease

With augmentation, training images become harder.

So train accuracy may decrease.

But if test accuracy improves, augmentation is useful.

### Reasonable augmentation

For FashionMNIST, reasonable augmentation includes:

```text
small rotation
small translation
horizontal flip
```

Unreasonable augmentation may include:

```text
vertical flip
90-degree rotation
strong distortion
large crop
```

### Key takeaway

Data augmentation should create realistic variations.

It should not change the meaning of the image.

## Part 6 - Visualizing data augmentation

### Goal

Before trusting data augmentation, we should visualize augmented samples.

### Why visualize augmentation?

Data augmentation can help generalization, but bad augmentation can hurt the model.

Important question:

```text
Does the augmented image still keep the correct label?
```

### Original transform

```python
original_transform = build_fashion_mnist_transform(
    use_normalization=False,
    use_augmentation=False,
)
```

### Augmented transform

```python
augmented_transform = build_fashion_mnist_transform(
    use_normalization=True,
    use_augmentation=True,
)
```

### Why unnormalize before visualization?

Normalized images are not in the original pixel range.

To display them properly:

```python
image = image * std + mean
image = image.clamp(0, 1)
```

### What to check

When viewing augmented samples, check:

```text
Are images still recognizable?
Are labels still correct?
Are rotations too strong?
Are translations too large?
Does horizontal flip make sense for this task?
```

### Important rule

Good augmentation should create realistic variations.

Bad augmentation can create images outside the real data distribution.

### Key takeaway

Never blindly trust augmentation.

Always visualize augmented samples.