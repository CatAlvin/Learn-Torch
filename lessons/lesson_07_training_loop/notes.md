# Lesson 07 - Full Training and Validation Loop

## Core idea

A real training workflow usually includes:

- Dataset
- DataLoader
- training loop
- validation loop
- loss curves
- prediction on new data

## Why validation set?

Training loss only tells us how well the model fits training data.

Validation loss tells us how well the model performs on unseen data.

## Train/validation split

We can use:

```python
from torch.utils.data import random_split

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
```

Example:

```text
20 samples
80% training -> 16 samples
20% validation -> 4 samples
```

## DataLoaders

Training DataLoader usually uses:

```python
shuffle=True
```

Validation DataLoader usually uses:

```python
shuffle=False
```

Common rule:

```text
train_loader: shuffle=True
val_loader  : shuffle=False
test_loader : shuffle=False
```

## train_one_epoch

A training function updates model parameters.

```python
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        pred = model(batch_features)
        loss = loss_fn(pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## evaluate

An evaluation function does not update model parameters.

```python
def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            pred = model(batch_features)
            loss = loss_fn(pred, batch_labels)
```

## model.train and model.eval

Training mode:

```python
model.train()
```

Evaluation mode:

```python
model.eval()
```

## torch.no_grad

During validation or prediction, use:

```python
with torch.no_grad():
    pred = model(x)
```

This saves memory and avoids tracking gradients.

## Train loss vs validation loss

Good situation:

```text
train loss decreases
validation loss decreases
```

Possible overfitting:

```text
train loss decreases
validation loss increases
```

Possible underfitting:

```text
train loss remains high
validation loss remains high
```

## Preprocessing rule

Training data and new prediction data must use the same preprocessing.

Example:

```python
normalized_train = (train_features - feature_mean) / feature_std
normalized_new = (new_student - feature_mean) / feature_std
```

## Key takeaway

A complete training workflow needs both training and validation, not only training loss.