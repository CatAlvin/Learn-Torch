# Lesson 06 - Dataset and DataLoader

## Core idea

`Dataset` defines how to access data.

`DataLoader` defines how to load data in batches.

## Why Dataset and DataLoader?

In small examples, we can write:

```python
x = torch.tensor(...)
y = torch.tensor(...)
```

But in real projects, data may come from:

- CSV files
- image folders
- text files
- databases

So we use:

```text
Dataset   -> stores and returns samples
DataLoader -> creates batches and shuffles data
```

## Custom Dataset

A custom Dataset usually inherits from:

```python
torch.utils.data.Dataset
```

Example:

```python
class StudentScoreDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
```

## __len__

`__len__` returns the number of samples.

```python
len(dataset)
```

## __getitem__

`__getitem__` returns one sample.

```python
x, y = dataset[0]
```

Usually it returns:

```text
features, label
```

## DataLoader

A DataLoader creates mini-batches.

```python
dataloader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
)
```

Meaning:

```text
batch_size=4 -> each batch has 4 samples
shuffle=True -> shuffle data before each epoch
```

## Batch shape

If each sample has 4 features and batch size is 4:

```text
batch_features shape: [4, 4]
batch_labels shape  : [4, 1]
```

## Training with DataLoader

```python
for batch_features, batch_labels in dataloader:
    batch_features = batch_features.to(device)
    batch_labels = batch_labels.to(device)

    pred = model(batch_features)
    loss = loss_fn(pred, batch_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Why move batch to device?

In real projects, the full dataset may be too large for GPU memory.

So we usually move only one batch to GPU at a time.

```python
batch_features = batch_features.to(device)
batch_labels = batch_labels.to(device)
```

## Key takeaway

`Dataset` answers:

```text
How many samples do I have?
What is sample i?
```

`DataLoader` answers:

```text
How do I create batches?
Should I shuffle the data?
```