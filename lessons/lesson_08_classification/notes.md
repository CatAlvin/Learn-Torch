# Lesson 08 - Classification Basics

## Core idea

Regression predicts continuous values.

Classification predicts discrete classes.

Examples:

```text
Regression: predict score, price, temperature
Classification: predict pass/fail, cat/dog, spam/not spam
```

## Binary classification with CrossEntropyLoss

In this lesson:

```text
0 = fail
1 = pass
```

The model outputs 2 class scores.

```text
logits shape: [batch_size, 2]
labels shape: [batch_size]
labels dtype: torch.long
```

## Logits

Logits are raw class scores from the model.

```python
logits = model(x)
```

They are not probabilities yet.

## Softmax

Softmax converts logits into probabilities.

```python
probabilities = torch.softmax(logits, dim=1)
```

Use `dim=1` because class scores are on dimension 1.

Shape:

```text
logits shape        : [batch_size, num_classes]
probabilities shape : [batch_size, num_classes]
```

## Argmax

`argmax` finds the predicted class.

```python
predicted_classes = torch.argmax(logits, dim=1)
```

Example:

```text
logits = [0.2, 1.8]
predicted class = 1
```

## CrossEntropyLoss

For multi-class classification, use:

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

Important:

```text
Do not apply softmax before CrossEntropyLoss.
CrossEntropyLoss expects raw logits.
```

## Label format

Correct label format:

```python
labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
```

Wrong for CrossEntropyLoss:

```text
labels shape [batch_size, 1]
one-hot labels
float labels
```

## Accuracy

Accuracy measures the fraction of correct predictions.

```python
predicted_classes = torch.argmax(logits, dim=1)
correct = (predicted_classes == labels).sum().item()
accuracy = correct / labels.size(0)
```

## Training loop

Classification training still follows the same structure:

```python
logits = model(batch_features)
loss = loss_fn(logits, batch_labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Prediction

During prediction:

```python
model.eval()

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
```

## Key takeaway

For CrossEntropyLoss:

```text
model output: raw logits
labels: class indices with dtype torch.long
```