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