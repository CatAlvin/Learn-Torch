# Learn-Torch

A structured PyTorch learning project from Tensor basics to CNN image classification.

This repository records my step-by-step PyTorch learning process, including Tensor operations, autograd, neural network modules, Dataset/DataLoader, training loops, classification, CNNs, model saving/loading, and experiment tracking.

## Environment

- Python: 3.14.0
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- OS: Windows 11

## Main Topics

| Lesson | Topic |
|---|---|
| lesson_00_setup | Environment setup |
| lesson_01_tensor_basics | Tensor creation, shape, dtype, device, reshape |
| lesson_02_tensor_operations | Tensor operations, broadcasting, matrix multiplication |
| lesson_03_autograd | Autograd and manual gradient descent |
| lesson_04_linear_regression | Linear regression with PyTorch standard components |
| lesson_05_nn_module | Custom models with `nn.Module` |
| lesson_06_dataset_dataloader | Dataset and DataLoader |
| lesson_07_training_loop | Full training and validation loop |
| lesson_08_classification | Classification and CrossEntropyLoss |
| lesson_09_cnn_intro | CNN basics, FashionMNIST training, save/load model |
| lesson_10_final_project | Mini FashionMNIST CNN project |

## Project Structure

```text
Learn-Torch/
├─ lessons/
│  ├─ lesson_00_setup/
│  ├─ lesson_01_tensor_basics/
│  ├─ lesson_02_tensor_operations/
│  ├─ lesson_03_autograd/
│  ├─ lesson_04_linear_regression/
│  ├─ lesson_05_nn_module/
│  ├─ lesson_06_dataset_dataloader/
│  ├─ lesson_07_training_loop/
│  ├─ lesson_08_classification/
│  ├─ lesson_09_cnn_intro/
│  └─ lesson_10_final_project/
│
├─ src/
│  ├─ models.py
│  ├─ data_utils.py
│  ├─ train_utils.py
│  └─ utils.py
│
├─ data/
│  ├─ raw/
│  └─ processed/
│
├─ outputs/
│  ├─ models/
│  ├─ figures/
│  └─ logs/
│
├─ notebooks/
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Setup

Create virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install CUDA PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install common packages:

```powershell
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel tqdm rich
```

Check environment:

```powershell
python lessons\lesson_00_setup\00_check_env.py
```

Expected result:

```text
CUDA available: True
PyTorch version: 2.11.0+cu128
CUDA version used by PyTorch: 12.8
```

## FashionMNIST Mini Project

The final mini project trains a CNN classifier on FashionMNIST.

Classes:

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

Main project files:

| File | Purpose |
|---|---|
| `src/models.py` | CNN model definitions |
| `src/data_utils.py` | FashionMNIST dataset and DataLoader utilities |
| `src/train_utils.py` | Training, evaluation, prediction utilities |
| `01_train_project.py` | Train baseline CNN |
| `02_predict_project.py` | Load baseline model and predict |
| `03_train_improved_cnn.py` | Train improved CNN with BatchNorm, Dropout, AdamW |
| `04_predict_improved_cnn.py` | Predict with improved CNN |
| `05_evaluate_improved_cnn.py` | Evaluate with confusion matrix and error samples |
| `06_train_with_scheduler_early_stopping.py` | Train with scheduler and early stopping |
| `07_train_with_augmentation.py` | Train with data augmentation |
| `08_visualize_augmentation.py` | Visualize augmented images |
| `09_compare_experiments.py` | Compare experiment results |

## Run Final Project

Train baseline CNN:

```powershell
python lessons\lesson_10_final_project\01_train_project.py
```

Predict with baseline CNN:

```powershell
python lessons\lesson_10_final_project\02_predict_project.py
```

Train improved CNN:

```powershell
python lessons\lesson_10_final_project\03_train_improved_cnn.py
```

Predict with improved CNN:

```powershell
python lessons\lesson_10_final_project\04_predict_improved_cnn.py
```

Evaluate improved CNN:

```powershell
python lessons\lesson_10_final_project\05_evaluate_improved_cnn.py
```

Train with scheduler and early stopping:

```powershell
python lessons\lesson_10_final_project\06_train_with_scheduler_early_stopping.py
```

Train with data augmentation:

```powershell
python lessons\lesson_10_final_project\07_train_with_augmentation.py
```

Visualize augmentation:

```powershell
python lessons\lesson_10_final_project\08_visualize_augmentation.py
```

Compare experiments:

```powershell
python lessons\lesson_10_final_project\09_compare_experiments.py
```

## Key PyTorch Training Loop

```python
for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = loss_fn(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Key Lessons Learned

- Tensor shape, dtype, and device are the first things to check when debugging.
- `requires_grad=True` allows PyTorch to track computation and calculate gradients.
- `nn.Module` is the base class for custom PyTorch models.
- `Dataset` defines how to access samples.
- `DataLoader` creates mini-batches.
- `CrossEntropyLoss` expects raw logits and `torch.long` labels.
- CNNs process image tensors in `[N, C, H, W]` format.
- `model.train()` and `model.eval()` control training/evaluation behavior.
- `torch.no_grad()` should be used during validation and prediction.
- `state_dict` is the recommended way to save model weights.
- BatchNorm, Dropout, AdamW, scheduler, early stopping, and data augmentation are useful training improvements.
- Experiment logs help compare model changes objectively.

## Git Notes

The following folders are ignored by Git:

```text
data/raw/
data/processed/
outputs/models/
outputs/figures/
outputs/logs/
```

This avoids uploading large datasets, model weights, generated figures, and logs.

Only source code and lesson notes are tracked.