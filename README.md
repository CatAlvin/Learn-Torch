# PyTorch Learning Lab

This repository records my step-by-step PyTorch learning journey.

## Goals

- Learn PyTorch from fundamentals to practical projects
- Understand Tensor, autograd, neural networks, Dataset, DataLoader, and training loops
- Build small but complete machine learning and deep learning projects
- Keep all learning code organized and GitHub-ready

## Environment

- Python: 3.14.0
- Virtual environment: `.venv`
- Deep learning framework: PyTorch
- Main libraries: NumPy, Pandas, Matplotlib, Scikit-learn, Jupyter

## Project Structure

```text
pytorch-learning-lab/
├─ lessons/
├─ src/
├─ data/
├─ outputs/
└─ notebooks/
```

## Setup

### Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### Install CUDA PyTorch

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Install common packages

```powershell
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel tqdm rich
```

### Check environment

```powershell
python lessons\lesson_00_setup\00_check_env.py
```

## Lesson Map

| Lesson | Topic |
|---|---|
| `lesson_00_setup` | Environment setup |
| `lesson_01_tensor_basics` | Tensor basics |
| `lesson_02_tensor_operations` | Tensor operations |
| `lesson_03_autograd` | Automatic differentiation |
| `lesson_04_linear_regression` | Linear regression from scratch |
| `lesson_05_nn_module` | Building models with `nn.Module` |
| `lesson_06_dataset_dataloader` | Dataset and DataLoader |
| `lesson_07_training_loop` | Full training loop |
| `lesson_08_classification` | Classification tasks |
| `lesson_09_cnn_intro` | CNN introduction |
| `lesson_10_final_project` | Final project |

---

# Lesson 00 - Environment Setup

## What we did

In this lesson, we created a clean PyTorch learning environment.

## Key ideas

- A virtual environment keeps project dependencies isolated.
- PyTorch can run on CPU or GPU.
- CUDA-enabled PyTorch requires an NVIDIA GPU and compatible driver.
- For normal PyTorch learning, installing the CUDA PyTorch wheel is usually enough.
- We use a structured folder layout so every lesson has a clear place.

## Important commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel tqdm rich
python lessons\lesson_00_setup\00_check_env.py
```

## Check result

The most important line is:

```text
CUDA available: True
```

If it is true, GPU acceleration is ready.

If it is false, PyTorch still works, but it will run on CPU.