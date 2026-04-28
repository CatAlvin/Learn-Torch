import sys
import platform

import torch
import numpy as np
import pandas as pd
import matplotlib
import sklearn


def main():
    print("=" * 60)
    print("PyTorch Learning Lab - Environment Check")
    print("=" * 60)

    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")

    print("-" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")

    print("-" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU name: {torch.cuda.get_device_name(0)}")

        x = torch.rand(3, 3).cuda()
        print("-" * 60)
        print("A random tensor on GPU:")
        print(x)
        print(f"Tensor device: {x.device}")
    else:
        print("CUDA is not available. PyTorch will run on CPU.")
        x = torch.rand(3, 3)
        print("-" * 60)
        print("A random tensor on CPU:")
        print(x)
        print(f"Tensor device: {x.device}")

    print("=" * 60)
    print("Environment check finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()