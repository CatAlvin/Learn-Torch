from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_fashion_mnist_transform(use_normalization=False):
    transform_list = [
        transforms.ToTensor(),
    ]

    if use_normalization:
        transform_list.append(
            transforms.Normalize(
                mean=(0.2860,),
                std=(0.3530,),
            )
        )

    return transforms.Compose(transform_list)


def get_fashion_mnist_datasets(
    root="data/raw",
    train_subset_size=5000,
    test_subset_size=1000,
    use_normalization=False,
):
    transform = build_fashion_mnist_transform(
        use_normalization=use_normalization,
    )

    train_dataset_full = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset_full = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    if train_subset_size is not None:
        train_dataset = Subset(
            train_dataset_full,
            list(range(train_subset_size)),
        )
    else:
        train_dataset = train_dataset_full

    if test_subset_size is not None:
        test_dataset = Subset(
            test_dataset_full,
            list(range(test_subset_size)),
        )
    else:
        test_dataset = test_dataset_full

    return train_dataset, test_dataset, train_dataset_full, test_dataset_full


def get_fashion_mnist_loaders(
    batch_size=64,
    train_subset_size=5000,
    test_subset_size=1000,
    num_workers=0,
    use_normalization=False,
):
    train_dataset, test_dataset, train_dataset_full, test_dataset_full = (
        get_fashion_mnist_datasets(
            train_subset_size=train_subset_size,
            test_subset_size=test_subset_size,
            use_normalization=use_normalization,
        )
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, train_dataset_full, test_dataset_full