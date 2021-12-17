import os
import torch
from torchvision import transforms, datasets
from filelock import FileLock
from torch.utils.data import Subset


#could possible take in subset range as arugments? 
def get_data_loader(sample_idx=[]):
    """Safely downloads data. Returns training/validation set dataloader."""
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader, test_loader = None, None

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):

        dataset = datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms, 
            )

        if len(sample_idx) > 0:
            train_loader = torch.utils.data.DataLoader(Subset(dataset, sample_idx)
                ,
                batch_size=128,
                shuffle=True,
            )


        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("~/data", train=False, transform=mnist_transforms),
            batch_size=128,
            shuffle=True,
        )
    return train_loader, test_loader
