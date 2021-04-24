"""MNIST DataModule"""
import argparse
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split

from torchvision import transforms

from cifar_cnn.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"


class CIFAR(BaseDataModule):
  def __init__(self, args: argparse.Namespace) -> None:
    super().__init__(args)
    self.data_dir = DOWNLOADED_DATA_DIRNAME
    self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    self.dims = (3, 32, 32)  # dims are returned when calling `.size()` on this object.
    self.output_dims = (1,)
    self.mapping = list(range(10))

  def prepare_data(self, *args, **kwargs) -> None:
    CIFAR10(self.data_dir, train=True, download=True)
    CIFAR10(self.data_dir, train=False, download=True)

  def setup(self, stage=None) -> None:
    """Split into train, val, test, and set dims."""
    """Split into train, val, test, and set dims."""
    cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
        
    self.data_train, self.data_val = random_split(cifar_full, [45000, 5000])
    self.data_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

if __name__ == "__main__":
    load_and_print_info(CIFAR)
