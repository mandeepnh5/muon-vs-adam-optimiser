"""muon_bench: Muon optimizer from scratch + CIFAR-10 time-to-accuracy benchmark."""

from .muon import Muon, newton_schulz_orthogonalize, split_muon_params
from .resnet9 import ResNet9, build_resnet9
from .data import GPUCifar10, load_cifar10, CIFAR10_MEAN, CIFAR10_STD
from .utils import set_seed, CudaBlockTimer, ModelEMA, env_info, save_json

__version__ = "1.0.0"

__all__ = [
    "Muon",
    "newton_schulz_orthogonalize",
    "split_muon_params",
    "ResNet9",
    "build_resnet9",
    "GPUCifar10",
    "load_cifar10",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "set_seed",
    "CudaBlockTimer",
    "ModelEMA",
    "env_info",
    "save_json",
]
