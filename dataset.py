from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import os.path as osp
import numpy as np


def my_numpy_loader(filename):
    return np.load(filename)


def get_data_set(opt, split, transform=ToTensor()):

    data_set = ImageFolder(root=osp.join(opt.root_path, split+'_patches'), loader=my_numpy_loader, transform=transform)
    return data_set

