
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
from random import shuffle

import torch
import glob as glob


class patch_data(Dataset):

    def __init__(self, opt, split, transform=None):
        """

        :param opt: Command line option(/defaults)
        :param split: train | val | test
        :param transform: NotImplemented
        """
        data_path = osp.join(opt.root_path, split + '_patches')

        self.CON_files = glob.glob(data_path+'/0/*.npy')
        len_CON = len(self.CON_files)
        self.CON_labels = np.zeros(len_CON)

        self.ASD_files = glob.glob(data_path+'/1/*.npy')
        len_ASD = len(self.ASD_files)
        self.ASD_labels = np.ones(len_ASD)

        self.total_len = len_CON + len_ASD

        self.all_files = np.array(self.CON_files + self.ASD_files)  # converting a list np.array for indexing later
        all_labels = np.concatenate((self.CON_labels, self.ASD_labels))

        # Shuffle the order
        ids = np.arange(self.total_len)
        shuffle(ids)
        self.all_files = list(self.all_files[ids])  # indexing possible because of np.array conversion
        all_labels = all_labels[ids]

        self.labels = torch.from_numpy(all_labels).type(torch.LongTensor)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):

        data = np.load(self.all_files[item])  # pick up a filename

        self.data = torch.from_numpy(np.expand_dims(data, axis=0)).type(torch.FloatTensor)
        return self.data, self.labels[item]


def get_data_set(opt, split, transform=None):

    data_set = patch_data(opt,
                        split=split,
                        transform=transform)
    return data_set
