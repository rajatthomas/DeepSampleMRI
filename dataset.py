
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
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
        self.len_CON = len(self.CON_files)
        self.CON_labels = np.zeros(self.len_CON)

        self.ASD_files = glob.glob(data_path+'/1/*.npy')
        self.len_ASD = len(self.ASD_files)
        self.ASD_labels = np.ones(self.len_ASD)

        self.all_files  = self.CON_files + self.ASD_files
        all_labels = np.concatenate((self.CON_labels, self.ASD_labels))

        self.labels = torch.from_numpy(all_labels).type(torch.LongTensor)

    def __len__(self):
        return self.len_CON + self.len_ASD

    def __getitem__(self, item):

        data = np.load(self.all_files[item])

        self.data = torch.from_numpy(np.expand_dims(data, axis=1)).type(torch.FloatTensor)
        return self.data[item], self.labels[item]


def get_data_set(opt, split, transform=None):

    data_set = patch_data(opt,
                        split=split,
                        transform=transform)
    return data_set
