import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from tqdm import tqdm


class animals10Dataset(Dataset):
    def __init__(self, root="./datasets/animals10_160to300/", transform=None):
        self.root = root
        self.df_translate = pd.read_csv(root + "translate.csv")
        self.df_animals10 = pd.read_csv(root + "animals10_160to300.csv")
        self.transform = transform

        self.classes = self.df_translate.columns.to_list()

        self.data = []
        self.targets = []

        # load the picked data
        for c in tqdm(self.classes):
            file_path = os.path.join(root, f"{c}")
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 300, 300)
        self.data = self.data.transpose((0, 2, 3, 1))# to HWC

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def class_to_idx(self):
        class_to_idx = dict()
        for i, c in enumerate(self.classes):
            class_to_idx[c] = i
        return class_to_idx
    