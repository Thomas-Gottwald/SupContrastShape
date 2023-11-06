import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from tqdm import tqdm


class animals10Dataset(Dataset):
    def __init__(self, root="./datasets/", transform=None):
        self.root = root
        self.df_translate = pd.read_csv(root + "animals10_160to300/translate.csv")
        self.df_animals10 = pd.read_csv(root + "animals10_160to300/animals10.csv")
        self.transform = transform

        self.classes = self.df_translate.columns.to_list()

        self.data = []
        self.targets = []

        # load the picked data
        for c in tqdm(self.classes):
            file_path = os.path.join(root, f"animals10_160to300/{c}")
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        # to HWC
        self.data = np.vstack(self.data).reshape(-1, 300, 300, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        # PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def class_to_idx(self):
        class_to_idx = dict()
        for i, c in enumerate(self.classes):
            class_to_idx[c] = i
        return class_to_idx
    