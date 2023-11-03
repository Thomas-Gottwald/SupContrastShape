import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io


class animals10Dataset(Dataset):
    def __init__(self, path="./datasets/animals10/"):
        self.path = path
        self.df_translate = pd.read_csv(path + "translate.csv")
        self.df_animals10 = pd.read_csv(path + "animals10.csv")

    def __len__(self):
        return len(self.df_animals10)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_loc = os.path.join(self.path, self.df_animals10.iloc[idx,0])

        image = torch.Tensor(io.imread(img_loc)/255.0)
        label = self.df_animals10.iloc[idx,1]

        return image, label


def create_csv_files(path="./datasets/animals10/"):
    df_translate = pd.DataFrame.from_dict(
        {"cane": ["dog"], "cavallo": ["horse"], "elefante": ["elephant"], "farfalla": ["butterfly"], "gallina": ["chicken"],
        "gatto": ["cat"], "mucca": ["cow"], "pecora": ["sheep"], "ragno": ["spider"], "scoiattolo": ["squirrel"]})

    images = []
    labels = []
    for i, c in enumerate(df_translate):
        img_paths = glob.glob(path + f"raw-img/{c}/*")

        for img_path in img_paths:
            img = img_path.replace(path, '').replace('\\', '/')
            images.append(img)
            labels.append(i)

    df_animals10 = pd.DataFrame.from_dict({'image': images, 'label': labels})

    df_translate.to_csv(path_or_buf = path + "translate.csv", index=False)
    df_animals10.to_csv(path_or_buf = path + "animals10.csv", index=False)