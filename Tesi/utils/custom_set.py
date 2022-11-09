from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import torch
import PIL
class Custom_set(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform= None, expand = None):
        self.img_labels = pd.read_csv(annotations_file) # file con i nomi dei file e il corrispondente label
        self.img_dir = img_dir # directory delle immagini
        self.transform = transform # augmentation-trasformazioni da applicare
        self.target_transform = target_transform
        self.expand = expand # per espandere le img ir a 3 channel image

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
      # per il cifar 1o corrotto
      if '.npy' in self.img_dir:
        read = np.load(self.img_dir)
        image = read[idx]
      # per le img salvate nelle folder
      else:
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = PIL.Image.open(img_path)
      label = self.img_labels.iloc[idx, 1]
      if self.transform: 
        image = self.transform(image)
        if image.shape[0] == 1 and self.expand == True:
          image = torch.cat([image, image, image], dim=0)
        else:
          pass
      if self.target_transform:
        label = self.target_transform(label)
      return image, label

class Custom_set_imagenet(Dataset):
    def __init__(self, dataset, transform=None, target_transform= None, expand = None):
        self.dataset = dataset
        self.transform = transform # augmentation-trasformazioni da applicare
        self.target_transform = target_transform
        self.expand = expand # per espandere le img ir a 3 channel image

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
      image = self.dataset[idx]['image']
      label = self.dataset[idx]['label']
      if self.transform:
        image = self.transform(image)
        if image.shape[0] == 1 and self.expand == True:
          image = torch.cat([image, image, image], dim=0)
        else:
          pass
      if self.target_transform:
        label = self.target_transform(label)
      return image, label
