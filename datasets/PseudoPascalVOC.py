import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, random_split


class PseudoPascalVOC(Dataset):
    def __init__(self, root, transform=None):
        # PASCAL VOC 2007 with pseudo label created by ORT
        self.root = root
        self.image_path = os.path.join(self.root, "images")
        self.label_path = os.path.join(self.root, "labels")
        self.transform = transform

        self.imgs = sorted(os.listdir(self.image_path), key=lambda x: int(os.path.splitext(x)[0]))
        self.labels = sorted(os.listdir(self.label_path), key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load images and bounding boxes
        img_path = os.path.join(self.image_path, self.imgs[idx])
        label_path = os.path.join(self.label_path, self.labels[idx])
        image = Image.open(img_path)
        convert_tensor = transforms.ToTensor()
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = convert_tensor(image)

        # Read annotations
        box_list = pd.read_csv(label_path)
        boxes = box_list[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = box_list['label'].values

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Create a dictionary for target
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.as_tensor(idx)
        }

        return img, target

    def get_train_val_datasets(self, val_split=0.2):
        train_size = int((1 - val_split) * len(self))
        val_size = len(self) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def get_reduced(self):
        # Method to get 100 random images
        # size = int(0.01 * len(self))
        size = int(100)
        rest = len(self) - size
        generator = torch.Generator().manual_seed(42)
        dataset, _ = random_split(self, [size, rest], generator=generator)
        return dataset
