from typing import Callable, Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from torchvision.datasets import VOCDetection

VOC_BBOX_LABEL_NAMES = (
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


def indices_to_one_hot(num_classes, class_indices):
    # Transform target classes array to one hot array
    target = torch.zeros(num_classes, dtype=torch.float32)

    target[class_indices] = 1
    return target


class WeaklySupervisedPascalVOC(Dataset):
    def __init__(self, root: str, year: str = '2007', image_set: str = 'trainval',
                 transform: Optional[Callable] = None):
        # Weakly Supervised dataset version of PASCAL VOC 2007
        self.voc_dataset = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        convert_tensor = transforms.ToTensor()
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = convert_tensor(image)
        annotations = target['annotation']
        labels = []
        for obj in annotations['object']:
            labels.append(VOC_BBOX_LABEL_NAMES.index(obj['name']))
        target = {
            'labels': torch.as_tensor(indices_to_one_hot(21, labels), dtype=torch.int64),
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
