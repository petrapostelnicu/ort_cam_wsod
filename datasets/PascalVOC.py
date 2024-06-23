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


class PascalVOC(Dataset):
    def __init__(self, root: str, year: str = '2007', image_set: str = 'trainval',
                 transform: Optional[Callable] = None):
        self.voc_dataset = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.voc_dataset)

    def resize_boxes(self, boxes, original_size, new_size):
        # If the images are resized, we need to resize the ground truth bounding boxes in the annotations
        orig_width, orig_height = original_size
        new_width, new_height = new_size
        ratios = [new_width / orig_width, new_height / orig_height]
        new_boxes = []
        for box in boxes:
            resized_box = [
                box[0] * ratios[0],
                box[1] * ratios[1],
                box[2] * ratios[0],
                box[3] * ratios[1]
            ]
            new_boxes.append(resized_box)
        return new_boxes

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        original_size = image.size
        convert_tensor = transforms.ToTensor()
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = convert_tensor(image)

        new_size = (img.shape[-1], img.shape[-2])

        annotations = target['annotation']
        boxes = []
        labels = []
        for obj in annotations['object']:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            xmax = float(bbox['xmax'])
            ymin = float(bbox['ymin'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_BBOX_LABEL_NAMES.index(obj['name']))

        # Resize bounding boxes
        if transforms is not None:
            boxes = self.resize_boxes(boxes, original_size, new_size)

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

    def categorize_images(self):
        # Method to split dataset into localization, multi instance and multi class
        localization = []
        multi_instance = []
        multi_class = []

        for idx in range(len(self.voc_dataset)):
            _, target = self[idx]
            labels = target['labels']
            unique_labels = set(labels.numpy())

            if len(labels) == 1:
                localization.append(idx)
            elif len(unique_labels) == 1:
                multi_instance.append(idx)
            else:
                multi_class.append(idx)

        return localization, multi_instance, multi_class

    def get_small_large_localization(self, localization_dataset, threshold_percentage=0.2):
        # Method to split localization into small and large
        small_objects = []
        large_objects = []

        for idx in localization_dataset.indices:
            img, target = self[idx]
            box = target['boxes'][0]
            xmin, ymin, xmax, ymax = box
            box_area = (xmax - xmin) * (ymax - ymin)

            # Calculate the area of the image
            image_area = img.shape[-1] * img.shape[-2]

            # Calculate the threshold based on the image area
            threshold_area = threshold_percentage * image_area

            if box_area < threshold_area:
                small_objects.append(idx)
            else:
                large_objects.append(idx)

        small_objects_dataset = torch.utils.data.Subset(self, small_objects)
        large_objects_dataset = torch.utils.data.Subset(self, large_objects)

        return small_objects_dataset, large_objects_dataset

    def get_split_datasets(self):
        # Method to get the data splits
        localization, multi_instance, multi_class = self.categorize_images()

        localization_dataset = torch.utils.data.Subset(self, localization)
        multi_instance_dataset = torch.utils.data.Subset(self, multi_instance)
        multi_class_dataset = torch.utils.data.Subset(self, multi_class)

        small_localization_dataset, large_localization_dataset = self.get_small_large_localization(localization_dataset)

        return localization_dataset, small_localization_dataset, large_localization_dataset, multi_instance_dataset, multi_class_dataset
