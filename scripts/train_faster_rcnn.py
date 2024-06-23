import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import PascalVOC
from loggers import LossLogger
from models import FasterRCNN


def run(resize='no_resize'):
    if resize == 'no_resize':
        transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])
        dataset = PascalVOC('../data/pascal_voc_2007', transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])
        dataset = PascalVOC('../data/pascal_voc_2007', transform=transform)
    train_dataset, val_dataset = dataset.get_train_val_datasets(val_split=0.2)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),
                            generator=generator)

    loss_logger = LossLogger(f'logs/train_faster_rcnn_pascal_2007_final_{resize}.csv')

    model = FasterRCNN(num_classes=21, model_path=f'pretrained_models/faster_rcnn_pascal_2007_final_{resize}.pth')
    model.train_model(data_loader_train=train_loader, data_loader_val=val_loader, num_epochs=20,
                      loss_logger=loss_logger)
