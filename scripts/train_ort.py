import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import WeaklySupervisedPascalVOC
from loggers import LossLogger
from models import ORT


def run(classifier_name, fpn_layer='0', resize='no_resize'):
    if resize == 'no_resize':
        transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])
        dataset = WeaklySupervisedPascalVOC('../data/pascal_voc_2007', transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])
        dataset = WeaklySupervisedPascalVOC('../data/pascal_voc_2007', transform=transform)

    train_dataset, val_dataset = dataset.get_train_val_datasets(val_split=0.2)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),
                            generator=generator)
    if classifier_name == 'vgg16':
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_{resize}.pth'
        loss_logger = LossLogger(f'logs/train_{classifier_name}_classifier_pascal_2007_final_{resize}.csv')
    else:
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.pth'
        loss_logger = LossLogger(
            f'logs/train_{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.csv')

    model = ORT(num_classes=21, model_path=model_path,
                classifier_name=classifier_name, fpn_layer=fpn_layer)
    model.train_model(data_loader_train=train_loader, data_loader_val=val_loader, num_epochs=20,
                      loss_logger=loss_logger)
