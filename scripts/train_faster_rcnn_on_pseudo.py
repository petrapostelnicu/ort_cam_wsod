import torch
from torch.utils.data import DataLoader

from datasets import PseudoPascalVOC
from models import FasterRCNN
from loggers import LossLogger


def run(model_name, fpn_layer='0', resize='no_resize'):
    if model_name == 'vgg16':
        dataset = PseudoPascalVOC(f'../data/pascal_voc_2007/cam_pseudo_train_{model_name}_{resize}')
        loss_logger = LossLogger(f'logs/train_faster_rcnn_pascal_2007_pseudo_{model_name}_final_{resize}.csv')
        model_path = f'pretrained_models/faster_rcnn_pascal_2007_pseudo_{model_name}_final_{resize}.pth'
    else:
        dataset = PseudoPascalVOC(f'../data/pascal_voc_2007/cam_pseudo_train_{model_name}_{fpn_layer}_{resize}')
        loss_logger = LossLogger(f'logs/train_faster_rcnn_pascal_2007_pseudo_{model_name}_layer{fpn_layer}_final_{resize}.csv')
        model_path = f'pretrained_models/faster_rcnn_pascal_2007_pseudo_{model_name}_layer{fpn_layer}_final_{resize}.pth'
    train_dataset, val_dataset = dataset.get_train_val_datasets(val_split=0.2)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), generator=generator)

    model = FasterRCNN(num_classes=21, model_path=model_path)
    model.train_model(data_loader_train=train_loader, data_loader_val=val_loader, num_epochs=20,
                      loss_logger=loss_logger)
