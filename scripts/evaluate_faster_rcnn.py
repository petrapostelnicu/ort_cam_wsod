import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image

from datasets import PascalVOC
from loggers import EvaluationLogger
from models import FasterRCNN
from utils import calculate_mAP_pin_pointing, calculate_mAP_detection, save_data_faster_rcnn


def filter_predictions(predictions, threshold):
    filtered_predictions = []

    for output in predictions:
        # Apply threshold
        mask = output['scores'] > threshold
        filtered_output = {
            'boxes': output['boxes'][mask],
            'scores': output['scores'][mask],
            'labels': output['labels'][mask]
        }
        filtered_predictions.append(filtered_output)

    return filtered_predictions


def generate_pin_pointing_labels(predictions):
    # Method to create pin points from Faster-RCNN bounding box predictions
    for prediction in predictions:
        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']

        pin_points = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            pin_points.append([center_x.item(), center_y.item()])

        prediction['pin_points'] = torch.tensor(pin_points)
        prediction['pin_points_labels'] = labels
        prediction['pin_points_scores'] = scores

    return predictions


def run(from_pseudo=False, model_name='vgg16', fpn_layer='0', resize='no_resize'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if resize == 'resize':
        full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test', transform=transform)
        images = full_dataset.voc_dataset.images
        resize_img = transforms.Resize((224, 224))
        images = [resize_img(read_image(img)) for img in images]
    else:
        full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test')
        images = full_dataset.voc_dataset.images
        images = [read_image(img) for img in images]
    localization_dataset, small_localization_dataset, large_localization_dataset, multi_instance_dataset, multi_class_dataset = full_dataset.get_split_datasets()

    # Prepare data loaders for the full dataset and its subsets
    datasets = {
        "full_dataset": full_dataset,
        "localization_dataset": localization_dataset,
        "small_localization_dataset": small_localization_dataset,
        "large_localization_dataset": large_localization_dataset,
        "multi_instance_dataset": multi_instance_dataset,
        "multi_class_dataset": multi_class_dataset,
    }

    data_loader_full = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    targets_full = [data_loader_full.dataset.__getitem__(i)[1] for i in range(data_loader_full.dataset.__len__())]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    targets_full = [{k: v.to(device) for k, v in t.items()} for t in targets_full]

    if from_pseudo:
        if model_name == 'vgg16':
            model = FasterRCNN(num_classes=21,
                               model_path=f'pretrained_models/faster_rcnn_pascal_2007_pseudo_{model_name}_final_{resize}.pth')
            eval_logger = EvaluationLogger(
                f'logs/evaluate_test_faster_rcnn_pascal_2007_pseudo_{model_name}_final_{resize}.csv')
        else:
            model = FasterRCNN(num_classes=21,
                               model_path=f'pretrained_models/faster_rcnn_pascal_2007_pseudo_{model_name}_layer{fpn_layer}_final_{resize}.pth')
            eval_logger = EvaluationLogger(
                f'logs/evaluate_test_faster_rcnn_pascal_2007_pseudo_{model_name}_final_layer{fpn_layer}_{resize}.csv')
    else:
        model = FasterRCNN(num_classes=21, model_path=f'pretrained_models/faster_rcnn_pascal_2007_final_{resize}.pth')
        eval_logger = EvaluationLogger(
            f'logs/evaluate_test_faster_rcnn_pascal_2007_final_{resize}.csv')

    model.load_weights()

    total_start_time = time.time()
    predictions_full = model.evaluate_model(data_loader=data_loader_full)
    total_time = time.time() - total_start_time
    eval_logger.log_time(total_time)

    predictions_full = filter_predictions(predictions_full, 0.5)
    predictions_full = generate_pin_pointing_labels(predictions_full)

    predictions_full = [{k: v.to(device) for k, v in p.items()} for p in predictions_full]

    # Evaluate each subset using the full dataset predictions and targets
    for name, dataset in datasets.items():
        if name == "full_dataset":
            predictions = predictions_full
            targets = targets_full
        else:
            subset_indices = dataset.indices
            predictions = [predictions_full[i] for i in subset_indices]
            targets = [targets_full[i] for i in subset_indices]

        print('-----------------------------------------------------------------------------')
        print(f'Evaluating data split {name}:')

        pin_pointing_aps, pin_pointing_map = calculate_mAP_pin_pointing(predictions, targets)
        eval_logger.log_mAP_pin_pointing(pin_pointing_aps, pin_pointing_map, name)
        print(f'Pin Pointing AP per class: {pin_pointing_aps}')
        print(f'Pin Pointing mAP: {pin_pointing_map}')
        print('------------------')

        detection_map = calculate_mAP_detection(predictions, targets)
        eval_logger.log_mAP_detection(detection_map, name)
        print(f'Detection mAP: {detection_map}')
        print('------------------')

        if name == "full_dataset":
            save_data_faster_rcnn(predictions, targets, images, from_pseudo=from_pseudo, model_name=model_name,
                                  fpn_layer=fpn_layer, resize=resize)


def run_reduced(from_pseudo=False, model_name='vgg16', fpn_layer='0', resize='no_resize'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test', transform=transform)
    images = full_dataset.voc_dataset.images
    resize_img = transforms.Resize((224, 224))
    images = [resize_img(read_image(img)) for img in images]
    full_dataset = full_dataset.get_reduced()

    data_loader_full = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    targets_full = [data_loader_full.dataset.__getitem__(i)[1] for i in range(data_loader_full.dataset.__len__())]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    targets_full = [{k: v.to(device) for k, v in t.items()} for t in targets_full]

    # model = FasterRCNN(num_classes=21, model_path='pretrained_models/faster_rcnn_pascal_2007_20_epochs_new.pth')
    model = FasterRCNN(num_classes=21, model_path='pretrained_models/faster_rcnn_pascal_2007_final.pth')
    model.load_weights()

    predictions_full = model.evaluate_model(data_loader=data_loader_full)

    predictions_full = filter_predictions(predictions_full, 0.5)
    predictions_full = generate_pin_pointing_labels(predictions_full)

    predictions_full = [{k: v.to(device) for k, v in p.items()} for p in predictions_full]

    print('-----------------------------------------------------------------------------')
    print(f'Evaluating data split full dataset:')

    pin_pointing_aps, pin_pointing_map = calculate_mAP_pin_pointing(predictions_full, targets_full)
    print(f'Pin Pointing AP per class: {pin_pointing_aps}')
    print(f'Pin Pointing mAP: {pin_pointing_map}')
    print('------------------')

    detection_map = calculate_mAP_detection(predictions_full, targets_full)
    print(f'Detection mAP: {detection_map}')
    print('------------------')

    images = [images[i] for i in full_dataset.indices]
    save_data_faster_rcnn(predictions_full, targets_full, images, from_pseudo=from_pseudo, model_name=model_name,
                          fpn_layer=fpn_layer, resize=resize)
