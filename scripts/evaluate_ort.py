import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import PascalVOC
from loggers import EvaluationLogger
from models import ORT
from utils import calculate_mAP_pin_pointing, calculate_mAP_detection, calculate_mAP_classification, compute_corloc, \
    save_data, save_pseudo_labels


def evaluate_test_set(classifier_name, fpn_layer='0', resize='no_resize'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if resize == 'no_resize':
        full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test')
    else:
        full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test', transform=transform)
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
    # targets_full = [{k: v.to(device) for k, v in t.items()} for t in targets_full]

    if classifier_name == 'vgg16':
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_{resize}.pth'
        eval_logger = EvaluationLogger(
            f'logs/evaluate_test_{classifier_name}_classifier_pascal_2007_final_{resize}.csv')
    else:
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.pth'
        eval_logger = EvaluationLogger(
            f'logs/evaluate_test_{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.csv')

    model = ORT(num_classes=21, model_path=model_path,
                classifier_name=classifier_name, fpn_layer=fpn_layer)
    model.load_weights()

    total_start_time = time.time()
    predictions_full = model.evaluate_model_detection(data_loader=data_loader_full, classification_threshold=0.5)
    total_time = time.time() - total_start_time
    eval_logger.log_time(total_time)

    # predictions_full = [{k: v.to(device) for k, v in p.items()} for p in predictions_full]

    class_predictions_full = model.evaluate_model_classification(data_loader=data_loader_full)

    # Evaluate each subset using the full dataset predictions and targets
    for name, dataset in datasets.items():
        if name == "full_dataset":
            predictions = predictions_full
            targets = targets_full
            class_predictions = class_predictions_full
        else:
            subset_indices = dataset.indices
            predictions = [predictions_full[i] for i in subset_indices]
            targets = [targets_full[i] for i in subset_indices]
            class_predictions = [class_predictions_full[i] for i in subset_indices]

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

        classification_ap, classification_map = calculate_mAP_classification(class_predictions, targets)
        eval_logger.log_mAP_classification(classification_ap, classification_map, name)
        print(f'Classification mAP: {classification_map}')
        print(f'Classification AP: {classification_ap}')

        if name == "full_dataset":
            save_data(predictions, targets, classifier_name, fpn_layer, resize)


def evaluate_test_set_reduced(classifier_name, fpn_layer='0', resize='no_resize'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    if resize == 'no_resize':
        full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test')
    else:
        full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test', transform=transform)
    full_dataset = full_dataset.get_reduced()

    data_loader_full = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    targets_full = [data_loader_full.dataset.__getitem__(i)[1] for i in range(data_loader_full.dataset.__len__())]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # targets_full = [{k: v.to(device) for k, v in t.items()} for t in targets_full]

    if classifier_name == 'vgg16':
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_{resize}.pth'
        eval_logger = EvaluationLogger(
            f'logs/evaluate_test_{classifier_name}_classifier_pascal_2007_final_{resize}.csv')
    else:
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.pth'
        eval_logger = EvaluationLogger(
            f'logs/evaluate_test_{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.csv')

    model = ORT(num_classes=21, model_path=model_path,
                classifier_name=classifier_name, fpn_layer=fpn_layer)
    model.load_weights()

    total_start_time = time.time()
    predictions_full = model.evaluate_model_detection(data_loader=data_loader_full, classification_threshold=0.5)
    total_time = time.time() - total_start_time
    eval_logger.log_time(total_time)

    predictions_full = [{k: v.to(device) for k, v in p.items()} for p in predictions_full]

    class_predictions_full = model.evaluate_model_classification(data_loader=data_loader_full)

    predictions = predictions_full
    targets = targets_full
    class_predictions = class_predictions_full

    print('-----------------------------------------------------------------------------')
    print(f'Evaluating data split:')

    pin_pointing_aps, pin_pointing_map = calculate_mAP_pin_pointing(predictions, targets)
    eval_logger.log_mAP_pin_pointing(pin_pointing_aps, pin_pointing_map, 'full_dataset')
    print(f'Pin Pointing AP per class: {pin_pointing_aps}')
    print(f'Pin Pointing mAP: {pin_pointing_map}')
    print('------------------')

    detection_map = calculate_mAP_detection(predictions, targets)
    eval_logger.log_mAP_detection(detection_map, 'full_dataset')
    print(f'Detection mAP: {detection_map}')
    print('------------------')

    classification_ap, classification_map = calculate_mAP_classification(class_predictions, targets)
    eval_logger.log_mAP_classification(classification_ap, classification_map, 'full_dataset')
    print(f'Classification mAP: {classification_map}')
    print(f'Classification AP: {classification_ap}')

    save_data(predictions, targets, classifier_name, fpn_layer, resize)


def evaluate_train_set(classifier_name, fpn_layer='0', resize='no_resize'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if resize == 'no_resize':
        full_dataset = PascalVOC(root='../data/pascal_voc_2007')
    else:
        full_dataset = PascalVOC(root='../data/pascal_voc_2007', transform=transform)
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

    # Evaluate the full dataset first
    data_loader_full = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    targets_full = [data_loader_full.dataset.__getitem__(i)[1] for i in range(data_loader_full.dataset.__len__())]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # targets_full = [{k: v.to(device) for k, v in t.items()} for t in targets_full]

    if classifier_name == 'vgg16':
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_{resize}.pth'
        eval_logger = EvaluationLogger(
            f'logs/evaluate_train_{classifier_name}_classifier_pascal_2007_final_{resize}.csv')
    else:
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.pth'
        eval_logger = EvaluationLogger(
            f'logs/evaluate_train_{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_{resize}.csv')

    model = ORT(num_classes=21, model_path=model_path,
                classifier_name=classifier_name, fpn_layer=fpn_layer)
    model.load_weights()

    total_start_time = time.time()
    predictions_full = model.evaluate_model_detection(data_loader=data_loader_full, classification_threshold=0.5)
    total_time = time.time() - total_start_time
    eval_logger.log_time(total_time)

    # predictions_full = [{k: v.to(device) for k, v in p.items()} for p in predictions_full]

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

        cor_loc_per_class, mean_cor_loc = compute_corloc(predictions, targets)
        eval_logger.log_cor_loc(cor_loc_per_class, mean_cor_loc, name)
        print(f'CorLoc per class: {cor_loc_per_class}')
        print(f'Mean CorLoc: {mean_cor_loc}')

        if name == "full_dataset":
            save_pseudo_labels(predictions, classifier_name, fpn_layer, resize)


def run(classifier_name, fpn_layer='0', resize='no_resize'):
    print('Evaluation on test set')
    # evaluate_test_set_reduced(classifier_name, fpn_layer, resize)
    evaluate_test_set(classifier_name, fpn_layer, resize)
    print('--------------------------------------------------------------------------------------------------')
    print('Evaluation on trainval set')
    evaluate_train_set(classifier_name, fpn_layer, resize)
