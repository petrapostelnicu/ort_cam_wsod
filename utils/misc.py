import os

import torch
from torchvision.utils import save_image


def check_tensor(tensor):
    if tensor.requires_grad:
        tensor.requires_grad_(False)

    if len(tensor.shape) == 3 and tensor.shape[0] > 3:
        # Convert from HWC to CHW format expected by save_image
        tensor = tensor.permute(2, 0, 1)  # Change to (3, 200, 200)

    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)  # Convert to float32
    if len(tensor) != 0 and tensor.max().item() > 1.0:
        tensor = tensor / 255.0  # Normalize if it's not in the range [0, 1]

    return tensor


def save_data(predictions, targets, model_name, fpn_layer, resize):
    # Create directories
    if model_name == 'vgg16':
        image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/images'
        cam_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/cam_images'
        cam_image_per_class_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/cam_images_per_class'
        txt_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/labels'
        thresholds_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/thresholds'
        thresholds_image_per_class_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/thresholds_per_class'
        targets_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/ground_truth_labels'
        pin_points_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/pin_points'
        thresholds_pin_points_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/thresholds_pin_pointing'
        thresholds_pin_points_image_per_class_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{resize}/thresholds_pin_pointing_per_class'
    else:
        image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/images'
        cam_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/cam_images'
        cam_image_per_class_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/cam_images_per_class'
        txt_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/labels'
        thresholds_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/thresholds'
        thresholds_image_per_class_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/thresholds_per_class'
        targets_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/ground_truth_labels'
        pin_points_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/pin_points'
        thresholds_pin_points_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/thresholds_pin_pointing'
        thresholds_pin_points_image_per_class_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{model_name}_{fpn_layer}_{resize}/thresholds_pin_pointing_per_class'

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(cam_image_dir, exist_ok=True)
    os.makedirs(cam_image_per_class_dir, exist_ok=True)
    os.makedirs(thresholds_image_dir, exist_ok=True)
    os.makedirs(thresholds_image_per_class_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)
    os.makedirs(pin_points_dir, exist_ok=True)
    os.makedirs(thresholds_pin_points_image_dir, exist_ok=True)
    os.makedirs(thresholds_pin_points_image_per_class_dir, exist_ok=True)

    for index, entry in enumerate(predictions):
        # Save the image
        image_path = os.path.join(image_dir, f'{index}.png')
        image = check_tensor(entry['image'])
        save_image(image, image_path)

        # Save the CAM image
        cam_image_path = os.path.join(cam_image_dir, f'{index}.png')
        cam_image = check_tensor(entry['cam_image'])
        if len(cam_image) != 0:
            save_image(cam_image, cam_image_path)
        else:
            save_image(image, cam_image_path)

        # Save the CAM image per class
        cam_images = check_tensor(entry['cam_images'])
        for cam_image, class_label in zip(cam_images, entry['labels'].unique()):
            cam_image_path = os.path.join(cam_image_per_class_dir, f'{index}_{class_label}.png')
            cam_image = check_tensor(cam_image)
            if len(cam_image) != 0:
                save_image(cam_image, cam_image_path)
            else:
                save_image(image, cam_image_path)

        # Save the CAM with segmentation applied image
        threshold_image_path = os.path.join(thresholds_image_dir, f'{index}.png')
        threshold_image = check_tensor(entry['threshold_cam'])
        if len(threshold_image) != 0:
            save_image(threshold_image, threshold_image_path)
        else:
            save_image(image, threshold_image_path)

        # Save the CAM with bounding box segmentation applied image per class
        threshold_images = check_tensor(entry['threshold_images'])
        for threshold_image, class_label in zip(threshold_images, entry['labels'].unique()):
            threshold_image_path = os.path.join(thresholds_image_per_class_dir, f'{index}_{class_label}.png')
            threshold_image = check_tensor(threshold_image)
            if len(threshold_image) != 0:
                save_image(threshold_image, threshold_image_path)
            else:
                save_image(image, threshold_image_path)

        # Save the labels and boxes to a text file
        txt_path = os.path.join(txt_dir, f'{index}.txt')
        with open(txt_path, 'w') as f:
            f.write('label,xmin,ymin,xmax,ymax\n')
            for label, box in zip(entry['labels'], entry['boxes']):
                # box format is assumed to be [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                f.write(f'{label},{xmin},{ymin},{xmax},{ymax}\n')

        # Save the pinpoints labels and points to a text file
        pin_points_path = os.path.join(pin_points_dir, f'{index}.txt')
        with open(pin_points_path, 'w') as f:
            f.write('label,cx,cy\n')
            for label, point in zip(entry['pin_points_labels'], entry['pin_points']):
                # box format is assumed to be [xmin, ymin, xmax, ymax]
                cx, cy = point
                f.write(f'{label},{cx},{cy}\n')

        # Save the CAM with pin pointing segmentation applied image
        threshold_pin_pointing_image_path = os.path.join(thresholds_pin_points_image_dir, f'{index}.png')
        threshold_pin_pointing_image = check_tensor(entry['pin_pointing_threshold'])
        if len(threshold_pin_pointing_image) != 0:
            save_image(threshold_pin_pointing_image, threshold_pin_pointing_image_path)
        else:
            save_image(image, threshold_pin_pointing_image_path)

        # Save the CAM with pin pointing segmentation applied image per class
        threshold_pin_pointing_images = check_tensor(entry['pin_pointing_all_thresholds'])
        for threshold_pin_pointing_image, class_label in zip(threshold_pin_pointing_images, entry['labels'].unique()):
            threshold_pin_pointing_image_path = os.path.join(thresholds_pin_points_image_per_class_dir,
                                                             f'{index}_{class_label}.png')
            threshold_pin_pointing_image = check_tensor(threshold_pin_pointing_image)
            if len(threshold_pin_pointing_image) != 0:
                save_image(threshold_pin_pointing_image, threshold_pin_pointing_image_path)
            else:
                save_image(image, threshold_pin_pointing_image_path)

    for index, gt in enumerate(targets):
        # Save the ground truth labels
        targets_path = os.path.join(targets_dir, f'{index}.txt')
        with open(targets_path, 'w') as f:
            f.write('label,xmin,ymin,xmax,ymax\n')
            for label, box in zip(gt['labels'], gt['boxes']):
                # box format is assumed to be [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                f.write(f'{label},{xmin},{ymin},{xmax},{ymax}\n')


def save_pseudo_labels(data, model_name, fpn_layer, resize):
    # Create directories
    if model_name == 'vgg16':
        image_dir = f'../data/pascal_voc_2007/cam_pseudo_train_{model_name}_{resize}/images'
        txt_dir = f'../data/pascal_voc_2007/cam_pseudo_train_{model_name}_{resize}/labels'
    else:
        image_dir = f'../data/pascal_voc_2007/cam_pseudo_train_{model_name}_{fpn_layer}_{resize}/images'
        txt_dir = f'../data/pascal_voc_2007/cam_pseudo_train_{model_name}_{fpn_layer}_{resize}/labels'

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    for index, entry in enumerate(data):
        # Save the image
        image_path = os.path.join(image_dir, f'{index}.png')
        image = check_tensor(entry['image'])
        # print(image)
        save_image(image, image_path)

        # Save the labels and boxes to a text file
        txt_path = os.path.join(txt_dir, f'{index}.txt')
        with open(txt_path, 'w') as f:
            f.write('label,xmin,ymin,xmax,ymax\n')
            for label, box in zip(entry['labels'], entry['boxes']):
                # box format is assumed to be [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                f.write(f'{label},{xmin},{ymin},{xmax},{ymax}\n')


def save_data_faster_rcnn(predictions, targets, images, from_pseudo, model_name, fpn_layer, resize):
    # Create directories
    if from_pseudo:
        if model_name == 'vgg16':
            image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{resize}/images'
            txt_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{resize}/labels'
            targets_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{resize}/ground_truth_labels'
            pin_points_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{resize}/pin_points'
        else:
            image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{fpn_layer}_{resize}/images'
            txt_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{fpn_layer}_{resize}/labels'
            targets_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{fpn_layer}_{resize}/ground_truth_labels'
            pin_points_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{model_name}_{fpn_layer}_{resize}/pin_points'
    else:
        image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{resize}/images'
        txt_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{resize}/labels'
        targets_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{resize}/ground_truth_labels'
        pin_points_dir = f'../data/pascal_voc_2007/cam_pseudo_test_faster_rcnn_{resize}/pin_points'

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)
    os.makedirs(pin_points_dir, exist_ok=True)

    for index, (entry, image) in enumerate(zip(predictions, images)):
        # Save the image
        image_path = os.path.join(image_dir, f'{index}.png')
        image = check_tensor(image)
        save_image(image, image_path)

        # Save the labels and boxes to a text file
        txt_path = os.path.join(txt_dir, f'{index}.txt')
        with open(txt_path, 'w') as f:
            f.write('label,xmin,ymin,xmax,ymax\n')
            for label, box in zip(entry['labels'], entry['boxes']):
                # box format is assumed to be [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                f.write(f'{label},{xmin},{ymin},{xmax},{ymax}\n')

        # Save the pinpoints labels and points to a text file
        pin_points_path = os.path.join(pin_points_dir, f'{index}.txt')
        with open(pin_points_path, 'w') as f:
            f.write('label,cx,cy\n')
            for label, point in zip(entry['pin_points_labels'], entry['pin_points']):
                # box format is assumed to be [xmin, ymin, xmax, ymax]
                cx, cy = point
                f.write(f'{label},{cx},{cy}\n')

    for index, gt in enumerate(targets):
        # Save the ground truth labels
        targets_path = os.path.join(targets_dir, f'{index}.txt')
        with open(targets_path, 'w') as f:
            f.write('label,xmin,ymin,xmax,ymax\n')
            for label, box in zip(gt['labels'], gt['boxes']):
                # box format is assumed to be [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                f.write(f'{label},{xmin},{ymin},{xmax},{ymax}\n')
