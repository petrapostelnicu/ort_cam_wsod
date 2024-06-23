import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches

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


def load_bboxes_and_labels(txt_file):
    bboxes = []
    labels = []
    with open(txt_file, 'r') as f:
        for line in list(f.readlines())[1:]:
            label, xmin, ymin, xmax, ymax = line.strip().split(',')
            labels.append(label)
            bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
    return bboxes, labels


def load_pin_points(file):
    points = []
    labels = []
    with open(file, 'r') as f:
        for line in list(f.readlines())[1:]:
            label, cx, cy = line.strip().split(',')
            labels.append(label)
            points.append([float(cx), float(cy)])
    return points, labels


def show(image, bboxes_list, labels, gt_bboxes, gt_labels, points, points_labels):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    labels = [VOC_BBOX_LABEL_NAMES[int(label)] for label in labels]
    gt_labels = [VOC_BBOX_LABEL_NAMES[int(gt_label)] for gt_label in gt_labels]
    points_labels = [VOC_BBOX_LABEL_NAMES[int(points_label)] for points_label in points_labels]

    # Plot original image with bounding boxes
    axs[0].imshow(np.asarray(image))  # Display the background image
    for bbox, label in zip(bboxes_list, labels):
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)

        # Add label text at the top-left corner of the bounding box
        axs[0].text(xmin, ymin - 5, label, fontsize=10, color='white', backgroundcolor='red', fontweight='bold')

    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
        gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin, linewidth=2,
                                    edgecolor='green', facecolor='none')
        axs[0].add_patch(gt_rect)

        # Add label text at the top-left corner of the bounding box
        axs[0].text(gt_xmin, gt_ymin - 5, gt_label, fontsize=10, color='white', backgroundcolor='green',
                    fontweight='bold')

    axs[0].set_title("Original Image with CAM generated BBoxes")
    axs[0].axis('off')  # Hide the axes

    # Plot original image with pin points
    axs[1].imshow(np.asarray(image))  # Display the background image
    for point, label in zip(points, points_labels):
        x, y = point
        circle = patches.Circle((x, y), radius=5, linewidth=2, edgecolor='r', facecolor='none')
        axs[1].add_patch(circle)

        # Add label text next to the point
        axs[1].text(x + 6, y - 6, label, fontsize=10, color='white', backgroundcolor='red', fontweight='bold')

    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
        gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin, linewidth=2,
                                    edgecolor='green', facecolor='none')
        axs[1].add_patch(gt_rect)

        # Add label text at the top-left corner of the bounding box
        axs[1].text(gt_xmin, gt_ymin - 5, gt_label, fontsize=10, color='white', backgroundcolor='green',
                    fontweight='bold')

    axs[1].set_title("Original Image with CAM generated Pin Points")
    axs[1].axis('off')  # Hide the axes

    plt.tight_layout()
    plt.show()


def run(from_pseudo=False, model_name='vgg16', fpn_layer='0', resize='no_resize'):
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
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')],
                         key=lambda x: int(os.path.splitext(x)[0]))
    txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')],
                       key=lambda x: int(os.path.splitext(x)[0]))
    targets_files = sorted([f for f in os.listdir(targets_dir) if f.endswith('.txt')],
                           key=lambda x: int(os.path.splitext(x)[0]))
    pin_points_files = sorted([f for f in os.listdir(pin_points_dir) if f.endswith('.txt')],
                              key=lambda x: int(os.path.splitext(x)[0]))

    for image_file, txt_file, target_file, pin_points_file in zip(image_files, txt_files, targets_files,
                                                                  pin_points_files):
        # Load original and CAM images
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        # Load bounding boxes and labels
        txt_path = os.path.join(txt_dir, txt_file)
        bboxes, labels = load_bboxes_and_labels(txt_path)

        # Load pinpoints and their labels
        pin_points_path = os.path.join(pin_points_dir, pin_points_file)
        points, points_labels = load_pin_points(pin_points_path)

        # Load ground truth bounding boxes and labels
        target_path = os.path.join(targets_dir, target_file)
        gt_bboxes, gt_labels = load_bboxes_and_labels(target_path)

        # Call the show function to visualize
        show(image, bboxes, labels, gt_bboxes, gt_labels, points, points_labels)
