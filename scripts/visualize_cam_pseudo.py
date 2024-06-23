import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches
from matplotlib.gridspec import GridSpec

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


def show_all(image, cam_image, threshold_image, bboxes_list, labels, gt_bboxes, gt_labels, points, points_labels,
             threshold_pin_pointing_image, file_name, classifier_name, fpn_layer, resize):
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1])

    # fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))

    labels = [VOC_BBOX_LABEL_NAMES[int(label)] for label in labels]
    gt_labels = [VOC_BBOX_LABEL_NAMES[int(gt_label)] for gt_label in gt_labels]
    points_labels = [VOC_BBOX_LABEL_NAMES[int(points_label)] for points_label in points_labels]

    # Plot original image with bounding boxes
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(np.asarray(image))  # Display the background image
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
        gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin, linewidth=2,
                                    edgecolor='green', facecolor='none')
        ax0.add_patch(gt_rect)

        # Add label text at the top-left corner of the bounding box
        ax0.text(gt_xmin, gt_ymin - 5, gt_label, fontsize=10, color='white', backgroundcolor='green', fontweight='bold')
    for bbox, label in zip(bboxes_list, labels):
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax0.add_patch(rect)

        # Add label text at the top-left corner of the bounding box
        ax0.text(xmin, ymin - 5, label, fontsize=10, color='white', backgroundcolor='red', fontweight='bold')

    # axs[0].set_title("Original Image with CAM generated BBoxes")
    ax0.axis('off')  # Hide the axes

    # Plot CAM image
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(np.asarray(cam_image))
    # axs[1].set_title("Grad-CAM++")
    ax1.axis('off')  # Hide the axes

    # Plot threshold image
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(np.asarray(threshold_image))
    # axs[2].set_title("Segmentation threshold for bounding boxes")
    ax2.axis('off')  # Hide the axes

    # Plot threshold image
    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(np.asarray(threshold_pin_pointing_image))
    # axs[3].set_title("Segmentation threshold for pin pointing")
    ax3.axis('off')  # Hide the axes

    # Plot original image with pin points
    ax4 = fig.add_subplot(gs[4])
    ax4.imshow(np.asarray(image))  # Display the background image
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
        gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin, linewidth=2,
                                    edgecolor='green', facecolor='none')
        ax4.add_patch(gt_rect)

        # Add label text at the top-left corner of the bounding box
        ax4.text(gt_xmin, gt_ymin - 5, gt_label, fontsize=10, color='white', backgroundcolor='green',
                 fontweight='bold')
    for point, label in zip(points, points_labels):
        x, y = point
        circle = patches.Circle((x, y), radius=5, linewidth=2, edgecolor='b', facecolor='none')
        ax4.add_patch(circle)

        # Add label text next to the point
        ax4.text(x + 6, y - 6, label, fontsize=10, color='white', backgroundcolor='blue', fontweight='bold')

    # axs[4].set_title("Original Image with CAM generated Pin Points")
    ax4.axis('off')  # Hide the axes

    plt.tight_layout()

    if classifier_name == 'vgg16':
        path = f'../plots/{classifier_name}_{resize}/{file_name}.png'
    else:
        path = f'../plots/{classifier_name}_{fpn_layer}_{resize}/{file_name}.png'
    plt.savefig(path)

    plt.show()


def show_img_cam(image, cam_image, threshold_image, bboxes_list, labels, gt_bboxes, gt_labels, points, points_labels,
                 threshold_pin_pointing_image, file_name, classifier_name, fpn_layer, resize):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    labels = [VOC_BBOX_LABEL_NAMES[int(label)] for label in labels]
    gt_labels = [VOC_BBOX_LABEL_NAMES[int(gt_label)] for gt_label in gt_labels]
    points_labels = [VOC_BBOX_LABEL_NAMES[int(points_label)] for points_label in points_labels]

    # Plot original image with bounding boxes, pin points
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(np.asarray(image))  # Display the background image
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
        gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin, linewidth=2,
                                    edgecolor='green', facecolor='none')
        ax0.add_patch(gt_rect)
        ax0.text(gt_xmin, gt_ymin - 5, gt_label, fontsize=10, color='white', backgroundcolor='green', fontweight='bold')

    for bbox, label in zip(bboxes_list, labels):
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax0.add_patch(rect)
        ax0.text(xmin, ymin - 5, label, fontsize=10, color='white', backgroundcolor='red', fontweight='bold')

    for point, label in zip(points, points_labels):
        x, y = point
        circle = patches.Circle((x, y), radius=5, linewidth=2, edgecolor='b', facecolor='none')
        ax0.add_patch(circle)
        ax0.text(x + 6, y - 6, label, fontsize=10, color='white', backgroundcolor='blue', fontweight='bold')

    ax0.axis('off')  # Hide the axes

    # Plot CAM image
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(np.asarray(cam_image))
    ax1.axis('off')  # Hide the axes

    plt.tight_layout()

    if classifier_name == 'vgg16':
        path = f'../plots/{classifier_name}_{resize}/{file_name}.png'
    else:
        path = f'../plots/{classifier_name}_{fpn_layer}_{resize}/{file_name}.png'
    plt.savefig(path)

    plt.show()


def run(classifier_name, fpn_layer='0', resize='no_resize'):
    if classifier_name == 'vgg16':
        image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{resize}/images'
        cam_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{resize}/cam_images'
        txt_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{resize}/labels'
        thresholds_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{resize}/thresholds'
        targets_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{resize}/ground_truth_labels'
        pin_points_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{resize}/pin_points'
        thresholds_pin_points_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{resize}/thresholds_pin_pointing'
    else:
        image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{fpn_layer}_{resize}/images'
        cam_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{fpn_layer}_{resize}/cam_images'
        txt_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{fpn_layer}_{resize}/labels'
        thresholds_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{fpn_layer}_{resize}/thresholds'
        targets_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{fpn_layer}_{resize}/ground_truth_labels'
        pin_points_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{fpn_layer}_{resize}/pin_points'
        thresholds_pin_points_image_dir = f'../data/pascal_voc_2007/cam_pseudo_test_{classifier_name}_{fpn_layer}_{resize}/thresholds_pin_pointing'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')],
                         key=lambda x: int(os.path.splitext(x)[0]))
    cam_image_files = sorted([f for f in os.listdir(cam_image_dir) if f.endswith('.png')],
                             key=lambda x: int(os.path.splitext(x)[0]))
    thresholds_image_files = sorted([f for f in os.listdir(thresholds_image_dir) if f.endswith('.png')],
                                    key=lambda x: int(os.path.splitext(x)[0]))
    txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')],
                       key=lambda x: int(os.path.splitext(x)[0]))
    targets_files = sorted([f for f in os.listdir(targets_dir) if f.endswith('.txt')],
                           key=lambda x: int(os.path.splitext(x)[0]))
    pin_points_files = sorted([f for f in os.listdir(pin_points_dir) if f.endswith('.txt')],
                              key=lambda x: int(os.path.splitext(x)[0]))
    thresholds_pin_pointing_image_files = sorted(
        [f for f in os.listdir(thresholds_pin_points_image_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0]))

    for image_file, cam_image_file, thresholds_image_file, txt_file, target_file, pin_points_file, thresholds_pin_pointing_image_file in zip(
            image_files, cam_image_files, thresholds_image_files, txt_files, targets_files, pin_points_files,
            thresholds_pin_pointing_image_files):
        # Load original and CAM images
        image_path = os.path.join(image_dir, image_file)
        cam_image_path = os.path.join(cam_image_dir, cam_image_file)
        thresholds_image_path = os.path.join(thresholds_image_dir, thresholds_image_file)
        image = Image.open(image_path)
        cam_image = Image.open(cam_image_path)
        threshold_image = Image.open(thresholds_image_path)

        # Load bounding boxes and labels
        txt_path = os.path.join(txt_dir, txt_file)
        bboxes, labels = load_bboxes_and_labels(txt_path)

        # Load pinpoints and their labels
        pin_points_path = os.path.join(pin_points_dir, pin_points_file)
        points, points_labels = load_pin_points(pin_points_path)
        thresholds_pin_pointing_image_path = os.path.join(thresholds_pin_points_image_dir,
                                                          thresholds_pin_pointing_image_file)
        threshold_pin_pointing_image = Image.open(thresholds_pin_pointing_image_path)

        # Load ground truth bounding boxes and labels
        target_path = os.path.join(targets_dir, target_file)
        gt_bboxes, gt_labels = load_bboxes_and_labels(target_path)

        # Call the show function to visualize
        show_all(image, cam_image, threshold_image, bboxes, labels, gt_bboxes, gt_labels, points, points_labels,
                 threshold_pin_pointing_image, image_file, classifier_name, fpn_layer, resize)
