import numpy as np


def calculate_precision_recall(predicted_points, predicted_scores, gt_bboxes):
    # Sort predictions by scores in descending order
    sorted_indices = np.argsort(predicted_scores)[::-1]
    predicted_points = [predicted_points[i] for i in sorted_indices]
    predicted_scores = [predicted_scores[i] for i in sorted_indices]

    # Initialize lists to hold true positives and false positives
    tp_list = []
    fp_list = []

    # Track points and bounding boxes that have been taken
    taken_points = set()
    taken_bboxes = set()

    # Loop over predicted points
    for point_idx, point in enumerate(predicted_points):
        x, y = point
        match_found = False
        best_score = float('inf')
        best_bbox_idx = -1

        # Check if the point falls inside any ground truth bounding box
        for bbox_idx, (xmin, ymin, xmax, ymax) in enumerate(gt_bboxes):
            if bbox_idx in taken_bboxes:
                continue
            # The point closest to the ground truth bounding box center is considered the true positive
            if xmin <= x <= xmax and ymin <= y <= ymax:
                # Calculate the distance to the center of the bounding box
                gt_center_x = (xmin + xmax) / 2
                gt_center_y = (ymin + ymax) / 2
                distance_to_center = np.sqrt((x - gt_center_x) ** 2 + (y - gt_center_y) ** 2)

                if distance_to_center < best_score:
                    best_score = distance_to_center
                    best_bbox_idx = bbox_idx
                    match_found = True

        # If a match is found, mark it as a true positive
        if match_found:
            tp_list.append(1)
            fp_list.append(0)
            taken_points.add(point_idx)
            taken_bboxes.add(best_bbox_idx)
        else:
            # If no match is found, it is a false positive
            tp_list.append(0)
            fp_list.append(1)

    if np.all(np.array(tp_list) == 0):
        precision = np.zeros_like(tp_list, dtype=float)
        recall = np.zeros_like(tp_list, dtype=float)
        # print('no true positives found')
        return precision, recall

    # Calculate cumulative sum of true positives and false positives
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    # print(f'tp: {tp_cumsum}')
    # print('--------------')
    # print(f'fp: {fp_cumsum}')
    # print('--------------')

    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_bboxes)

    # print(f'recall: {recall}')
    # print('--------------')

    return precision, recall


def calculate_average_precision(precision, recall):
    # Define 101 recall levels (like COCO mAP; Pascal mAP uses 11 point interpolation)
    recall_levels = np.linspace(0, 1, 101)

    # Interpolate precision at each recall level
    interpolated_precision = []
    for r in recall_levels:
        if np.sum(recall >= r) == 0:
            interpolated_precision.append(0)
        else:
            interpolated_precision.append(np.max(precision[recall >= r]))

    # Calculate Average Precision
    ap = np.mean(interpolated_precision)

    return ap


def aggregate_predictions_and_targets(predictions, targets):
    classwise_predictions = {}
    classwise_gt = {}

    for image_predictions, image_targets in zip(predictions, targets):
        labels = image_predictions['pin_points_labels']
        points = image_predictions['pin_points']
        scores = image_predictions['pin_points_scores']
        for label, point, score in zip(labels, points, scores):
            if int(label) not in classwise_predictions:
                classwise_predictions[int(label)] = {'points': [], 'scores': []}
            classwise_predictions[int(label)]['points'].append(point.detach().cpu().numpy())
            classwise_predictions[int(label)]['scores'].append(score.detach().cpu().numpy())

        gt_labels = image_targets['labels']
        bboxes = image_targets['boxes']
        for label, bbox in zip(gt_labels, bboxes):
            if int(label) not in classwise_gt:
                classwise_gt[int(label)] = []
            classwise_gt[int(label)].append(bbox.detach().cpu().numpy())

    # print(classwise_predictions.keys())
    # print('----------------')
    # print(classwise_gt.keys())
    return classwise_predictions, classwise_gt


def calculate_mAP_pin_pointing(predictions, targets):
    classwise_predictions, classwise_gt = aggregate_predictions_and_targets(predictions, targets)
    aps = {}

    # Ensure all classes in either the predictions or the ground truth are considered
    all_classes = set(classwise_gt.keys()).union(set(classwise_predictions.keys()))

    for label in all_classes:
        if label in classwise_predictions:
            predicted_points = classwise_predictions[label]['points']
            predicted_scores = classwise_predictions[label]['scores']
        else:
            predicted_points = []
            predicted_scores = []

        gt_bboxes = classwise_gt.get(label, [])

        if len(predicted_points) == 0:
            # If there are no predictions for this class, AP is 0
            aps[label] = 0
        else:
            # Calculate precision and recall
            precision, recall = calculate_precision_recall(predicted_points, predicted_scores, gt_bboxes)
            # Calculate Average Precision using 101-point interpolation method
            ap = calculate_average_precision(precision, recall)
            aps[label] = ap

    # Mean Average Precision
    map_score = sum(aps.values()) / len(aps) if aps else 0.0
    return aps, map_score
