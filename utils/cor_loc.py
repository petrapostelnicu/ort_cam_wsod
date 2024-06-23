def compute_iou(box1, box2):
    # Compute the intersection over union (IoU) of two bounding boxes.
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou


def compute_corloc(predictions, targets, iou_threshold=0.5):
    class_tp_fp = {}

    # Initialize class_tp_fp for predicted labels
    for pred in predictions:
        for p_label in pred['labels']:
            p_label = int(p_label)
            if p_label not in class_tp_fp:
                class_tp_fp[p_label] = {'tp': 0, 'fp': 0}

    # Aggregate TP and FP for each class across all images
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        target_boxes = target['boxes']
        target_labels = target['labels']

        for p_box, p_label in zip(pred_boxes, pred_labels):
            p_label = int(p_label)
            match_found = False
            for t_box, t_label in zip(target_boxes, target_labels):
                if p_label == t_label:  # Match class
                    iou = compute_iou(p_box, t_box)
                    if iou >= iou_threshold:
                        match_found = True
                        break
            if match_found:
                class_tp_fp[p_label]['tp'] += 1
            else:
                class_tp_fp[p_label]['fp'] += 1

    class_corloc = {}
    for class_id, counts in class_tp_fp.items():
        tp = counts['tp']
        fp = counts['fp']
        if tp + fp > 0:
            corloc = tp / (tp + fp)
            class_corloc[class_id] = corloc

    mean_corloc_value = sum(class_corloc.values()) / len(class_corloc)
    return class_corloc, mean_corloc_value
