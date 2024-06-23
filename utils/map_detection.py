from torchmetrics.detection.mean_ap import MeanAveragePrecision


def calculate_mAP_detection(predictions, targets):
    metric = MeanAveragePrecision(class_metrics=True, iou_thresholds=[0.5])
    metric.update(predictions, targets)
    return metric.compute()
