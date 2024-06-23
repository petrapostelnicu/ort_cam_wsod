import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchmetrics.classification import MultilabelAveragePrecision


def compute_accuracy(y_true, y_scores, threshold=0.5):
    class_targets = [targ['labels'] for targ in y_true]
    one_hot_targets = []
    for labels in class_targets:
        target_vector = torch.zeros(21, dtype=torch.int)
        for label in labels:
            target_vector[label] = 1
        one_hot_targets.append(target_vector)

    preds = [p.cpu() for p in y_scores]
    preds = [p.detach() for p in preds]
    preds = np.array(preds)

    one_hot_preds = []
    for pred in preds:
        pred_vector = torch.zeros(21, dtype=torch.int)
        idx = np.where(pred >= threshold)
        for label in idx:
            pred_vector[label] = 1
        one_hot_preds.append(pred_vector)

    print(one_hot_preds)
    print('--------------')
    print(one_hot_targets)
    exact_match_ratio = accuracy_score(one_hot_targets, one_hot_preds)
    print('accuracy: ', exact_match_ratio)

    return exact_match_ratio


def calculate_mAP_classification(predictions, targets):
    ap = MultilabelAveragePrecision(num_labels=21, average=None, thresholds=None)
    map = MultilabelAveragePrecision(num_labels=21, average='macro', thresholds=None)
    class_targets = [targ['labels'] for targ in targets]
    one_hot_targets = []
    for labels in class_targets:
        target_vector = torch.zeros(21, dtype=torch.int)
        for label in labels:
            target_vector[label] = 1
        one_hot_targets.append(target_vector)

    preds = [p.cpu() for p in predictions]
    preds = [p.detach() for p in preds]
    return ap(torch.stack(preds), torch.stack(one_hot_targets)), map(torch.stack(preds), torch.stack(one_hot_targets))
