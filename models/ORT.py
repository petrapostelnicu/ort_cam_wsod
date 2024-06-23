import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet50_Weights, vgg16, VGG16_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class VGG16ClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(VGG16ClassifierModel, self).__init__()
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, batch):
        outputs = []
        for x in batch:
            x = x.unsqueeze(0)
            output = self.model(x)
            outputs.append(output)
        return torch.cat(outputs)


class FPNClassifierModel(nn.Module):
    def __init__(self, num_classes, fpn_layer):
        super(FPNClassifierModel, self).__init__()
        # FPN with ResNet50 backbone
        self.backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT)
        # Feature map to use
        self.fpn_layer = fpn_layer
        # Convolutional layer used as target layer for GradCAM++
        self.final_conv = nn.Conv2d(self.backbone.out_channels, 256, kernel_size=1)
        # Use average pooling to resize feature maps
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        out_channels = self.backbone.out_channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 7 * 7, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, num_classes, bias=True)
        )

    def forward(self, batch):
        outputs = []
        for x in batch:
            features = self.backbone(x)
            x = self.final_conv(features[self.fpn_layer])
            x = self.pool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            outputs.append(logits)

        return torch.cat(outputs)


class ORT:
    def __init__(self, num_classes, model_path, classifier_name, fpn_layer='0'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print(f'Running on {torch.cuda.get_device_name(0)}')

        self.classifier_name = classifier_name

        if self.classifier_name == 'fpn':
            self.model = FPNClassifierModel(num_classes, fpn_layer)
        elif self.classifier_name == 'vgg16':
            # self.model = vgg16(weights=VGG16_Weights.DEFAULT)
            # self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
            self.model = VGG16ClassifierModel(num_classes)
        self.model_path = model_path
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.num_classes = num_classes

    def train_model(self, data_loader_train, data_loader_val, loss_logger, num_epochs=20):
        total_start_time = time.time()
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            progress_bar_train = tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
            train_running_loss = 0.0
            for train_images, train_targets in progress_bar_train:
                # images = torch.stack([image.to(self.device) for image in train_images])
                # targets = torch.stack([t['labels'].to(self.device).float() for t in train_targets])
                images = [image.to(self.device) for image in train_images]
                targets = torch.stack([t['labels'].to(self.device).float() for t in train_targets])

                class_probabilities = self.model(images)
                loss = self.criterion(class_probabilities, targets)
                train_running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_train_loss = train_running_loss / len(data_loader_train)
            train_losses.append(average_train_loss)
            progress_bar_train.close()
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Training - Loss: {average_train_loss}")

            # Validation
            self.model.eval()
            progress_bar_val = tqdm(data_loader_val, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
            val_running_loss = 0.0
            with torch.no_grad():
                for val_images, val_targets in progress_bar_val:
                    # images = torch.stack([image.to(self.device) for image in val_images])
                    # targets = torch.stack([t['labels'].to(self.device).float() for t in val_targets])
                    images = [image.to(self.device) for image in val_images]
                    targets = torch.stack([t['labels'].to(self.device).float() for t in val_targets])

                    class_probabilities = self.model(images)
                    loss = self.criterion(class_probabilities, targets)
                    val_running_loss += loss.item()

            average_val_loss = val_running_loss / len(data_loader_val)
            val_losses.append(average_val_loss)
            progress_bar_val.close()
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Validation - Loss: {average_val_loss}")

            # Check for improvement
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                self.save_weights()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # If no improvement for 2 epochs, restore best weights and reduce learning rate
            if epochs_since_improvement >= 2:
                tqdm.write(
                    f"Validation loss did not improve for 2 epochs. Reducing learning rate and restoring best model weights.")
                self.load_weights()
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                epochs_since_improvement = 0

        total_time = time.time() - total_start_time
        tqdm.write(f"Training completed in {total_time:.2f}s")
        loss_logger.log_all_losses(num_epochs=num_epochs, train_losses=train_losses, val_losses=val_losses)
        loss_logger.log_training_time(time=total_time)

    def save_weights(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_weights(self, weights_path=None):
        if weights_path is None:
            path = self.model_path
        else:
            path = weights_path
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def evaluate_model_detection(self, data_loader, classification_threshold=0.5, segmentation_threshold=0.2,
                                 min_box_area=0.01, max_box_area=0.8):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True
        predictions = []
        progress_bar_eval = tqdm(data_loader, desc=f"Evaluation")
        for idx, (images, _) in enumerate(progress_bar_eval):
            images = torch.stack([image.to(self.device) for image in images])
            output = torch.sigmoid(self.model(images))
            # Loop through each image and its outputs in the batch
            for img, probs in zip(images, output):
                # Find classes with probabilities above the threshold
                high_prob_indices = (probs >= classification_threshold).nonzero(as_tuple=True)[0]
                if len(high_prob_indices) != 0:
                    prediction = self.grad_cam_on_image(img, high_prob_indices, probs, segmentation_threshold,
                                                        min_box_area, max_box_area)
                # If no classes are predicted, return empty prediction
                else:
                    prediction = {
                        "image": img,
                        "cam_image": torch.tensor([]),
                        "cam_images": torch.tensor([]),
                        "threshold_cam": torch.tensor([]),
                        "threshold_images": torch.tensor([]),
                        "labels": torch.tensor([]),
                        "boxes": torch.tensor([]),
                        "scores": torch.tensor([]),
                        "pin_points": torch.tensor([]),
                        "pin_points_labels": torch.tensor([]),
                        "pin_points_scores": torch.tensor([]),
                        "pin_pointing_threshold": torch.tensor([]),
                        "pin_pointing_all_thresholds": torch.tensor([])
                    }

                predictions.append(prediction)
        return predictions

    def evaluate_model_classification(self, data_loader):
        self.model.eval()
        class_predictions = torch.zeros((len(data_loader), self.num_classes), device=self.device)

        with torch.no_grad():
            progress_bar_eval = tqdm(data_loader, desc="Evaluation")
            for idx, (images, _) in enumerate(progress_bar_eval):
                images = torch.stack([image.to(self.device) for image in images])
                output = torch.sigmoid(self.model(images))
                class_predictions[idx] = output

        return class_predictions

    def grad_cam_on_image(self, tensor_image, target_class_labels, class_probabilities, segmentation_threshold,
                          min_box_area, max_box_area):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

        if self.classifier_name == 'fpn':
            target_layers = [self.model.final_conv]
        elif self.classifier_name == 'vgg16':
            target_layers = [self.model.model.features[29]]
            # target_layers = [self.model.features[29]]

        cam_images = []
        labels_for_image = []
        cam_bboxes_for_image = []
        scores = []
        threshold_images = []
        pin_points_for_image = []
        pin_points_labels = []
        pin_points_scores = []
        thresholds_pin_pointing = []

        if isinstance(target_class_labels, int):
            target_class_labels = [target_class_labels]

        # Apply GradCAM++ for each predicted class label
        for i in target_class_labels:
            target_classes = [ClassifierOutputTarget(i)]
            cam = GradCAMPlusPlus(self.model, target_layers)
            tensor_image.requires_grad_(True)
            grayscale_cam = cam(tensor_image.unsqueeze(0), target_classes)

            grayscale_cam = grayscale_cam[0, :]
            img = tensor_image.detach().cpu()
            img = np.array(img.numpy())
            img = np.float32(img) / 255
            img = np.transpose(img, (1, 2, 0))
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            cam_images.append(cam_image)
            # Get bounding boxes
            bounding_boxes, threshold_cam = self.generate_bounding_boxes_from_cam(grayscale_cam,
                                                                                  threshold_ratio=segmentation_threshold,
                                                                                  min_box_area=min_box_area,
                                                                                  max_box_area=max_box_area)
            # Get pin points
            pin_points, threshold_pin_pointing = self.pin_point(greyscale_cam_image=grayscale_cam)
            cam_bboxes_for_image.extend(bounding_boxes)
            threshold_images.append(threshold_cam)
            pin_points_for_image.extend(pin_points)
            thresholds_pin_pointing.append(threshold_pin_pointing)
            # Use the class probability as score for each predicted bounding box
            for num_bbox_of_class_i in range(len(bounding_boxes)):
                labels_for_image.append(i)
                scores.append(class_probabilities[i])
            # Use the class probability as score for each predicted pin point
            for num_pin_points_of_class_i in range(len(pin_points)):
                pin_points_labels.append(i)
                pin_points_scores.append(class_probabilities[i])

        # Overlay the CAMs for bounding boxes for visualization purposes
        cam_arrays = [np.array(cam) for cam in cam_images]
        max_cam = np.maximum.reduce(cam_arrays)
        # Overlay the thresholded CAMs for bounding boxes for visualization purposes
        thresholds_arrays = [np.array(t) for t in threshold_images if len(t) > 0]
        max_threshold_images = np.maximum.reduce(thresholds_arrays)
        # Overlay the thresholded CAMs for pin points for visualization purposes
        thresholds_pin_pointing_arrays = [np.array(t) for t in thresholds_pin_pointing]
        max_threshold_pin_pointing = np.maximum.reduce(thresholds_pin_pointing_arrays)

        data_for_img = {}
        data_for_img['image'] = tensor_image
        data_for_img['cam_image'] = torch.tensor(max_cam)
        data_for_img['cam_images'] = torch.tensor(np.array(cam_arrays))
        data_for_img['threshold_cam'] = torch.tensor(max_threshold_images)
        data_for_img['threshold_images'] = torch.tensor(np.array(thresholds_arrays))
        data_for_img['labels'] = torch.tensor(labels_for_image)
        data_for_img['boxes'] = torch.tensor(cam_bboxes_for_image)
        data_for_img['scores'] = torch.tensor(scores)
        data_for_img['pin_points'] = torch.tensor(pin_points_for_image)
        data_for_img['pin_points_labels'] = torch.tensor(pin_points_labels)
        data_for_img['pin_points_scores'] = torch.tensor(pin_points_scores)
        data_for_img['pin_pointing_threshold'] = torch.tensor(max_threshold_pin_pointing)
        data_for_img['pin_pointing_all_thresholds'] = torch.tensor(np.array(thresholds_pin_pointing_arrays))

        return data_for_img

    def generate_bounding_boxes_from_cam(self, greyscale_cam_image, threshold_ratio=0.2, min_box_area=0.01,
                                         max_box_area=0.8, max_recursion_depth=5,
                                         current_depth=0):
        if current_depth > max_recursion_depth:
            return [], np.uint8([])  # Prevents infinite recursion

        # Normalize CAM with IVR
        greyscale_cam_normalized = self.ivr_normalization(greyscale_cam_image)

        # Apply threshold
        threshold_value = threshold_ratio * np.max(greyscale_cam_normalized)
        _, binary_thresh = cv2.threshold(greyscale_cam_normalized, threshold_value, 1, cv2.THRESH_BINARY)

        # Detect contours
        contours, _ = cv2.findContours(np.uint8(binary_thresh * 255), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        image_area = greyscale_cam_image.shape[0] * greyscale_cam_image.shape[1]
        bounding_boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Skip very small bounding boxes
            if area < min_box_area * image_area:
                continue

            # If a bounding box is too large, increase the threshold and try again
            if area > max_box_area * image_area:
                return self.generate_bounding_boxes_from_cam(greyscale_cam_image, threshold_ratio=threshold_ratio + 0.1,
                                                             max_recursion_depth=max_recursion_depth,
                                                             current_depth=current_depth + 1)

            bounding_boxes.append([x, y, x + w, y + h])

        return bounding_boxes, np.uint8(binary_thresh * 255)

    def ivr_normalization(self, cam, percentile=30):
        # Compute the p-percentile value
        pct_value = np.percentile(cam, percentile)

        # Subtract the percentile value from the CAM
        cam_adjusted = cam - pct_value

        # Normalize the CAM by its new maximum value
        cam_max = np.max(cam_adjusted)
        cam_normalized = cam_adjusted / cam_max

        return cam_normalized

    def pin_point(self, greyscale_cam_image, top_percent=0.5, min_contour_area=0.01):
        # Normalize CAM with IVR
        greyscale_cam_normalized = self.ivr_normalization(greyscale_cam_image)

        # Determine the threshold value to keep the top 'top_percent' of the activation areas
        threshold_value = float(1 - top_percent) * np.max(greyscale_cam_normalized)

        # Apply threshold
        _, binary_thresh = cv2.threshold(greyscale_cam_normalized, threshold_value, 1, cv2.THRESH_BINARY)

        # Find contours in the binary threshold image
        contours, _ = cv2.findContours(np.uint8(binary_thresh * 255), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        keypoints = []

        image_area = greyscale_cam_image.shape[0] * greyscale_cam_image.shape[1]
        for cnt in contours:
            # Calculate the area of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Skip small contours
            if area < min_contour_area * image_area:
                continue

            # Create a mask for the contour
            mask = np.zeros(greyscale_cam_normalized.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, (1), thickness=cv2.FILLED)

            # Use the mask to find the highest value inside the contour
            masked_cam = greyscale_cam_normalized * mask
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_cam)

            keypoints.append(max_loc)

        return keypoints, binary_thresh
