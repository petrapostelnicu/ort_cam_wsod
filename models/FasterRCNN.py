import time

import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from loggers import LossLogger


class FasterRCNN:
    def __init__(self, num_classes, model_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print(f'Running on {torch.cuda.get_device_name(0)}')
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.to(self.device)
        self.model_path = model_path

    def train_model(self, data_loader_train, data_loader_val, loss_logger: LossLogger, num_epochs=10):
        total_start_time = time.time()  # Start timing here
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_since_improvement = 0

        # Training
        for epoch in range(num_epochs):
            progress_bar_train = tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
            train_running_loss = 0.0
            for train_images, train_targets in progress_bar_train:
                images = list(image.to(self.device) for image in train_images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in train_targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                train_running_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            average_train_loss = train_running_loss / len(data_loader_train)
            train_losses.append(average_train_loss)
            progress_bar_train.close()
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Training - Loss: {average_train_loss}")

            # Validation
            progress_bar_val = tqdm(data_loader_val, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
            val_running_loss = 0.0
            with torch.no_grad():
                for val_images, val_targets in progress_bar_val:
                    images = list(image.to(self.device) for image in val_images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in val_targets]

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_running_loss += losses.item()

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

    def save_weights(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_weights(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        print(f"Model loaded from {self.model_path}")

    def evaluate_model(self, data_loader):
        self.model.eval()
        outputs = []
        with torch.no_grad():
            progress_bar_eval = tqdm(data_loader, desc=f"Evaluation")
            for idx, (images, targets) in enumerate(progress_bar_eval):
                images = list(img.to(self.device) for img in images)
                outputs.extend(self.model(images))
        return outputs
