import time

import torch
from torchvision import transforms
from torchvision.transforms.functional import resize
from ultralytics import YOLO

from datasets import PascalVOC


# YOLOv8 only accepts images with sizes that are multiples of 32.
def resize_to_multiple_of_32(image):
    # Get original dimensions
    w, h = image.size
    # Calculate new dimensions
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    # Resize image
    return resize(image, (new_h, new_w))


def run():
    transform = transforms.Compose([
        transforms.Lambda(resize_to_multiple_of_32),
        transforms.ToTensor()
    ])

    full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test', transform=transform)
    reduced_set = full_dataset.get_reduced()
    device = torch.device('cpu')
    model = YOLO('yolov3u.pt')
    model.to(device)
    # Measure inference time
    inference_times = []

    for (img, target) in reduced_set:
        img = img.unsqueeze(0).to(device)  # Add batch dimension
        start_time = time.time()
        results = model(img, device=device)
        end_time = time.time()
        inference_times.append(end_time - start_time)

    # Calculate average inference time
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f'Average time: {average_inference_time}')
