import time

from torch.utils.data import DataLoader

from datasets import PascalVOC
from models import FasterRCNN


def run():
    full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test')
    reduced_set = full_dataset.get_reduced()
    data_loader = DataLoader(reduced_set, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    model = FasterRCNN(num_classes=21, model_path='pretrained_models/faster_rcnn_2007_final_no_resize.pth')
    total_start_time = time.time()
    predictions = model.evaluate_model(data_loader=data_loader)
    total_time = time.time() - total_start_time
    print(f'Total time:{total_time}')
    print(f'Average time: {total_time / 100}')
