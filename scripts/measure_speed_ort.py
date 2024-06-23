import time

from torch.utils.data import DataLoader

from datasets import PascalVOC
from models import ORT


def run(classifier_name, fpn_layer='0'):
    full_dataset = PascalVOC(root='../data/pascal_voc_2007', image_set='test')
    reduced_set = full_dataset.get_reduced()
    data_loader = DataLoader(reduced_set, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    if classifier_name == 'vgg16':
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_no_resize.pth'
    else:
        model_path = f'pretrained_models/{classifier_name}_classifier_pascal_2007_final_layer{fpn_layer}_no_resize.pth'
    model = ORT(num_classes=21, model_path=model_path,
                classifier_name=classifier_name, fpn_layer=fpn_layer)
    model.load_weights()
    total_start_time = time.time()
    predictions = model.evaluate_model_detection(data_loader=data_loader, classification_threshold=0.5)
    total_time = time.time() - total_start_time
    print(f'Total time:{total_time}')
    print(f'Average time: {total_time / 100}')
