import argparse

from scripts import evaluate_faster_rcnn
from scripts import evaluate_ort
from scripts import measure_speed_faster_rcnn
from scripts import measure_speed_ort
from scripts import measure_speed_yolo
from scripts import train_faster_rcnn
from scripts import train_faster_rcnn_on_pseudo
from scripts import train_ort
from scripts import visualize_cam_pseudo
from scripts import visualize_faster_rcnn_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different scripts based on the provided arguments")
    parser.add_argument('script_name', type=str, help='The name of the script to run')
    parser.add_argument('--classifier_name', type=str, help='The name of the classifier to use', default='vgg16')
    parser.add_argument('--fpn_layer', type=str, help='The layers in the fpn to use', default='0')
    parser.add_argument('--from_pseudo', type=bool, help='Whether to train the Faster RCNN from pseudo labels',
                        default=False)
    parser.add_argument('--resize', type=str, help='Whether to resize the images to 224x224 or not',
                        default='no_resize')

    args = parser.parse_args()

    if args.script_name == 'train_faster_rcnn':
        train_faster_rcnn(resize=args.resize)
    elif args.script_name == 'evaluate_faster_rcnn':
        evaluate_faster_rcnn(from_pseudo=True, model_name=args.classifier_name, fpn_layer=args.fpn_layer,
                             resize=args.resize)
    elif args.script_name == 'visualize_cam_pseudo':
        visualize_cam_pseudo(classifier_name=args.classifier_name, fpn_layer=args.fpn_layer, resize=args.resize)
    elif args.script_name == 'evaluate_ort':
        evaluate_ort(classifier_name=args.classifier_name, fpn_layer=args.fpn_layer, resize=args.resize)
    elif args.script_name == 'train_ort':
        train_ort(classifier_name=args.classifier_name, fpn_layer=args.fpn_layer, resize=args.resize)
    elif args.script_name == 'visualize_faster_rcnn_output':
        visualize_faster_rcnn_output(from_pseudo=True, model_name=args.classifier_name, fpn_layer=args.fpn_layer,
                                     resize=args.resize)
    elif args.script_name == 'train_faster_rcnn_on_pseudo':
        train_faster_rcnn_on_pseudo(model_name=args.classifier_name, fpn_layer=args.fpn_layer, resize=args.resize)
    elif args.script_name == 'measure_speed_ort':
        measure_speed_ort(classifier_name=args.classifier_name, fpn_layer=args.fpn_layer)
    elif args.script_name == 'measure_speed_faster_rcnn':
        measure_speed_faster_rcnn()
    elif args.script_name == 'measure_speed_yolo':
        measure_speed_yolo()
    else:
        print(f"Script '{args.script_name}' not found.")
