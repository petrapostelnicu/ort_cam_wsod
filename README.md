# Object Roughly There (ORT): CAM - based Weakly Supervised Object Detection

### Reducing the labelling efforts for deep learned object detectors

------------------------------------

This is the repository for [this paper](todo), submitted as my Bachelor Thesis at Delft University of Technology.

--------------------

*Highly performing object detectors require large training datasets, which entail class and bounding box annotations. To
reduce the labelling effort of curating such datasets, Weakly Supervised Object Detection is concerned with training
object detectors from only class labels. The most performant weakly supervised detectors (MIL-based) have high inference
times, while faster methods (CAM-based) have been primarily studied in the context of localizing just one object in an
image. This research proposes an extension to weakly supervised CAM-based detectors that allows them to detect multiple
objects in an image and asseses their performance at localizing the full extent of objects with bounding boxes, as well
as their general location with pin-points. VGG16 and a novel FPN-based classifier are experimented with as the backbone
of the network, followed by GradCAM++ which indicates through heatmaps the locations of the objects predicted by the
classifiers. Additionally, the proposed method is used to create pseudo-labels on which any fully supervised detector
could be trained on. Results show that while the proposed method is not suitable for detecting the full extent of
objects, it can accurately pin-point their general location in near real-time, thus showing the Object is Roughly
There (ORT).*

### Architecture

![img](/images/pipeline.png)

### Visualizations

![img](/images/visual_results_comaprisson.png)

-------------------------

### Installation

1. Clone the repository

```sh
git clone hhttps://github.com/petrapostelnicu/ort_cam_wsod.git & cd ort_cam_wsod
```

2. Install the required libraries (make sure you have Python and pip already installed)

```sh
pip install -r requirements.txt
```

3. Create a ```pretrained_models``` folder inside the root of the project and a ```data``` folder outside the root of
   the project. Create a folder called ```pascal_voc_2007``` inside the ```data``` folder.
   The structure should look like so:

```
...
├── data/
    └── pascal_voc_2007/
└── ort_cam_wsod/
    ├── datasets/
    ├── loggers/
    ├── logs/
    ├── models/
    ├── pretrained_models/
    ├── pytorch_grad_cam
    ├── scripts/
    ├── utils/
    └── main.py
```

-------------------

### Usage

**Train** the ORT, by training its backbone classifier:
\
Note that this will automatically download the PASCAL VOC 2007 dataset to the ```data``` folder.

- VGG16

```
python main.py train_ort --classifier_name vgg16
```

- FPN P5 (you can switch between the different fpn feature maps using ```--fpn_layer```)

```
python main.py train_ort --classifier_name fpn --fpn_layer 5
```

**Evaluate** the ORT:
\
Note that this provides an evaluation on the whole test set, as well as creates the pseudo-labels
for the two-stage method. If you want to perform an evaluation on the reduced test set, modify the
```run()``` method in the ```scripts/evaluate_ort.py``` file to only run ```evaluate_test_set_reduced```.

- ORT-VGG16

```
python main.py evaluate_ort --classifier_name vgg16
```

- ORT-FPN P5 (you can switch between the different fpn feature maps using ```--fpn_layer```)

```
python main.py evaluate_ort --classifier_name fpn --fpn_layer 5
```

**Visualize** the ORT predictions:
\
Note that this provides visualizations for all the images in the test set that were evaluated.
If you want to see just a few images, perform the evaluation on the reduced test set.

- ORT-VGG16

```
python main.py visualize_cam_pseudo --classifier_name vgg16
```

- ORT-FPN P5 (you can switch between the different fpn feature maps using ```--fpn_layer```)

```
python main.py visualize_cam_pseudo --classifier_name fpn --fpn_layer 5
```

**Train two-stage method** by training the Faster-RCNN on the pseudo-labels
generated with ORT:
\
Note that for this to work, the evaluation has to have been performed in full.

- ORT-VGG16 + Faster-RCNN

```
python main.py train_faster_rcnn_on_pseudo --classifier_name vgg16
```

- ORT-FPN P5 + Faster-RCNN(you can switch between the different fpn feature maps using ```--fpn_layer```)

```
python main.py train_faster_rcnn_on_pseudo --classifier_name fpn --fpn_layer 5
```

**Evaluate two-stage method**:

- ORT-VGG16 + Faster-RCNN

```
python main.py evaluate_faster_rcnn --from_pseudo True --classifier_name vgg16
```

- ORT-FPN P5 + Faster-RCNN(you can switch between the different fpn feature maps using ```--fpn_layer```)

```
python main.py evaluate_faster_rcnn --from_pseudo True --classifier_name fpn --fpn_layer 5
```

**To use the resized images** (224x224), provide the flag ```--resize resize```. By default the flag is set
to ```no_resize```.
