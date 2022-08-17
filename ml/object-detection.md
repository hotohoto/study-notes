# Object detection

## terms

- two-stage detectors
  - R-CNN
  - SPPNet
  - Fast R-CNN
  - Faster R-CNN
  - Pyramid Networks
- one-stage detectors
  - YOLO
  - SSD
  - Retina-Net

- repooling
  - explicit feature localization
- Region Proposal Network (RPN)
  - originated from Faster R-CNN
- Non-Maximum Suppression (NMS)
- ROI pooling
  - TODO https://erdem.pl/2020/02/understanding-region-of-interest-ro-i-pooling
  - introduced by Fast R-CNN
  - quantize area by dropping decimal part apply max-pooling or another pooling method for the area
- ROI warp
  - quantize area in the middle
- ROI align
  - https://erdem.pl/2020/02/understanding-region-of-interest-part-2-ro-i-align
  - quantize area not dropping ROI
  - gives better precision on average

- soft ROI pooling
- backbone
  - model to translate a image to feature map
- feature map
  - the same as activation map

- FCN(Fully Convolutional Network)
- FPN(Feature Pyramid Network)
- Prototype
  - Bag of visual Words
    - https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision
- repooling
  - explicit feature localization

## Evaluation

- IOU
  - https://silhyeonha-git.tistory.com/3
  - GIOU
- precision
  - per object given an IOU threshold
- recall
  - per object given an IOU threshold
- F1
  - a single point in a PR curve
  - a function of a threshold
- AUC
  - area under a PR curve
- AP
  - the sum of precisions over the equally gapped recall values regarding a PR curve
  - $\int P(R) dR$
    - $P$: Precision
    - $R$: Recall
- mAP
  - mean Average Precision (mAP)
  - average AP over all classes
  - $\text{mAP} = {1 \over N} \sum\limits_{i=0}^N \text{AP}_i$
  - https://www.v7labs.com/blog/mean-average-precision
- COCO mAP
  - average mAP over a range of IOU thresholds
  - step size
    - recall values: 0.01
    - IOU thresholds: 0.05

## Papers

(2022)

- DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection
  - https://arxiv.org/abs/2203.03605
  - https://github.com/IDEACVR/DINO
  - goals
    - faster training
    - performance
    - scalability
  - novel methods
    - contrastive denoising training
      - TODO
    - mixed query selection
      - TODO
    - look forward twice for box prediction (?)
      - TODO
  - other adoptions
    - dynamic anchor boxes
      - TODO
    - deformable attention
      - TODO
  - model architecture
    - a backbone
    - a multi-layer Transformer encoder
    - a multi-layer Transformer decoder
    - multiple prediction heads
  - predecessors
    - DETR
    - Deformable DETR
    - DN-DETR
    - DAB-DETR
  - datasets
    - COCO
    - Object365

- YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object
detectors
  - https://arxiv.org/abs/2207.02696v1


(2021)

- (YOLOv6)
  - no paper published
  - https://github.com/meituan/YOLOv6

- (YOLOv5)
  - no paper published
  - no architecture change
  - Flexible control of model size (?)
  - Hardswish activation function (?)
  - https://github.com/ultralytics/yolov5
  - https://blog.paperspace.com/train-yolov5-custom-data/

(2020)

- Deformable DETR: Deformable Transformers for End-to-End Object Detection
  - address issues on DETR
    - faster training
    - better at small object detection

- End-to-End Object Detection with Transformers
  - DETR
  - https://arxiv.org/abs/2005.12872
  - CNN + Transformer(Encoder/Decoder)
  - bipartite matching by Hungarian algorithm
    - NMS not used
  - GIoU
  - cons
    - also good at panoptic segmentation
  - cons
    - slow training convergence due to decoder cross-attention
    - bad at small object detection
    - the meaning of queries is unclear

- PP-YOLO: An Effective and Efficient Implementation of Object Detector
  - https://arxiv.org/abs/2007.12099

- YOLOv4: Optimal Speed and Accuracy of Object Detection
  - https://arxiv.org/abs/2004.10934
  - Spatial Pyramid Pooling (SPP)
  - Mish activation function
    - [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681)
  - Data augmentation
    - Mosaic
      - place multiple images within an image
    - Mixup
      - [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
  - Generalized Intersection over Union (GIOU) for loss function

(2019)

- `YOLACT`
  - https://arxiv.org/abs/1904.02689
  - no explicit localization
  - single stage model with 2 subtask branch
    - input
    - ResNet 101 - C1, ..., C5
    - Feature Pyramid Network - P3, ..., P7
      - https://arxiv.org/abs/1612.03144
    - (branch)
      - Protonet(prototype mask)
        - generating a dictionary of non-local prototype masks over the entire image
      - Prediction Head
        - Fast NMS
        - predicting a set of linear combination coefficients per instance
    - (multiplication)
    - crop
    - threshold
  - https://github.com/dbolya/yolact
- `YOLACT++`

(2018)

- YOLOv3: An Incremental Improvement
  - https://arxiv.org/abs/1804.02767
  - Darknet-53

(2017)

- `Mask-RCNN`
  - two stage detector
    - RPN(Region Proposal Network)
    - segmentation for each bounding box
  - instance segmentation
  - influenced by Faster R-CNN
- Soft NMS
  - https://arxiv.org/abs/1704.04503
  - https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
  - An improved version of Non-maximum Suppression (NMS)

(2016)

- YOLO9000: Better, Faster, Stronger
  - https://arxiv.org/abs/1612.08242
  - YOLOv2
  - Batch Normalization
  - High-resolution classifier
  - Darknet-19 instead of GooleNet

- Faster R-CNN
  - single stage model
  - introduced region proposal network
    - similar to attention
    - regression with fully connected network
  - NMS(Non-Maximum Suppression)
    - search the local maximum value and suppress others
  - mAP(mean Average Precision)
  - up to 17 fps

- You Only Look Once: Unified, Real-Time Object Detection
  - YOLOv1
  - https://arxiv.org/abs/1506.02640
  - one stage detector
  - Create 7 x 7 grid
  - Each grid can hold 2 bounding boxes
  - A model outputs 7x7x30 as its prediction
    - 30 = B * 5 + C
    - x, y, w, h, confidence,
    - C = number of class
  - For each grid cell, there can be up to 2 bounding boxes

- SSD
  - single shot detector
  - object detection

(2015)

- Fast R-CNN
  - almost single stage model
  - R-CNN + SPPnet
  - ROI pooling layer
    - single level SPP
    - (r, c, h, w)
  - bounding box regression
  - still slow inference

(2014)

- R-CNN
  - multi-stage model
    - selective search
    - (region proposal)
    - cnn
    - (features)
    - svm
  - each stage should be trained separately
  - very slow training / inference

- SPPnet
  - Spatial Pyramid Pooling(SPP)

## datasets

- COCO
  - test-dev
    - good for validating model scalability

## development evironment

### Setup CUDA

```bash
sudo apt install nvidia-cuda-toolkit
nvcc --version
```

### Download COCO via fiftyone

```bash
# install a prerequisite for fiftyone running with mongodb on ubuntu 22.04
sudo echo "deb http://security.debian.org/debian-security stretch/updates main" | sudo tee /etc/apt/sources.list.d/debian-security.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv AA8E81B4331F7F50
sudo apt update
sudo apt install libssl1.1

pip install fiftyone
fiftyone zoo datasets load quickstart
fiftyone app launch quickstart
```

### IceVision

- https://airctic.com/
- https://airctic.com/0.12.0/using_fiftyone_in_icevision/

## References

- [양우식 - Fast R-CNN & Faster R-CNN](https://youtu.be/Jo32zrxr6l8)
- [YOLACT 설명 및 정리 - (1)](https://ganghee-lee.tistory.com/42)
- [Evaluating Object Detection Models: Guide to Performance Metrics](https://manalelaidouni.github.io/manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html)
- [Details about faster RCNN for object detection - NMS](https://xiaoyuliu.github.io/2017/12/27/faster-rcnn-details/)
