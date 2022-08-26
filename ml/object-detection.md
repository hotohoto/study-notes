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

- anchor based detectors
  - regressing from a box
- anchor free detectors
  - regressing from a point
  - types
    - key point based
      - CornerNet
      - Grid-RCNN for two-stage models
      - RepPoints
    - center based
      - YOLO
      - GA-RPN for Faster R-CNN
      - FSAF for RetinaNet
      - FCOS
      - FoveaBox

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

- DICE
  - TODO
- IOU
  - https://silhyeonha-git.tistory.com/3
  - GIOU
- precision
  - per object given an IOU threshold
- recall
  - per object given an IOU threshold
- F1
  - corresponds to a single point in a PR curve
  - a function of a threshold
- AUC
  - area under a PR curve
- AP
  - the sum of precisions over the equally gapped recall values regarding a PR curve
  - $\int P(R) dR$
    - $P$: Precision
    - $R$: Recall
- mAP
  - mean Average Precision
  - average AP over all classes
  - $\text{mAP} = {1 \over N} \sum\limits_{i=0}^N \text{AP}_i$
  - https://www.v7labs.com/blog/mean-average-precision
- COCO mAP
  - average mAP over a range of IOU thresholds
  - step size
    - recall values: 0.01
    - IOU thresholds: 0.05
- Confusion matrix in bounding box detection
  - there exist confidence/IOU thresholds
  - calculate IOU values for the combinations of each true label and each prediction
  - if there are multiple labels matching for a single predicted object, we take the object of the biggest IOU among them
  - if there are multiple objects detected for a single label, we take the object of the biggest IOU among them
  - we don't take a single prediction nor a single true label as multiple occurance, so we filter out less important overlaps correctly
  - diagonal count:
    - for each true label object, if we detect it as a correct class we take it into account here
  - non diagonal count
    - for each true lable object, if we detect it as a wrong class we take it into account here
  - background FP
    - for each true label object, if we don't detect it at all we take it account here
  - background FN
    - for each detection occurance, if there is no true label there at all we take it account here

## Tips

- if a big object is not detected
  - check the receptive field
- if a small object is not detected
  - consider using focal loss

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
  - new model configuration file
  - https://github.com/ultralytics/yolov5
  - https://blog.paperspace.com/train-yolov5-custom-data/
  -

(2020)

- Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection
  - CVPR 2020
  - proposes Adaptive Tranining Sample Selection (ATSS)
    - automatically select positive and negative samples according to statistical characteristics of object

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
  - auto learning bounding box
  - 16bit floating point
  - Cross Stage Partial (CSP) backbone
  - Path Aggregation Network (PANet) is used for the neck
  - SGD (default), Adam (alternatively)
  - loss funciton
    - $
\begin{aligned}
\mathcal{L}=& 1-\text{IoU}+\frac{\rho^{2}\left(b, b^\text{gt}\right)}{c^{2}}+\alpha v-\\
& \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} I_{i j}^{o b j}\left[\hat{C}_{i} \log \left(C_{i}\right)+\left(1-\hat{C}_{i}\right) \log \left(1-C_{i}\right)\right]-\\
& \lambda_\text{noobj} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} I_{i j}^\text{noobj}\left[\hat{C}_{i} \log \left(C_{i}\right)+\left(1-\hat{C}_{i}\right) \log \left(1-C_{i}\right)\right]-\\
& \sum_{i=0}^{S^{2}} I_{ij}^\text{obj} \sum_{c \in \text {classes}}\left[\hat{p}_{i}(c) \log \left(p_{i}(c)\right)+\left(1-\hat{p_{i}}(c)\right) \log \left(1-p_{i}(c)\right)\right]
\end{aligned}
$
    - cross entropy terms
    - CIoU
      - $1 - \text{IoU} + \text{DIoU} + \alpha v$
      - taking aspect ratio into account
      - DIoU
        - $\rho^2(b, b^\text{gt}) \over c^2$
          - $\rho^2(\cdot, \cdot)$
            - Euclidean distance
          - $c$
            - the minimum length of diagonal enclosing both boxes
      - $v = {4 \over \pi}(\arctan{w^\text{gt} \over h^\text{gt}} - \arctan{w \over h})^2$
      - $\alpha = {v \over (1 - \text{IoU}) + v}$
    - Focal loss (optional)
  - references:
    - https://towardsai.net/p/computer-vision/yolo-v5%e2%80%8a-%e2%80%8aexplained-and-demystified
    - https://blog.roboflow.com/yolov5-improvements-and-evaluation/


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

- An intriguing failing of convolutional neural networks and the CoordConv solution
  - NeurIPS 2018
  - vanilla CNNs are not good at coordinates transformation
  - CoordConv
    - giving convolution access to its own input coordinates
      - through the use of extra coordinate channels
    - good at MNIST, Atari games
  - TODO

- nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation
  - https://arxiv.org/abs/1809.10486
  - (Revised version)
    - nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
    - Nature Methods 2020
    - https://www.nature.com/articles/s41592-020-01008-z
  - the Medical Segmentation Decatholon Challenge
    - 7 open datasets with 7 holdout sets
    - 3 hidden datasets
    - supposed to create an automatic method
    - the method is evaluated on the 3 hidden datasets
  - network architectures
    - 2D U-Net
      - generates full resolution segmentations
    - 3D U-Net
      - generates full resolution segmentations
      - maybe for patch images
    - U-Net Cascade
      - (1st stage) 3D U-Net low resolution
      - (2nd stage) 3D U-Net high resolution refinement
    - (modifications)
      - instance normalization instead of BN
      - leaky ReLU instead of ReLU
  - preprocessing
    - cropping
    - resampling
    - normalization
  - training
    - 5-fold CV
    - loss
      - $L_\text{dice} + L_\text{CE}$
    - optimizer setting
      - ADAM
      - early stopping
    - data augmentation
      - (on the fly)
        - random rotation
        - random scaling
        - random slastic defromations
        - gamma correction
        - mirroring
    - patch sampling
      - make sure that for each batch 1/3 samples have at least one random foreground class
  - inference
    - patch-based strategy
    - ensembling across test-time augmentations and 5 networks
  - post-processing
    - enforcing single connected components if applicable
  - ensemble models and model selection
    - candidates
      - 2D
      - 3D
      - Cascade
      - 2D + 3D
      - 2D + Cascade
      - 3D + Cascade
    - based on CV scores on the training set

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

- 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
  - MICCAI 2016
  - https://arxiv.org/abs/1606.06650
  - TODO

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
  - loss function
    - $
\begin{aligned}
\mathcal{L}(\hat{z}, z) &=\lambda_{\text {coord }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {obj }}\left[\left(x_{i}-\hat{x}_{i}\right)^{2}+\left(y_{i}-\hat{y}_{i}\right)^{2}\right] \\
&+\lambda_{\text {coord }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {obj }}\left[\left(\sqrt{w_{i}}-\sqrt{\hat{w}_{i}}\right)^{2}+\left(\sqrt{h_{i}}-\sqrt{\hat{h}_{i}}\right)^{2}\right] \\
&+\sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {obj }}\left(C_{i}-\hat{C}_{i}\right)^{2}+\lambda_{\text {noobj }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text {noobj }}\left(C_{i}-\hat{C}_{i}\right)^{2} \\
&+\sum_{i=0}^{S^{2}} \mathbb{1}_{i}^{\text {obj }} \sum_{c \in \text { classes }}\left(p_{i}(c)-\hat{p}_{i}(c)\right)^{2}
\end{aligned}
$

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
