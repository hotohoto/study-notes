# Instance Segmentation

## Terms

- FCN(Fully Convolutional Network)
- Prototype
  - Bag of visual Words
    - https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision

## History

- R-CNN (2014)
  - multi-stage model
    - selective search
    - (region proposal)
    - cnn
    - (features)
    - svm
  - each stage should be trained separately
  - very slow training / inference
- SPPnet (2014)
  - Spatial Pyramid Pooling(SPP)
- Fast R-CNN (2015)
  - almost single stage model
  - R-CNN + SPPnet
  - ROI pooling layer
    - single level SPP
    - (r, c, h, w)
  - bounding box regression
  - still slow inference
- Faster R-CNN (2015-2016)
  - single stage model
  - introduced region proposal network
    - similar to attention
    - regression with fully connected network
  - NMS(Non-Maximum Suppression)
    - search the local maximum value and suppress others
  - mAP(mean Average Precision)
  - up to 17 fps
- YOLO (2015-2016)
  - https://arxiv.org/abs/1506.02640
  - You only look onece
  - 7x7 로 이미지 자름
  - 7x7x30 prediction tensor를 생성
    - 30 = B * 5 + C
    - x, y, w, h, confidence,
    - C = number of class
  - grid 1개에 bounding box 가 2개까지 나올 수 있음.
- SSD(2015-2016)
  - single shot detector
  - object detection
- YOLOv2 (or YOLO 9000, 2016)
  - 1 stage detector
- `Mask-RCNN` (2017)
  - 2 stage detector
    - RPN(Region Proposal Network)
    - segmentation for each bounding box
  - instance segmentation
  - influenced by Faster R-CNN
- Soft NMS (2017)
  - https://arxiv.org/abs/1704.04503
  - https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
  - An improved version of Non-maximum Suppression (NMS)
- `YOLACT` (2019)
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
- `YOLACT++` (2019)

## Questions

- repooling
  - explicit feature localization


## References

- [양우식 - Fast R-CNN & Faster R-CNN](https://youtu.be/Jo32zrxr6l8)
- [YOLACT 설명 및 정리 - (1)](https://ganghee-lee.tistory.com/42)
- [Evaluating Object Detection Models: Guide to Performance Metrics](https://manalelaidouni.github.io/manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html)
- [Details about faster RCNN for object detection - NMS](https://xiaoyuliu.github.io/2017/12/27/faster-rcnn-details/)
