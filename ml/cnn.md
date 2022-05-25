# Convolutional Neural Network

## components

convolution

- use filters to find feature
- takes fewer parameters
- terms
  - in-channel
  - out-channel
  - no-padding
    - `valid` option in TF
  - zero-padding
    - `same` option in TF
  - stride

sub-sampling

- getting smaller images
- types
  - max pooling 
  - average pooling

local contrast normalization

fully connected layer

- takes many parameters

## history

https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

- LeNet-5 (1998)
- AlexNET (2012)
  - 처음으로 NN사용해서 1등
  - 8 layers
  - GPU memory 가 부족해서 네트워크를 2개로 나눠서 하다가 다시 합침.
  - ReLU 사용. (이전에는 얼마 안썼음.)
  - LRN(Local Response Normalization)
    - 일종의 regularization 테크닉
    - 일정 부분만 높게 activation 시키고 나머지는 낮게 하고 싶음.
  - regularization
    - data augmentation
      - label preserving augmentation
      - cropping(256x256 => 224x224)
      - Color variation
        - 학습 데이터에서 허용할 수 있는 만큼의 노이즈를 넣어서 추가 데이터를 만듬
    - dropout
      - 약간 특이하게 output 에 그냥 0.5 를 곱했음.
      - 별로여서 이후에는 이렇게 안함.
- VGGNet(2014)
  - 16 or 19 layers
  - stride 1
  - 3x3 convolution 을 사용함.
  - 간단하게 좋은 성능을 냄.
- GoogLeNet/Inception (2014)
  - 22 layers
  - inception module
    - 1x1 convolution
    - 1x1 and 3x3 convolution
    - 5x5 and 1x1 convolution
    - 3x3 max pooling and 1x1 convolution
  - parameter를 줄여주기 위해 1x1 convolution을 추가함.
  - 갈림길 때문에 입력 이미지에 미치는 reception field 영역이 dynamic 하게 만들어져서 성능 향상에 도움이 됨.
- ResNet(2015)
  - 152 layers
  - 여러 대회에서 1등을 했음.
  - residual connection
    - + 연산하는 루트를 만듬.
      - 입력과 출력 사이에 차이 많을 학습 하겠다는 의미가 됨.
      - degradation을 해결.
        - layer 가 deep 해지는데 성능이 좋아지지 않는 문제를 어느정도 해결함.
      - 대신 입력과 출력의 dimension 이 항상 같아야 함.
- Inception v4, Inception-ResNet (2016, by google)
  - inception module 안에서 1x7 and 7x1 convolution 을 끼워 넣어서.. 파라미터를 더 많이 줄임.
  - residual connection을 추가함.
