# Colab performance test (naive)

- 모델
  - MNIST CNN 분류 문제 (Classification)
  - [소스코드](https://github.com/pytorch/examples/tree/master/mnist)
  - CUDA 항목 제외하고 모두 디폴트 설정. (epoch: 10)
- Colab
  - PyTorch v1.1 설치되어 있었음.
- PC
  - 2019 Samsung i7 notebook (no GPU)
  - PyTorch v1.2

| |Colab-CPU|Colab-GPU|PC|
|-|-|-|-|
|걸린시간(sec)|550.8|194.0|353.1|

결론:

Colab은 꽤 쓸만한 노트북 한대를 머신러닝 공부용으로 제공해주는 효과가 있다. (모델에 따라 다르겠지만 개인 노트북에 비해 54% 정도로 시간 단축을 기대할수도 있다.)

---

기타:

조만간 Colab에서 PyTorch + TPU를 사용해볼수 있으면 좋겠네요.

> Today, we're pleased to announce that engineers on Google's TPU team are actively collaborating with core PyTorch developers to connect PyTorch to Cloud TPUs.

[원문](https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud)
