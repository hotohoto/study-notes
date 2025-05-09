# Linear model with PyTorch

## numpy

- https://jalammar.github.io/visual-numpy/

## linear model

- cost function / loss function
- hypothesis function

## forward propagation

## back propagation

- partial derivative
- chain rule

## linear model with regularization

## classification

- sigmoid function

## homeworks

### weigh-height 모델 만들기

(1) 데이터 파일 다운로드 받기

원하는 폴더로 이동 후 `wget https://xxx.xxxx.xxx/xxxx/xxx.csv` 하면 그 폴더에 파일 받아짐.
`wget https://raw.githubusercontent.com/hotohoto/ml_course/master/datasets/weight-height.csv`

`ls`와 `head` 명령을 사용해서 파일이 잘 받아졌는지 확인한다.

(2) jupyter notebook에서 pandas 파일 읽어서 확인하기.

(titanic 코드 참고.)

- jupyter notebook을 연다.
- `df = pd.read_csv('다운로드 받은 파일 경로')`
- 앞에 10개의 데이터만 표시해본다. (head())
- 평균 표준편차 등을 확인해본다. (describe())
- 빠진 데이터가 있지 않는지 확인해본다. (isnull())
- 성별에 대한 히스토그램을 그려본다.

(3) 데이터 split

(titanic 코드 참고)

- training set 80%
- test set 20%
로 분할한다.

(4) 데이터 변환

(pandas에서 한 후 최종적으로 numpy로 변환한다.)

- 성별을 남자는 0 여자는 1로 변환한다. (training set, test set 모두 적용)
- 성별과 키는 표준화한다. (training set의 평균과 표준편차 이용하여 평균 0 표준편차 1로 만듬. test set도 변환)
- describe() 를 호출해서 확인한다.
- 변환된 training set과 test set 을 아래 처럼 numpy 로 변환한다.

```py
x_train = train_data_x.values
y_train = train_data_y.values
x_test = test_data_x.values
y_test = test_data_y.values
```

(5) 학습

- 적당한 learning rate, epoch 값을 변수로 만들어 사용한다. (적당한 값이 몇인지 실험해보기)
- 미분은 수치적인 방법으로 해서 파라미터 업데이트 한다.
- 최종 weight 와 bias 를 화면에 출력해본다.
- 한 epoch 이 끝날때 마다 학습 데이터에 대한 mse 를 화면에 출력한다.
- regression 문제이므로 sigmoid 함수등은 필요하지 않다.

(6) test set 으로 성능 확인

- test set에서 MSE 를 출력해본다.

(7) 추가 문제

위와 동일한 과정으로 iris 데이터셋을 사용해서 꽃 받침 너비, 길이, 꽃입 너비 길이 를 사용해서 꽃의 품종을 예측하는 모델을 만들어보자.

- 데이터셋: https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv
- 품종인 `species` 필드는 3개의 one-hot encoding 된 필드로 만든다.
- softmax를 사용해서 모델을 구성한다.
- mse가 아닌 cross entropy 를 이용해서 학습할 수 있도록 한다.
- accuracy 로 성능을 평가한다.

## References

- https://www.edwith.org/aipython/lecture/24509/
- [Gradient descent and learning rate](http://www.onmyphd.com/?p=gradient.descent)
- [Simple linear regression lecture (slides)](https://www.sjsu.edu/faculty/guangliang.chen/Math261a/Ch2slides-simple-linear-regression.pdf)
