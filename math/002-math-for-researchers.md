TODO

- modification of derivative operators
- EM algorithm



# 1. Hilbert space behind Heat equation and Fourier series

https://youtu.be/5ZjIMjWV3O0



(주요 내용)

- $l < \infty$ 에서 열방정식을 풀면 푸리에 급수가 나온다. 
- 푸리에 급수는 함수를 vector로 본 hilbert space 에서 basis 들의 합이다.

(Remarks)

- complete inner product space
  - complete - 값이 비어있지 않은..



# 2. Conditional expectation and Fourier series

https://youtu.be/WQyP_NCtXZI

- random variable 은 함수이다.
- random variable 을 vector로 보고 pdf 를 사용해 inner product를 만들면 geometrical 한 구조가 생긴다.
- conditional expectation 의 결과는 random variable이다.
- conditional expectation 은 condition 에 해당하는 random variable 들 쪽으로 정사영하는 행위에 해당한다.
- sample space의 각 possible outcome들의 indicator function 들을 random variable로 만들면 해당 vector space에서의 orthogonal basis들을 찾을 수 있다.



# 3. Heat equation behind gambling and martingales

https://youtu.be/pcUfdYdlyvc

- (조금더 엄밀한 확률 공간에 대한 정의)
- martingale 도 random variable들의 vector space에서의 conditional expectation 으로 이해하면 정상영으로 이해할 수 있다.
- (항상 수익이 있을 수 있는 도박에서의 stopping time에 대한 예시)
- $l = \infty$에서의 열방정식의 해는 normal distribution 이 된다.



# 4. Geometric structure behind normal distributions

https://youtu.be/zKQ5Fcfs2Gg

- $l = \infty$에서의 열방정식의 해는 normal distribution 이 된다.
- topology 소개
- openset = 찰흙덩어리
- openset 은 topology 의 원소
- 미분 다양체 or smooth manifold
  - locally Euclidian
    - chart 함수를 사용해 P 점 근방을 유클리드 공간으로 옮겨줄 수 있다.
- Riemanian metric

# 5. Radon-Nikodym theorem and the definition of conditional expectation

https://youtu.be/UClqCvqWtPg

- sigma-field F 는 topology이기도 한건가(??)
  - 그래야 random variable이 continuous function 인지를 topology 상에서 정의할텐데..
- 확률공간 (S, F, P)
  - S: sample space
  - F: sigma field
  - P: F 에서 [0, 1]로 보내는 probability measure
- random variable의 정의
  - F에서 R로 보내는 함수이다. R의 임의의 Borel set 에 대하여 역상이 F의 원소이다.
- random variable X의 공역은 정의상 확률공간이 아니었지만 확률공간으로 만들 수 있다.
  - probability distribution 은 random variable 의 공역에서 정의된 probability measure이다.
- topology T를 포함하는 가장 작은 sigma field 는 유일하게 존재하며 이를 Borel sigma field라고 한다.
  - Rudin's Real Analysis 첫번째 문제
- Radon-Nikodym theorem
  - 한 sigma field F에 대해 두개의 measure 가  주어졌을때 한 measure인 $\mu$는 다른 measure인 $\nu$로 표현할 수 있다.
  - $\nu(E) = \int_E X d\mu = \int_E f_X(x)dx$
  - 이 때 X를 라돈니코딤 derivative이라고 부르고 ${d\nu \over d\mu}$ 로 쓴다.
- probability density function 은 Radon-Nikodym theorem 으로 정의된다.
- conditional expectation 의 정의
  - $Y = \mathbb{E}(X|G)$
    - given
      - 확률 공간 (S, F, P)
      - 거기서의 random variable X
      - F의 subset인 sub sigma field G
    - Random variable인 X 와 Y의 분포 $P_X$, $P_Y$ 가 G에서 [0,1]로 가는 함수이면,
    - $\nu(E) = \int_E X d\mu = \int_E X dP = \int_E Y dP = \int_E Y d\mu$ 로 쓸수 있다.
    - 이때 라돈니코딤 derivative 인 $Y={d\nu \over d\mu}$를 conditional expectation E(X|G) 로 정의한다.
  - G에서 확률밀도 함수를 동일하게 만들어주는 random variable?
- tower property
  - $E(E(X|G)) = E(X)$

