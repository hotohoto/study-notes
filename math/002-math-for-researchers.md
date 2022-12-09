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