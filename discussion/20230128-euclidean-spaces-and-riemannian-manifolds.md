## Euclidean spaces and Riemannian manifolds

(질문)

뭔가 제가 오해하고 있는데 도움을 요청드립니다.

- (1) Euclidean space 에서는 cartesian coordinate 를 쓰는 경우 dot product를 내적으로 사용하지만, 다른 좌표계를 사용하면 내적에 해당하는 metric tensor 가 달라진다고 알고 있는데요. 이게 맞다면,, 두 내적은 좌표 시스템이 달라서 다르게 보이지만 실제로는 같은 object인 것으로 이해해야할까요?
- (2) 그리고 Euclidean space에 대한 정의 혹은 공리는 Euclidean space와 non-Euclidean space 를 무엇으로 구분할까요? 위키를 찾아봐도 정확히 뭐라는건지 이해가 잘 가지 않습니다. "A Euclidean vector space is a finite-dimensional inner product space over the real numbers." 이런 부분이 있는데 이 조건만 맞으면 Euclidean space라는 것일까요? dot product를 를 cartesian coordinate에서 내적으로 쓴다는 것은 Euclidean space인지 아닌지를 구분해내는 것이 아니라고 봐야할까요?



(답변1 - 생략, 전공수학공부 오픈채팅)

(답변2 - 수즐 오픈채팅)

```
윗 분 질문은 리만기하학에서 공간의 정의가 뭔지 묻는 것입니다. 매니폴드와 리만매트릭, 레비 시비타 접속을 살펴보시고 그 가운데 유클리드 공간이 뭔지 보시면 됩니다. 
```



(정리)

- (1)

  - 👉 coordinate system 이 바뀐면 내적을 계산하는 법이 달라지지만 내적이 바뀌지는 않는다.

  - 👉 한 Euclidean space에서는 내적이 하나만 있다.

- (2)

  - 👉 Euclidean space의 정의는 인용한 것 그대로이다.
  - 👉 Riemannian space 혹은 Riemannian manifold 라고도 불리는 이것은 Euclidean space 가 아닌 공간이라는 식으로 정의되지 않으며 한 공간안에 사용되는 내적이 무수히 많아진다.

- 기타

  - Euclidean space 는 유한 차원 실수들의 공간에 정의된 inner product space이고 그러므로 해당 공간에서는 vector의 scalar 곱이 잘 정의되는 선형적인 특성들을 가진다.

  - 특정 n 차원을 가지는 Euclidean space는 내적을 어떻게 정의하냐에 따라 다양하게 여러개 만들 수 있지만 모두 isomorphic하다.

  - 당연하지만 한 Euclidean space 안에서 inner product는 고정되어 있다.

  - 한 Euclidean space 안에서 basis 가 바뀌고 coordinate system 이 바뀌면 내적은 계산하는 관점에서 다르게 표현되겠지만 여전히 동일한 내적을 쓰고 있는 것이다.

  - Riemannian geometry 에서의 공간은 manifold로서 local 하게만 Euclidean space인 것으로 본다. 이 경우는 한 공간에서 내적이 유일하지 않으며 local 한 Euclidean space에 살고 있는 vector들에만 그 tangent space에서의 특정 내적이 적용될 수 있다.

