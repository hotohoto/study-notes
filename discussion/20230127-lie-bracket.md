# Seeing partial derivatives as vector fields and about Lie Bracket

(질문1)

```
미분 연산자를 변수처럼 쓰는것을 물리학쪽 강의에서 처음 봤고, 저는 그냥 미분연산자가 선형 변환을 하는 연산이어서 matrix 같은 object 처럼 생각할 수 있다 정도로 추측하고 있는데, 더 엄밀하게는 어떻게 받아들여야할지 궁금하네요
```

(답변1)

```
미분을 취하는 함수를 하나 임의로 고정하고 그 함수에 대한 미분계수들을 보아나가는 것으로 보면 됩니다. 이 이면의 엄밀한 수학적인 묘사 로는 현재 만들고 있는 1차 미분계수들 간의 Lie Bracket이 0이 되는 구조 하에서 작업을 하고 있으므로 그로 인하여 결과적으로 연산 순서가 무관하기에 정말로 숫자의 연산처럼 생각을 해도 되는 것입니다.
```

(질문2)

```
언제 Lie bracket이 0 이 되는지?
```

(답변2)

```
이 이야기 타래의 디스커션은 편미방 수업에서 하는 이야기를 따라가는데 중요하진 않다고 여기나 질문의 답변을 남깁니다. 두 벡터장 X,Y가 주어질 때에 둘 간의 리브라켓 [X,Y]=XY-YX는 또다른 벡터장으로 직관적으로는 벡터 Y를 만들어내는 curve에 대해 다른 벡터장 X를 미분해서 만드는 것으로 일반적으로 임의의 두 벡터장의 리브라켓은 유클리드 공간의 경우를 포함해서 0이 아닙니다. 그러나 아주 특별한 벡터장들의 경우에는 유클리드 공간 뿐만 아니라 더 일반적인 공간들에서 국소좌표계 (x1,…,xn)으로부터 만들어낸 벡터장들인 경우에는 d/dx1, …, d/dxn으로 리브라켓이 0이 됩니다. 수업에서 살펴본 경우들은 1차 편미분 연산자가 이러한 벡터장에 해당하므로 문제가 없는 것입니다.
```



(정리)



위키피디아에서 Lie bracket 의 첫번째 정의를 설명 보면 이렇게 되어 있습니다.

- smooth vector field $X: M \to TM$ 는 manifold 에 있는 한 점을 그 접평면에 있는 vector로 보내는 역할을 한다.
- $X$는 $M$위에서 정의된 함수인 $f(p)$에 대해서 미분 연산을 하는 연산자로 볼 수 있다. ($p \in M$)
- $X$로 $f$를 미분해보면 $X(f)$ 로 쓰고 이것의 결과는 각 $p$ 점마다 $X(p)$ 방향으로 $f$의 방향 도함수를 뱉어주는 함수가 되게 된다.
- 아무튼 이렇게 볼때 각기 다른 smooth vector field는 각기 다른 미분 연산을 정의한다.
- 그러면 두개의 smooth vector field 들인 $X, Y$가 있으면 이를 사용하여 한번씩 미분하면 미분 연산자의 합성인 것으로 볼 수 있다.
- Lie bracket은 이러한 관점에서 다음과 같이 정의한다.

$$
[X,Y](f)=X(Y(f))-Y(X(f))\;\;{\text{  for all }}f\in C^{\infty }(M)
$$

- 즉 $f$를 $Y$로 미분한 후에 $X$로 미분한 것과  그 반대 순서로 미분한것의 차이를 나타낸다. (이 값이 0 이라면 어느것으로 먼저 미분해도 차이가 없다는 뜻이 된다.)



위에서 smooth vector field $X$ 가 만드는 미분에 대해 좀 더 생각해 봅시다.

- 우리는 gradient를 어떤 방향의 vector와 dot product하면 그 방향으로의 directional derivative 가 된다는 것을 이번 강의를 통해 알고 있습니다.
- 그러다면 여기서는 $T_pM$ 에서 함수 f의 정의역의 각 component 방향으로 f의 방향도함수를 구해 gradient를 구성하고 이것을 다시 $X(p)$ 와 dot product를 했을 거라고 생각해볼 수 있습니다.
- (앗 그런데 $f$ 가 $M$ 에서만 정의되었는데 $M$을 벗어나고 있을수도 있는 $T_pM$ 에서 방향 도함수를 구한다는게 말이 맞지가 않는것 같은데, 일단 $M$ 밖의 ambient space 에 대해서도 정의가 된다고 가정하자. 🤔)
- $X$ 가 이 과정에서 한 역할은 방향도함수의 방향을 정해준것 뿐이라는 것도 주목해 볼만 한 것 같습니다.



그러면 우리가 문제에서 사용했던 $u(x,y)$ 에 대한 partial derivative 연산자들은 어떤 vector field들에 해당했을까 하는 점이 궁금해집니다.

- partial derivative $\partial u \over \partial x$ 에 해당하는 vector field 는 아마도 모든 영역에서 $x$ 방향이 1, $y$ 방향으로 0 이되는 값으로 되어 있을 것입니다. 그래야 $\nabla_{(x,y)}u$ 를 구한뒤 $(1,0)$ 과 내적하면 $\partial u \over \partial x$ 가 나오는 것이 자연스럽기 때문입니다.
- 마찬가지로 partial derivative $\partial u \over \partial y$ 에 해당하는 vector field 는 아마도 모든 영역에서 $x$ 방향이 0, $y$ 방향으로 1 이되는 값으로 되어 있을 것입니다. 

긴 이야기를 했지만, 우리 문제에서는 Lie Bracket이 그냥 $[{\partial\over\partial x}, {\partial\over\partial y}](u) = ({\partial ^2\over\partial x \partial y} - {\partial ^2\over\partial y \partial x})u$ 이 될텐데 강의중에 순서와 무관하게 된다는 것을 언급하셨기 때문에 0 이될 것이라는 것이라 판단할 수 있습니다. 즉 Lie Bracket 이 0 이어서 두 편미분의 순서가 상관이 없다가 아니라 두 편미분의 상관없으므로 Lie Bracket이 0 이다고 말하는 것이 지금으로서는 이해하고 있습니다.

아마도 교수님 말씀을 포함해서 제가 오해했거나 놓치고 있는 부분이 있을거라 생각 되는데 차차 알아갈 수 있기를 바랍니다. 



참고한 글:

- https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields#Vector_fields_as_derivations

