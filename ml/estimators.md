# Estimator

- An estimator is a rule for calculating an estimate of a given quantity based on observed data.

## Point estimator

$\hat{\theta}_m= g(x^{(1)}, \dots, x^{(m)})$

- Point estimators yield single-valued results.
- "Single value" does not necessarily mean "single number", but includes vector valued or function valued estimators.

## Interval estimator

- Interval estimators would yield a range of plausible values.

## Estimator vs predictor
- An estimator uses data to guess at a parameter while a predictor uses the data to guess at some random value that is not part of the dataset.

## Bias variance tradeoff

MSE of an estimator can be decomposed as the squared bias and the variance.

$$
\text{Bias}(\hat{\theta}_m) = \mathbb{E}(\hat{\theta}_m) - \theta
$$

$$
\text{MSE} = \mathbb{E}[ (\hat{\theta}_m - \theta)^2] = \text{Bias}(\hat{\theta}_m)^2 + \text{Var}(\hat{\theta}_m)
$$

- https://en.wikipedia.org/wiki/Mean_squared_error#Estimator

Apart from this, we can also think about the bias variance tradeoff with respect to a predictor and the predictions.

- https://gist.github.com/petermchale/a4fc2ca750048d21a0cbb8fafcc690af

## Consistency

### Consistency condition

$$
\text{plim}_{m \to \infty} \hat{\theta}_m = \theta
$$


- The symbol plim indicates convergence in probability, meaning that for any $\epsilon \gt 0$, $P(|\hat{\theta}_m − \theta| \gt \epsilon) \to 0$ as $m \to \infty$.
- $m$: number of examples in the observed sample
- $\theta$: the true parameter value we may want to estimate
- $\hat{\theta}_m$: estimate given m observations

### Almost sure convergence condition

$$
p(\lim_{m \to \infty} \hat{\theta}_m = \theta) = 1
$$

## MLE

- MLE는 θ hat을 추정하는 point estimator 이다. (p(d|θ) 를 최대화 하기위한 θ hat값 하나를 찾는 estimator이다.)
- MLE는 빈도주의 방법이라고도 할 수 있다.
- MLE에서 d가 달라지면 θ hat이 바뀌므로 θ hat을 random variable로 생각할 수도 있다.
- 그러나 d가 정해진 상태에서는 θ hat은 deterministic 한 값이다.

## MAP
- MAP는 θ에 대한 point estimator이다. θ가 random variable 인 것으로 보며 그래서 θ 가 어떤 특정 분포를 따른다고 가정한다는 점에서 베이지안 방법이다.
- MAP에서 θ 가 uniform distribution을 따른다고 가정하고 θ|d 를 풀면 MLE와 같은 θ hat을 찾게된다.
- MAP regularizes MLE

## properties in common

- MLE든 MAP든 θ hat을 찾아내는 point estimator 이면서 동시에,, 해당 point θ hat은 학습에 사용한 데이터셋에 따라 달라질 것이므로,, confidence interval 을 논할 수 있다.
- 이렇게 찾아낸 θ hat에 대한 MSE도 confidence interval을 논할 수 있다.??

- 더 좋은 estimator는 평균적으로 낮은 MSE(ground truth θ 에 가까운 θ값을 찾아내겠지만, 데이터셋에 따라 어느 한 알고리즘이 다른 알고리즘에 비해 항상 더 좋은 θ를 찾아내지는 않는다. 특정 데이터 D에 대해 두 알고리즘의 confidence interval 이 겹치지 않는다면 한 알고리즘이 다른 알고리즘보다 통계적으로 유의미하게 더 좋다고 할 수는 있다.

## OX 퀴즈

1. MLE는 P(Data|θ) 를 최대화하는 θ를 추정한다.
2. MLE(Maximum Likelihood Estimation)는 point estimation이다.
3. MLE는 빈도주의자(frequentist)의 방법이다.
4. MAP는 P(Data|θ)P(θ) 를 최대화하는 θ를 추정한다.
5. MAP는 베이지안(Bayesian) 방법이다.
6. MAP는 point estimation이다.
7. MAP에서 prior 를 uniform distribution으로 두면 MLE가 된다.

## References

- https://www.deeplearningbook.org/contents/ml.html
