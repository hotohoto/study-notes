# Statistics

## Terminologies

- alpha level
  - 0.05, 0.01
- p-value
  - calculated from the currently observed sample
  - if this value is high, the null hypothesis is likely to be accepted.
    - it's likely that samples to be drawn are going to be more extreme than the current sample
      - because the current sample is not that extreme
  - if this value is low, th null hypothesis is likely to be rejected.
    - it's not likely that samples to be drawn are going to be more extreme than the current sample
      - because the current sample is already extreme
- independent variable
  - predictor
  - regressor
  - covariate
  - manipulated variable
  - explanatory variable
  - exposure variable
  - risk factor
  - feature
  - input variable
  - control variable
  - (exogenous variable)
- dependent variable
  - response variable
  - regressand
  - criterion
  - predicted variable
  - measured variable
  - explained variable
  - experimental variable
  - responding variable
  - outcome variable
  - output variable
  - target
  - label
  - (endogenous variable)
- loss function
  - cost function
  - risk function
- prediction intervals
  - confidence of predictions
- confidence intervals
  - confidence of population parameters

## Data types by the scale level of measurement

- Nominal
  - categories
- Ordinal
  - there is order
  - but the scale might not be the same all over the magnitude of values
- Interval
  - same scale
  - Celsius
- Ratio
  - same scale
  - greater than equal to zero
  - e.g. height, weight, duration

| Provides                                       | Nominal | Ordinal | Interval | Ratio |
|------------------------------------------------|---------|---------|----------|-------|
| The "order" of values is known                 |         | V       | V        | V     |
| "Counts" aka "frequency of Distribution"       | V       | V       | V        | V     |
| Mode                                           | V       | V       | V        | V     |
| Mean                                           |         | V       | V        | V     |
| Can quantify the difference between each value |         |         | V        | V     |
| Can add or substract values                    |         |         | V        | V     |
| Can multiply or divide values                  |         |         |          | V     |
| Has "true zero"                                |         |         |          | V     |


- https://www.mymarketresearchmethods.com/types-of-data-nominal-ordinal-interval-ratio/

## ANOVA

Analysis of variance (ANOVA)

- $H_0:\ \mu_1 = \mu_2 = \mu_3$
- $H_A:$ At least one pairs of samples is significantly different
- assumptions
  - normality
    - The data from each sample fall along a normal curve
  - homogeneity of variance
    - The data come from populations with similar amounts of variability

F-Ratio = "between-group variability" / "within-group variability"

$$
\text{F-ratio}
= {{\sum\limits_{i=1}^k n_i(\overline{X}_i - \overline{X}_G)^2 / (k-1)} \over {\sum\limits_{i=1}^k\sum\limits_{j=1}^{n_i}({X_{ij} - \overline{X}_i)^2 / (N - k)}}}
= {{SS_{between} / df_{between}} \over {SS_{within} / df_{within}}}
= {\text{MS}_{between} \over \text{MS}_{within}}
$$

If number of samples for each group is all the same as $n$,

$$
\text{F-ratio} = {{n\sum\limits_{i=1}^k (\overline{X}_i - \overline{X}_G)^2 / (k-1)} \over {\sum\limits_{i=1}^k\sum\limits_{j=1}^n({X_{ij} - \overline{X}_i)^2 / (N - k)}}}
$$
.

- $\overline{X}_G$: grand mean
- $\overline{X}_i$: sample mean for each group
- between group variability
  - The smaller the distance between sample means the less likely population means will differ significantly
  - The greater the distance between sample means, the more likely population means will differ significantly
- within group variability
  - The greater the variability of each individual sample, the less likely population means will differ significantly
  - The smaller the variability of each individual sample, the more likely population means will differ significantly
- df_total = df_between_group + df_within_group
- Lack-of-fit sum of squares ??
  - https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares
  - var_total = var_between_group + var_within_group
### Multiple comparision tests after ANOVA
This might be needed when the null hypothesis has been rejected.

#### Tukey's HSD

- Tukey's honesty significant difference
- Margin of error
- Compare margin of error with the pairwise mean difference to tell if it's significantly different

$$\text{(Tukey's HSD)} = q \sqrt{{\text{MS}_{\text{within}} \over n}}$$

- $q$: studentized range statistic
  - critical value of q is based on
    - 1 - alpha
    - k
    - df_within
- $\text{MS}_{\text{within}}$
  - $SS_{within} / df_{within}$

assumptions
- The observations being tested are independent within and among the groups.
- The groups associated with each mean in the test are normally distributed.
- There is equal within-group variance across the groups associated with each mean in the test (homogeneity of variance).

#### Cohen's d

- a measure of effect size

$$
\text{(Cohen's d)}
= {{\overline{X}_i - \overline{X}_j}\over{S_p}}
= {{\overline{X}_i - \overline{X}_j}\over{\sqrt{{\text{MS}_{\text{within}}}}}}
$$

where:
- $S_p$: the pooled standard deviation

#### $\eta^2$

- a measure of effect size
- The same as $r^2$ for t tests
- explained variation
- if it's greater than .14 it's considered big
- proportion of total variation that is due to between-group differences

$$
\eta^2 = {SS_{between} \over SS_{total}} = {SS_{between} \over {SS_{between} + SS_{within}}}
$$

## Correlation

$$
r = {COV_{XY} \over S_X S_Y}
$$

- $r$ is sample Pearson coefficient.
- $\rho$ is population Pearson coefficient.

### Causality

- Causality (also referred to as causation, or cause and effect) is influence by which one event, process, state or object (a cause) contributes to the production of another event, process, state or object (an effect) where the cause is partly responsible for the effect, and the effect is partly dependent on the cause.
- Correlation does not imply causation
  - http://www.tylervigen.com/spurious-correlations
- An experiment would be designed to be able to tell the causality relation between a dependent varaiable and an independent variable

### Hypothesis tests of $\rho$

- $H_0$
  - $\rho = 0$
- $H_A$
  - $\rho > 0$
    - positive directional alternative
  - $\rho < 0$
    - negative directional alternative
  - $\rho \neq 0$
    - non-directional alternative
$$
t = {r \sqrt{N - 2} \over \sqrt{1 - r^2}} \sim T(N-2)
$$

$$
\text{(degree of freedom)} = \nu = N - 2
$$

## Regression

In linear regression, line of best fit is

$$
\hat{y} = w_1 x + w_0
$$
,

$$
w_1 = \text{(slope)} = r * {S_Y \over S_X}
$$
.

$$
\text{(residual)} = y_i - \hat{y}_i
$$

$$
\text{(standard error of estimate)} = \sqrt{\sum (y - \hat{y})^2 \over N - 2}
$$

- The N−2 term accounts for the loss of 2 degrees of freedom in the estimation of the intercept and the slope.

### Hypothesis testing

- $\beta_1$
  - population slope
- $\beta_0$
  - population y-intercept

(for slope)

- $H_0$
  - $\beta_1 = 0$
- $H_A$
  - $\beta_1 \lt 0$
  - $\beta_1 \gt 0$
  - $\beta_1 \neq 0$

Uses t statistics.

(for y-intercept)

- It's possible to do the hypothesis testing for y-intercept.
- But usually it might not that important in real life.

### Multiple regression

- $R$
  - indicates weights before each predictor variable
- $R^2$
  - the proportion of the variation in the dependent variable that is predictable from the independent variable(s)

Note that $r^2$ is for single regression

## $\chi^2$ tests

$$
\chi^2 = \sum\limits_i {(f_o^{(i)} - f_e^{(i)})^2 \over f_e^{(i)}}
$$

Where

- $f_{o}^{(i)}$
  - observed frequency for $i$-th cell
- $f_{e}^{(i)}$
  - observed frequency for $i$-th cell
- assumtions
  - avoid dependent observations
    - e.g. each person contributes to no more than 1 field value frequencies.
  - avoid small expected frequencies
    - $n$ > 20
    - $n_{ij}$ > 5


### $\chi^2$ Goodness of fit test

- How well do our observed frequencies fit the expected frequencies?
- df = (n_categories - 1)
- $H_0$
  - observations are from the expected distribution
- $H_A$
  - observations are not from the expected distribution
- If k is larger the distribution is more similar to the normal distribution

### $\chi^2$ test for independence

$$
\chi ^{2}=\sum _{{i,j}}{\frac  {(n_{{ij}}-{\frac  {n_{{i.}}n_{{.j}}}{n}})^{2}}{{\frac  {n_{{i.}}n_{{.j}}}{n}}}}
$$

- Let's say we have a k-by-r contingency table for variable A and B corresponding to k and r
- expected distribution is when variable A has no dependency on variable B
- (degree of freedom) = (k - 1)(r - 1)
- $H_0$
  - observations are from the expected distribution where variable A has no dependency on variable B
- $H_A$
  - observations are from the expected distribution where variable A has dependency on variable B

Crammér's V
- sometimes denoted as $\varphi_c$
- a measure of association between two nominal variables, giving a value between 0 and +1 (inclusive).

$$
\varphi_c = V = \sqrt{
  \varphi ^ 2 \over \text{min}(k - 1, r - 1)
}
= \sqrt{
  {\chi ^ 2 / n} \over \text{min}(k - 1, r - 1)
}
$$

## etc

### Empirical cumulative distribution

- https://en.wikipedia.org/wiki/Empirical_distribution_function

## measures

### binary calssifier evaluation

- TPR
  - True Positive Rate
  - sensitivity
  - recall
  - hit-rate
  - TP / P
- TNR
  - True Negative rate
  - specificity
  - selectivity
  - TN / N
- PPV
  - Positive Predictive Value
  - precision
  - TP / TP + FP
- NPV
  - Negative Predictive Value
- FNR
  -
- FPR
  - False Positive Rate
  - fall-out
  - FP / N
  - 1 - TNR
- FDR
- PT
- TS
- ACC
- F1

#### Gini

- MAD(Mean Absolute Difference) or GMD(Gini Mean Difference)
  - most unequal when a half is extremly poor and another half is extremly rich
  - https://en.wikipedia.org/wiki/Mean_absolute_difference
  - GMD >= 0
  - MAD = MD = GMD
  - 4 formulas
    - based on absolut values
    - based on integral of cumulative distribution
    - based on covariance
    - based on lorenz curves (or integral of first moment distribution)
- gini coefficients
  - most unequal when the most are extremly poor and a few are extremly rich
  - Gini Coefficient =
- relative mean absolute difference = MAD / (arithmetic mean) = 2 * (Gini Coefficient)
- lorenz curve
- generalized lorenz curve
- Gini for binary classifier evaluation
  - https://www.listendata.com/2019/09/gini-cumulative-accuracy-profile-auc.html
  - Gini when it's used in the context of model evaluation, it's a normalized Gini which is always equal to accuracy ratio.
  - -1 ~ 1
  - the higher the better
  - Gini coefficient
    - sort data with predicted values
    - x-axis: number of samples
    - y-axis: accumulated numbers of positive ground truth
    - calculate `A / A + B`
  - normalized Gini
    - divide Gini coefficient by the optimal Gini coefficent
    - optimal Gini coefficient is calculated from an imaginary model which is able to predict the exact ground truth.

#### KS(Kolmogorov–Smirnov) test

- http://rstudio-pubs-static.s3.amazonaws.com/303414_fb0a43efb0d7433983fdc9adcf87317f.html
- https://arxiv.org/abs/1606.00496
- https://stats.stackexchange.com/questions/193439/two-sample-kolmogorov-smirnov-test-with-weights
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
- https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

- KS for binary classifier evaluation
  - max(TPR - FPR)
    - TPR: True Positive Rate
    - FPR: False Positive Rate
  - max()

### CV(Coefficient of Variation)

- = RSD(Relative Standard Deviation)
- $\sigma \over \mu$

### Shapiro-Wilk normality test

- $H_0$: The sample is drawn from a normally distributed population
- $H_1$: The sample is drawn from a not normally distributed population

(References)

- https://youtu.be/vo9DssNQA4E

## etc



## Statistics for thesis

얼마나 설문이 제대로 작성되었는지.

- 타당도(validity)
  - 측정 내용 등이 측정 목적에 부합하는지.
  - 내용 타당도(content validity)
    - 내용 타당도 평가는 주관적으로 함.
    - 전문가를 찾아가서 검토 받는 경우가 많음.
    - ex)
      - 안면 타당도
        - 얼마나 피시험자에게 친숙하게 검사지를 만들었는지.
  - 준거관련타당도(criterion-related validity)
    - 사람들이 인정하는 기존 검사지와 비교하기
    - ex)
      - 예언타당도(predictive validity)
        - 수능시험과 대학 학점의 연관성을 확인하기
          - 상관계수로 표현할 수 있음.
      - 공인타당도(concurrent validity)
        - 수능시험과 학교성적의 연관성을 확인하기.
          - 상관계수로 표현할 수 있음.
  - 구인타당도(construct validity)
    - 연구자가 설명한 요인이 동작하는 방식을 실재로 (설문지나) 실험이 잘 측정하도록 했는가?
    - construct
      - 연구자가 측정하려고 만들어낸 추상적인 개념임
      - 직접적으로는 관측하지 못할 수 있음.
    - aspects
      - 혹시 점수가 잘못나오면 결과적으로 생기는 문제가 무엇이있는지? 그래도 괜찮은지?
      - 관심있었던 내용을 측정한 것이 맞는지
      - 이론적인 근거가 적절하고 중요한지
      - 측정한 항목들이 관심있는 연구대상과 상관관계가 있는지?
      - 실험을 통해 잘 예측하고 분류해낼 수 있었는지.
      - 해당 실험이 다른 그룹이나 다른 행동들에도 잘 적용되는지.
    - 어떻게 계선하는지
      - 관련 없는 문항을 제거
- 신뢰도(reliability)
  - 일관성있고 정확히 측정하는가? 여러번했더니 결과가 다르게 나오지는 않는가?
  - `X = T + e`
    - X 측정값
    - T 실제값 (Ground truth)
    - e 오차
  - 재검사쇤뢰도(test-retest reliability)
    - 다음에 한번 더 동일하게 다시 실험
    - 상관계수 측정
    - 기억효과를 최소화 하려면 시간을 잘 선택해야 함.
    - 동기나 태도도 달라질 수 있음.
  - 동형검사신뢰도(parallel-form reliability)
    - 비슷하지만 다른 시험을 만들어 두 번 시험을 봄
    - 두개의 비슷하지만 다른 검사지를 만드는게 어렵긴함.
  - 내적일관성신뢰도(internal consistency reliability)
    - 시험을 한번만 봄 그 안에서 신뢰도를 평가함.
    - 반분검사신뢰도
      - 검사를 부분으로 나눔.
      - ex) 홀수와 짝수, 앞쪽과 뒷쪽
    - 문항내적일관성신뢰도
      - KR-20
      - KR-21
      - Cronbach α
        - 문항 하나가 하나의 검사라고 간주함.
        - 보수적으로 신뢰도를 약간 과소 평가하는 경향이 있을 수 있으나 바람직함.
  - Inter-Rater or Inter-Observer Reliability
    - 관찰자 신뢰도
      - 관찰자가 일관성 있게 측정하는지
    - 과찰자간 신뢰도
      - 과찰자들이 얼마나 유사한가
      - 상관관계 분석을 사용할 수 있음.

조사연구(Survey)
