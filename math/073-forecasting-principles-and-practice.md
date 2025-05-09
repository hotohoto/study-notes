# Forecasting: Principles and Practice

https://otexts.com/fpp3/

Rob J Hyndman and George Athanasopoulos
Monash University, Australia

## Questions

- aic, bic, R^2, adjusted R^2
  - and their theoretical background
- arima
  - which model to choose between ma and ar models
  - ar
    - acf
  - ma
    - pacf
      - https://en.wikipedia.org/wiki/Partial_correlation
    - rolling vs moving average
- what is exponential smoothing
  - ets
- prophet model
  - https://facebook.github.io/prophet/

## (personal notes)

- algorithms
  - ARMA (Autoregressive moving average)
  - ARIMA (Autoregressive integrated moving average)
    - p
      - number of time lags of the autoregressive model
    - d
      - degree of differencing which means how many times we apply differencing operator
      - usually 1 or 2 and rarely above
    - q
      - order of moving average model
  - LSTM with attention
  - GRU
  - EWMA
- Online resource in Korean
  - https://datascienceschool.net/view-notebook/9987e98ec60946c79a8a7f37cb7ae9cc/
- evaluation
  - MSD(Mean Squared Deviation)
    - Similar to MSE
  - MAD(Mean Absolute Deviation)
    - Similar to MAE
  - MAPE(Mean Absolute Percentage Error)
- decomposition
  - type of components
    - seasonal variational component
      - how?
        - find trend
        - detrend
          - division for the multiplicative model
          - subtraction for the additive model
        - smoothing
        - amplitude
    - trend variational component
    - irregular variational component
  - multiplicative model
    - `y(t) = Level * Trend * Seasonality * Noise`
  - additive model
    - `y(t) = Level + Trend + Seasonality + Noise`
- endogenous variables
- exogenous variables
- n_lags
  - https://en.wikipedia.org/wiki/Lag_operator
- window
  - linear map from R^L to R^KxL trajectory matrix
  - basically not directly related to timeseries

## 1. Getting started

https://otexts.com/fpp3/intro.html

### 1.1 What can be forecasted

### 1.2 Forecasting, planning and goals

- forecasting
  - as is
- goals
  - to be
- planning
  - how
- short-term forecasts
- midium-term forecasts
- long-term forecasts

### 1.3 Determinig what to forecast

### 1.4 Forecasting data and method

- qualitative forecasting
- quantitative forecasting

#### Time series forecasting

#### Predictor variable and time series forecasting

### 1.5 Some case studies

### 1.6 The basic steps in a forecasting task

- Step 1: Problem definition
- Step 2: Gathering information
- Step 3: Preliminary (exploratory) analysis
- Step 4: Choosing and fitting models
- Step 5: Using and evaluating a forecasting model

### 1.7 The statistical forecasting perspective

## 2. Time series graphics

https://otexts.com/fpp3/graphics.html

### 2.1 ts objects

#### Frequency of a time series

- frequency
  - the number of obervations before the seasonal pattern repeats
  - the opposite of the definition of frequency in physics, or in Fourier analysis

### 2.2 Time plots

### 2.3 Time series patterns

- drift
- trend
- seasonal
- cyclic

### 2.4 Seasonal plots

![Figure 2.4: Seasonal plot of monthly antidiabetic drug sales in Australia.](https://otexts.com/fpp3/fpp_files/figure-html/seasonplot1-1.png)

### 2.5 Seasonal subseries plots

![Figure 2.6: Seasonal subseries plot of monthly antidiabetic drug sales in Australia.](https://otexts.com/fpp3/fpp_files/figure-html/subseriesplot-1.png)

### 2.6 Scatterplots

Good for exploring relationships between time series

![Figure 2.8: Half-hourly electricity demand plotted against temperature for 2014 in Victoria, Australia.](https://otexts.com/fpp3/fpp_files/figure-html/edemand2-1.png)

#### Correlation

$$
r=\frac{\sum\left(x_{t}-\bar{x}\right)\left(y_{t}-\bar{y}\right)}{\sqrt{\sum\left(x_{t}-\bar{x}\right)^{2}} \sqrt{\sum\left(y_{t}-\bar{y}\right)^{2}}}
$$

#### Scatterplot matrices

![Figure 2.12: A scatterplot matrix of the quarterly visitor nights in five regions of NSW, Australia.](https://otexts.com/fpp3/fpp_files/figure-html/ScatterMatrixch2-1.png)

### 2.7 Lag plots

https://otexts.com/fpp3/lag-plots.html

![Figure 2.13: Lagged scatterplots for quarterly beer production.](https://otexts.com/fpp3/fpp_files/figure-html/beerlagplot-1.png)

### 2.8 Autocorrelation

https://otexts.com/fpp3/autocorrelation.html

$$
f(k) = r_{k}=\frac{\sum_{t=k+1}^{T}\left(y_{t}-\bar{y}\right)\left(y_{t-k}-\bar{y}\right)}{\sum_{t=1}^{T}\left(y_{t}-\bar{y}\right)^{2}}
$$

where
- $T$ is length of the time series

### 2.9 White noise

https://otexts.com/fpp3/wn.html

TODO

## 3. Time series decomposition

https://otexts.com/fpp3/decomposition.html

TODO

## 4. Time series features

https://otexts.com/fpp3/features.html

TODO

## 5. The forecaster's toolbox

https://otexts.com/fpp3/toolbox.html

### 5.1 Some simple forecasting methods

- Mean
- Naive
- Seasonal naive

### 5.2 Transformation and adjustments

- calendar adjustments
  - monthly milk production per cow data might need to be adjusted by the number of days in each month
  - monthly sales data might need to be adjusted by the number of trading days in each month
- population adjustments
  - number of hospital beds data might need to be adjust by population
- inflation adjustments
  - yearly house price data might need to be adjust by the Consumer Price Index
- mathmatical transformations
  - logarithmic transformations
  - power transforations
  - Box-Cox transformations
    - choose $\lambda$ that makes the size of seasonal variation about the same
    - bias adjustmnets for calculating means

### 5.3 Residual diagnosis

https://otexts.com/fpp3/residuals.html

(residuals)

$e_t = y_t - \hat{y_t}$

- Note that fitted values are often not true forecasts
  - estimated using all available observations in the time series, including future observations
- A good model will yield residuals but with the following properties
  - Uncorrelated residuals
    - can be tested with Breusch-Godfrey test
    - might be acquired by model correlated component of the residuals
  - Zero mean residuals, otherwise biased
  - Constant variance
    - comes in handy when calculating prediction intervals
    - might be acquired by box-cox transformation
  - Normally distributed
    - comes in handy when calculating prediction intervals
    - might be acquired by a box-cox transformation
- Portmanteau test for autocorrelation
  - for testing a group of autocorrelations to be less than a criterion along with $\chi^2$ test
    - degree of freedom is set as number of parameters used for the model
  - Box-Pierce test
    - $Q = T \sum_{k=1}^h r_k^2$
    - $T$: number of observations
    - $h$
      - 10 is recommended for non-seasonal data
      - min(2m, T/5) is recommended for seasonal data
        - m is the period of seasonality
  - Ljung-Box test
    - $Q^* = T(T+2) \sum_{k=1}^h (T-k)^{-1}r_k^2$

### 5.4 Evaluating forecast accuracy

https://otexts.com/fpp3/accuracy.html

- forecast errors
  - $e_t = y_t - \hat{y_t}$
  - test set
  - multi-step forecasts
    - forecasts with gap
- residuals
  - one-step forecasts
  - training set

(measures)

- MAE
  - scale is the same as the data
  - Minimising the MAE will lead to forecasts of the median.
- RMSE
  - scale is the same as the data
  - Minimising the RMSE will lead to forecasts of the mean.
  - more difficult to interpret
- MAPE
  - $p_t = 100 e_t / y_t$
  - $\text{MAPE} = \text{mean}(|p_t|)$
  - pros
    - unit free
    - better at comparing forecast performances between data sets
  - cons
    - undefined when $y_t = 0$
    - put a heavier penalty on negative errors than on positive errors
- sMAPE
  - symmetric MAPE
  - $\text{sMAPE} = \text{mean}(200|y_t  - \hat{y_t}| / (y_t + \hat{y_t}))$
  - can be negative
  - undefined when both $y_t$ and $\hat{y_t}$ are zero
  - not remmeneded, but widely used.
- MASE
  - Mean absoulute scaled error
  - 2006, by Hyndman & Koehler
  - better at comparing forecast performances between data sets
  - $\text{MASE} = \text{mean}(|q_j|)$
  - (for non seasonal data)
    - $q_j=\frac{e_j}{\frac{1}{T-m} \sum_{t=2}^{T}\left|y_{t}-y_{t-m}\right|}$
  - (for seasonal data)
    - $q_j=\frac{e_j}{\frac{1}{T-m} \sum_{t=m+1}^{T}\left|y_{t}-y_{t-m}\right|}$
  - scaling the errors based on the training data MAE of the seasonal/non-seasonal naive model

(Time series cross validation)

...

### 5.5 Prediction intervals

https://otexts.com/fpp3/prediction-intervals.html

$\hat{y}_{T+h \mid T} \pm 1.96 \hat{\sigma}_{h}$

- 95% prediction interval
- where $\hat\sigma_h$ is an estimate of the standard deviation of the
h-step forecast distribution
- assumption:
  - the forecast errors are normally distributed

Then, how can we estimate $\hat\sigma_h$?

(One-step prediction intervals)

- $\hat\sigma_h$ is slightly larger than $\hat\sigma$ which is the standard deviation of the residuals.
- sometime the difference is ignored
- for the naive method it's the same

(Multi-step prediction intervals)

- assumption
  - the residuals are uncorrelated

(Benchmark methods)

- mean forecasts:
  - $\hat\sigma_h = \hat\sigma\sqrt{1 + 1 / T}$
- naive forecasts:
  - $\hat\sigma_h = \hat\sigma\sqrt{h}$
- seasonal naive forecasts:
  - $\hat\sigma_h = \hat\sigma\sqrt{k + 1}$
    - $k = \operatorname{integer}((h - 1) / m)$
- drift forecasts
  - $\hat\sigma_h = \hat\sigma\sqrt{h(1 + h / T)}$

(Prediction intervals from bootstrapped residuals)

- $e_t = y_t - \hat{y}_{t|t -1}$
- $y_t = \hat{y}_{t|t -1} + e_t$

Then,

- $y_{T+1} = \hat{y}_{T + 1|T} + e_{T+1}$
  - we sample $e_{T+1}, ...$ from the collected residuals.
- $y_{T+2} = \hat{y}_{T + 2|T + 1} + e_{T+2}$
  - we sample $e_{T+2}, ...$ from the collected residuals.
- repeat to simulate an entire set of future values
- assumption
  - future errors will be similar to past errors
  - (it doesn't need to be normal forecast errors)

(Prediction intervals with transformations)

If a transformation has been used,
- the prediction interval should be coputed on the transformed scale.
- the endpoint should be back-transformed to give a prediction interval on the original scale.

## 6. Judgemental forecasts

https://otexts.com/fpp3/judgmental.html

When to do judgemental forecasting

- a complete lack of historical data
- a new product is being launched
- a new competitor enters the market
- during completely new and unique market conditions
- avaialbe data but
  - when it needs to analyze and improve data-driven forecasting
  - or when it needs to be combined with judgemental forecasting

"nowcasting"

how to improve judgemental forecasting

- important domain knowledge
- timely, up-to-date information
- well-structured approach

### 6.1 Beware of limitations

https://otexts.com/fpp3/judgmental-limitations.html

- subjective
  - a limited memory
  - the effect of psychological factors
  - wishful thinking
  - anchoring
- inconsistent
- depends on a limited memory
  - may render recent events more important
  - may ignore momentous events from the more distant past

### 6.2 Key principles

https://otexts.com/fpp3/judgmental-principles.html

- Set the forecasting task clearly and concisely
- Implement a systematic approach
  - list factors
  - list assumptions
  - make guidlines and rules defining how to combine them
- Document and justify
  - for consistency, accountability, evaluation
- Systematically evaluate forecasts
  - keep records
  - get feedbacks
  - improve accuracy
- Segregate forecasters and users
  - Forecasters should make forecasts as accurate as possible.
  - It's fine for users to interpret forecasts conservatively/optimistically for their own purpose.
  - Those processes should be independent.

Example: Pharmaceutical Benefits Scheme (PBS)

### 6.3 The Delphi method

https://otexts.com/fpp3/delphimethod.html

assumption:

Forecasts from a group are gernerally more accurate than those from individuals

stages:

- A facilitator is appointed
- A panel of experts is assembled
   - 5~20 people usually with diverse expertise
   - all the experts remain anonymous to avoid social/political pressures
     - no group meeting
     - everyone contributes
- Forecasting tasks are set and distributed to the experts.
- (Optional) Prelinary round of information gathering
- repeat (two or three rounds until a statisfactory consensus between the experts is reached)
  - Experts return forecasts and justifications.
  - feedback including
    - summary statistics of the forecasts
    - outlines of qualitative justifications with numerical/graphical summaries
    - the facilitator may direct the experts' attention to reponses that fall outside of interquartile range, and the qualitative justification for such forecasts
  - (Optionally) "estimate-talk-estimate"
    - allow interaction between experts between iterations
- Final forecasts are constructed by aggregating the experts' forecasts.
  - with equal weights

### 6.4 Forecasting by analogy

https://otexts.com/fpp3/analogies.html

- e.g. appraisal processes
- base forecasts on multiple analogies
- a structured analogy
  - a facilitator, a panel of experts
  - list analogies
  - and generate forecasts based on them
  - list similarities and differences of each analogy
  - the facilitator derive the forecasts with the set rule
    - e.g. weighted average

### 6.5 Scenario forecasting

https://otexts.com/fpp3/scenarios.html

- come up with plausible scenarios
  - by considering all possible factors, their relative impacts, the interaction between them, and the targets to be forecast
- extreme cases can be identified

### 6.6 New product forecasting

- can use
  - Delphi
  - forecasting by analogy
  - scenario forecasting
  - sales force composite
    - pros
      - they have valuable experiences
    - cons
      - they can be biased
  - executive opinion
    - similar to estimate-talk-estimate
  - customer intentions
    - how asking customers with a structured questionaire
    - challenges
      - collecting a representative sample
      - applying a time and cost effective method
      - dealing with non-responses
      - differentiate purchase intention from purchase behavior.
        - the correlation between them varies depending on
          - the time between intention and behavior
          - type of industry
          - whether it's regarding familiar products or compleetely new products

### 6.7 Judgemental adjustments

- Use adjustments sparingly
  - use it only when
    - there is significant additional information which is not included in the statistical model
    - and ther is strong evidence of the need for an adjustment
  - small adjustments don't seem t be useful
- Apply a structured approach

## 7. Time series regression models

https://otexts.com/fpp3/regression.html

terms

- forecast variable
  - regressand
  - dependent variable
  - explained variable
- predictor variables
  - regressors
  - independent variables
  - explanatory variables

### 7.1 The linear model

https://otexts.com/fpp3/regression-intro.html

Simple linear regression

$$y_t = \beta_0 + \beta_1 x_t + \epsilon$$

Multiple linear regression

$$y_t = \beta_0 + \beta_1 x_{1,t} + \beta_2 x_{2,t} + \cdots + \beta_k x_{k,t} + \epsilon \qquad{(5.1)}$$

assumptions:
- the model is a reasonable approximation to reality
- assumptions on errors $\epsilon_1, \cdots, \epsilon_T$
  - they have mean zero
  - they are not autocorrelated
  - unrelated to the predictor variables

### 7.2 Least squares estimation

https://otexts.com/fpp3/least-squares.html

$$
\hat{\beta}_{0}, \ldots, \hat{\beta}_{k}
=\argmin\limits_{\beta_0, \ldots, \beta_k}\sum_{t=1}^{T} \varepsilon_{t}^{2}
=\argmin\limits_{\beta_0, \ldots, \beta_k}\sum_{t=1}^{T}\left(y_{t}-\beta_{0}-\beta_{1} x_{1, t}-\beta_{2} x_{2, t}-\cdots-\beta_{k} x_{k, t}\right)^{2}
$$

Fitted values:

$$\hat{y}_{t}=\hat{\beta}_{0}+\hat{\beta}_{1} x_{1, t}+\hat{\beta}_{2} x_{2, t}+\cdots+\hat{\beta}_{k} x_{k, t} \qquad{(5.2)}$$

- Note that it's not forecasting future values of $y$.

#### Goodness-of-fit

"R-squared" or "the coefficient of determination"

$$
R^2
= 1 - {\text{RSS} \over \text{TSS}}
$$

- For linear regression:

$$
R^2
= 1 - {\text{RSS} \over \text{TSS}}
= {\text{ESS} \over \text{TSS}}
= \frac{\sum\left(\hat{y}_{t}-\bar{y}\right)^{2}}{\sum\left(y_{t}-\bar{y}\right)^{2}}
$$

- where
  - TSS: [Total sum of squares](https://en.wikipedia.org/wiki/Total_sum_of_squares)
    - $\propto$ (total variance by the model)
  - RSS: [Residual sum of squares](https://en.wikipedia.org/wiki/Residual_sum_of_squares)
    - $\propto$ (unexplained variance by the model)
  - ESS: [Explained sum of squares](https://en.wikipedia.org/wiki/Explained_sum_of_squares)
    - $\propto$ (explained variance by the model)
- notes
  - "TSS = ESS + RSS" holds only under the linear regression context.
    - https://en.wikipedia.org/wiki/Partition_of_sums_of_squares
    - In the linear regression case
      - the value lies in $[0, 1]$.
  - A polynomial regressions is also a kind of multiple linear regressions.
  - limits
    - cannot tell if the model is biased
      - a model still can be improved although $R^2$ value is quite high
    - Adding more predictor variables always increases leading to overfitting.
  - in multiple linear regressions
    - $R^2 = \rho_{\hat{y}y}$
  - in simple linear regressions
    - $r^2 = \rho_{\hat{y}y} = \rho_{xy}$
      - https://rpubs.com/beane/n3_1b
  - extensions
    - adjusted $R^2$

- used to address the issue that adding more predictor variables

#### Standard error of the regression

$$\hat{\sigma}_{e}=\sqrt{\frac{1}{T-k-1} \sum_{t=1}^{T} e_{t}^{2}}$$

- where we have $k + 1$ parameters

### 7.3 Evaluating the regression model

https://otexts.com/fpp3/regression-evaluation.html

#### ACF plot of residuals

- draw ACF plot of residuals
  - see if there are significantly high or low values
  - unbiased model can have autocorrelated residiuals
    - such model is inefficient in that there is some information left over
- do the Breusch-Godfrey test
  - or Lagrange Multiplier (LM) test for serial correlation
  - test of autocorrelation in the residuals designed to take account for the regression model
    - null hypothesis:
      - there is no autocorrelation in the residuals
    - a small p-value indicates that there is significant autocorrelation remaining in the residuals
- draw histogram of residuals
  - check normality

![Figure 7.8: Analysing the residuals from a regression model for US quarterly consumption.](https://otexts.com/fpp3/fpp_files/figure-html/uschangeresidcheck-1.png)

#### Residual plots against predictors

- draw scatter plots of the residuals against in-model predictors
  - if there is a pattern
    - it means there is a non-linear relationship
    - consider doing nonlinear regression
- draw scatter plots of the residuals against out-of-model predictors
  - if there is a pattern
    - consider including the predictor

#### Residual plots against fitted values

- draw scatter plots of the residuals against fitted values
  - if there is a pattern
    - it may indicate there is "heteroscedasticity" in the errors
      - it means the variance of the residuals may not be constant
    - a transformation of the forecast variable such as a logarithm or sqaure root may be required

#### Outliers and influential observations

- outliers
  - Observations that take extreme values compared to the majority of the data
- influential observations
  - Observations that have a large influence on the estimated coefficients of a regression model
  - Usually, influential observations are also outliers that are extreme in the $x$ direction
- tips
  - draw a scatter plot and see if there are outliers
  - study the reason behind the outliers
  - remove non sensible outliers
  - report results both with and without outliers

#### Spurious regression

- signs of spurious regression:
  - High $R^2$ and high residual auto correlation
    - maybe due to non stationary conditional distribution over time
- e.g. Australian air passengers on rice production in Guinea
- even a spurious regression may give reasonable short-term forecasts, but it's not a good model

![Figure 7.13: Residuals from a spurious regression.](https://otexts.com/fpp3/fpp_files/figure-html/tslmspurious2-1.png)

### 7.4 Some useful predictors

https://otexts.com/fpp3/useful-predictors.html

#### Trend

$y_t = \beta_0 + \beta_1 t + \epsilon_t$

#### Dummy variables

- one-hot encoding
  - can be used for an outlier e.g.
    - 1 for the outlier
    - 0 for the other values

#### Seasonal dummy variables

- for a seasonal categorical variable with n keys, we need n - 1 dummy variables
  - e.g. for the day of week we need 6 dummy variables
    - 100000, 010000, 001000, 000100, 000010, 000001, 000000
    - we know there is no other case
    - the 7th category will be captured by the intercept

#### Intervention variables

- e.g.
  - competitor activity
  - advertising expenditure
  - industrial action
- indicate events with
  - a dummy variable
      - spike variable
        - for one period
      - step variable
        - for permanent effects
  - a trend that bands at the time of intervention
    - which is non-linear
    - also for permanent effects

#### Trading days

- business days for monthly or quarterly data

#### Distributed lags

- advertisement

#### Easter

#### Fourier series

e.g. when 6 predictors are added

$$x_1,t = \sin\left({2\pi t \over m}\right)$$
$$x_2,t = \cos\left({2\pi t \over m}\right)$$
$$x_3,t = \sin\left({4\pi t \over m}\right)$$
$$x_4,t = \cos\left({4\pi t \over m}\right)$$
$$x_5,t = \sin\left({6\pi t \over m}\right)$$
$$x_6,t = \cos\left({6\pi t \over m}\right)$$


- often need fewer predictor variables than using seasonal dummy variables
  - especially when it comes to a large $m$
- called a harmonic regression

### 7.5 Selecting predictors

https://otexts.com/fpp3/selecting-predictors.html

- "NOT" recommended approaches
  - Plotting the forecast variable against a particular predictor and if there is no noticeable relationship, drop that predictor from the model.
  - Doing a multiple linear regression on all the predictors and disregard all variables whose p-values are greater than 0.05.
    - Interpreting p-value
      - to determine whether the relationships observed in the sample also exist in the larger population
      - $H_0$: the independent variable has no correlation with the dependent variable
        - meaning $\beta_{\hat{H_0}} = 0$
      - $t = {\hat{\beta} - \beta_{H_0} \over \operatorname{s.e.}(\hat{\beta})}$
      - compare the p-value to the significance level
    - references
      - https://en.wikipedia.org/wiki/T-statistic
      - [how to interpret p-value of a coefficient in a multiple linear regression](https://statisticsbyjim.com/regression/interpret-coefficients-p-values-regression/)
      - [how to calculate p-value of a coefficient in a multiple linear regression](https://stats.stackexchange.com/a/352501/193645)

#### Adjusted $R^2$

$$
\bar{R}^2=1- {\text{RSS} / \text{df}_\text{res} \over \text{TSS} / \text{df}_\text{tot}} = 1 - (1 - R^2){T - 1 \over T - k - 1}
$$

- where
  - $T$: the number of observations
  - $k$: the number of predictors
- has been widely used
- tends to select too many predictor variables

#### Cross-validation

- cross validation works as expected
  - [Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction.](https://doi.org/10.1016/j.csda.2017.11.003)
- steps (LOOCV)
  - Remove observation t from the data set, and fit the model using the remaining data. Then compute the error ($e = y_t - \hat{y}_t$) for the omitted observation. (This is not the same as the residual because the $t$th observation was not used in estimating the value of $\hat{y}_t$.)
  - Repeat step 1 for $t=1,...,T$.
  - Compute the MSE from $e_1^*,..., e_T^*$.
    - We shall call this the CV.

#### Akaike's Information Criterion

$$
\text{AIC} = -2 \log(\hat{L}) + 2p \overset{?}{=} T \log\left({\text{RSS} \over T}\right) + 2(k  + 2)
$$

- where
  - $\hat{L}$: the maximum value of the likelihood function of the model
    - https://youtu.be/GyllCbuUr5E
  - $T$: the number of observations
    - for large values of $T$, minimising the AIC is equivalent to minimising the `CV` value.
  - $k$: the number of predictors
  - $p = k + 2$: the number of parameters
    - for
      - $k$ predictors
      - 1 intercept
      - 1 variance of the residuals

#### Corrected Akaike's Information Criterion

- For small values of $T$, the AIC tends to select too many predictors, and so a bias-corrected version of the AIC has been developed.

$$
\text{AIC}_c = \text{AIC} + {2p^2 + 2p \over T - p - 1} = \text{AIC} + {2(k + 2)(k+3) \over T - k - 3}
$$

#### Schwarz's Bayesian Information Criterion

$$
\text{BIC} = -2 \log(\hat{L}) + p\log(T) \overset{?}{=} T \log\left({\text{RSS} \over T}\right) + (k  + 2)\log(T)
$$

- derived to serve as an asymptotic approximation to a transformation of the Bayesian posterior probability of a candidate model
- assumptions
  - i.i.d.
  - linear models
  - the likelihood is from the regular exponential family
  - the prior of $\theta$ is flat
- BIC likes models with fewer parameters more than AIC likes those models.
- For large value of $T$, minimising BIC is similar to leave-v-out cross-validation
  - when $v = T[1 - {1 \over \log(T) - 1}]$.
- If there is a true underlying model, the BIC will select that model given enough data.
- popular
- references
  - [The Bayesian information criterion: background, derivation, and applications](https://doi.org/10.1002/wics.199)

#### Which measure should we use?

- In reality, there is rarely a true underlying model.
- Even if there was a true underlying model, selecting that model will not necessarily give the best forecasts.
 - Because the parameter estimates may not be accurate due to Cramér–Rao bound.

#### Best subset regression

- or "all possible subsets" regression

#### Stepwise regression

- Basically we need to try all the combinatinos
- What if there are many combinations?
  - backwards
    - start with the model containing all potential predictors
    - remove one predictor at a time
  - forward
    - start with the model contianing only the intercept
    - add one predictor at a time
  - hybrid
    - start with a model containing a subset of potential predictors

#### Beware of inference after selecting predictors

- Any procedure involving selecting predictors first will invalidate the assumptions behind the p-values when we do the statistical inference of the predictors.

### 7.6 Forecasting with regression

- TODO https://otexts.com/fpp3/forecasting-regression.html

#### Ex-ante versus ex-post forecasts

- Ex-ante forecasts
  - uses only the information that is available in advance
- Ex-post forecasts
  - uses only the information on the predictors

#### Exmaple: Australian quarterly beer production

#### Scenario based forecasting

- assume specific future values for all predictors

#### Building a predictive regression model

$$
y_{t+h}=\beta_{0}+\beta_{1} x_{1, t}+\cdots+\beta_{k} x_{k, t}+\varepsilon_{t+h}
$$

#### Prediction intervals

- For simple regressions, the prediction intervals are calculated as follows.

$$
\hat{y} \pm 1.96 \hat{\sigma}_{e} \sqrt{1+\frac{1}{T}+\frac{(x-\bar{x})^{2}}{(T-1) s_{x}^{2}}}
$$

- It's more certain about our forecasts when considering values of the predictor variable close to its sample mean.

(Example)

![Figure 7.19: Prediction intervals if income is increased by its historical mean of 0.73 % versus an extreme increase of 12%.](https://otexts.com/fpp3/fpp_files/figure-html/conSimplePI-1.png)

(Derivation)

https://www.sjsu.edu/faculty/guangliang.chen/Math261a/Ch2slides-simple-linear-regression.pdf

- simple regression model (population)
  - $y = \beta_0 + \beta_1x + \epsilon$
- sample regression model
  - $y_i = \beta_0 + \beta_1 x_i + \epsilon_i$
    - $i = 1, ..., n$
    - Depending on the context, $y_i$ can be either a random variable or an observed value of it.
- notations
  - $\bar{x} = \sum\limits_{i=1}^n {x_i}$
  - $\bar{y} = \sum\limits_{i=1}^n {y_i}$
  - $S_{xx} = \sum\limits_{i=1}^n (x_i - \bar{x})^2 = \sum\limits_{i=1}^n {x_i}^2 - n\bar{x}^2$
  - $S_{xy} = \sum\limits_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) = \sum\limits_{i=1}^n {x_i y_i} - n\bar{x}\bar{y}$
- Least-sqaure (LS) estimators
  - $\min\limits_{\hat{\beta}_0,\hat{\beta}_1} \overset{\text{def}}=\sum\limits_{i=1}^n(y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i))^2$
  - solutions:
    - $\hat\beta_0 = \bar{y} - \beta_1 \bar{x}$
    - $\hat\beta_1 = {S_{xy} \over S_{xx}}$
  - regression line
    - $y = \hat\beta_0 + \hat\beta_1 x = \bar{y} + \hat\beta_1(x - \bar{x})$
  - TODO p13 (homework 2.25)
    - https://www.sjsu.edu/faculty/guangliang.chen/Math261a/Ch2slides-simple-linear-regression.pdf

## References

- [MATH 261A: Regression Theory and Methods](https://www.sjsu.edu/faculty/guangliang.chen/Math261a.html)
  - San Jose State University, Fall 2020





















