# Stochastic process and timeseries forecasting

## stochastic process

### Auto correlation

- Auto correlation
  - in [-1, 1]
  - also known as serial correlation
- Partial auto correlation
  - the correlation that results after removing the effect of any correlations due to the terms at shorter lags.
  - useful for selecting `p` for AR/ARIMA models

(Reference)

- https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

### Stationarity

- strictly stationary process (strong stationary process)
- wide-sense stationary process (weak stationary process)
- ergodicity
  - $\lim\limits_{N \to \infty} {1 \over N} \sum\limits^{N} Y_t = E[Y_t]$
  - $\lim\limits_{N \to \infty} {1 \over N} \sum\limits^{N} Y_t Y_{t+k} = E[Y_t Y_{t+k}]$
- non-stationary process

### conditional expectation

### Martingales in Discrete Time

### Maringale inequality and convergence

### Markov Chain

### Stochastic process in continuous time

### Itô stchastic calculus

## References

