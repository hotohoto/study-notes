# Singular Spectrum Analysis (SSA)

https://en.wikipedia.org/wiki/Singular_spectrum_analysis

## Methodology

- N
  - length of timeseries
- L
  - window length
- K
  - K = N - L + 1
- trajectory matrix X
  - L by K
  - Hankel matrix
- $C_X$
  - lag covariance matrix
  - M by M
  - Toeplitz matrix
- $N' = N - M + 1$
- Empirical Orthogonal Function (EOF)
  - Eigenvectors of $C_X$
- stationary process assumption
  - unconditional join probability distribution does not change when shifted in time

- Toeplitz matrix
  - each descending diagonal from left to right is constant
- Hankel matrix
  - each ascending skew-diagonal from left to right is constant


### Decomposition and reconstruction

### Multivariate extension (1)

- Mult-channel SSA (M-SSA)
- Extended EOF (EEOF)

### Prediction

- Maximum Entropy Method

### Spatio-temporal gap filling

## As a model free tool

### Basic SSA

#### Main algorithm

1. embedding
2. SVD
3. Eigentripple groupping
4. Diagonal averaging

#### Theory of separability

### Frecasting by SSA

### Multivariate extension (2)

#### MSSA and casuality

#### MSSA and EMH

#### MSSA, SSA, and business cycles

#### MSSA, SSA, and unit root

### Gap-filling

### Detection of structural changes

### Relation between SSA and other methods

Autoregression
spectral Fourier Analysis
Linear Recurrance Relations
Signal subspace methods
State space models
Independent Component Analysis
Regression
Linear Filters
Density Estimation

## Questions

## Other decomposition methods

- STL decomposition
  - seasonal / trend / l??
- FFT, DFT
- SSA


## References

- [2020 Particularities and commonalities of singular spectrum analysis as a method of time series analysis and signal processing](https://arxiv.org/abs/1907.02579)
- [2001 Analysis of Time Series Structure: SSA and Related Techniques](https://books.google.co.kr/books?id=L0HjfJCIrNYC&lpg=PR9&ots=HdGWJsckhk&lr&pg=PP1#v=onepage&q&f=false)


