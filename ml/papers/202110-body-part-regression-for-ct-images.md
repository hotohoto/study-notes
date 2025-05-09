# Body Part Regression for CT Images

- https://arxiv.org/abs/2110.09148
- self supervised learning model
  - predicts a latent variable
  - a predicted value is transformed into the [1, 100] scale
    - with respect to the range from the start of pelvis to the end of eye sockets
  - trained by a pseudo objective to optimize the relative positions of slices

## Notations

- $s_{ij}$
  - $i$
    - volume index
  - $j$
    - slice index
- $\Delta s_{ij} = s_{ij+1} - s_{ij}$
- $N$
  - number of available CT volumes
- $M_i$
  - number of slices of volume $i$

## 4 Materials and Methods

### 4.1 Body Part Regression

#### 4.1.1 Dataset selection

#### 4.1.2 Annotations

$l_k$

- a landmark class
- $k \in \{1, ..., 12\}$

#### 4.1.3 Learning procedure

- $\Delta h$
  - the physical distance in mm between any pair of slices
- $z$
  - z-spacing which is the physical distance in mm between two consecutive slices
- $k = \operatorname{round} \left( {\Delta h \over z} \right)$
  - index difference between any pair of slices

#### 4.1.4 Order loss

##### Classification order loss

##### Heuristic order loss

#### 4.1.5 Data augmentations

### 4.2 Evaluation metrics

The pseudo label $\bar{s}_{l_k}$ is defined as below.
$$
\bar{s}_{l_k} = {1 \over N_{l_k}} \sum\limits_{i=1}^{N_{l_k}} s_{il_k}
$$

- where
  - it is calculate in the training set
  - $N _{l _k}$
    - number of CT volumes where the landmark $l _k$ was annotated.

#### 4.2.1 Landmark mean square error

$$
s^\star = {100 \cdot (s-\bar{s} _{l _1}) \over \bar{s}_{l_{12}} - \bar{s}_{l_{1}}}
\tag{4.12}
$$

The landmark mean square error (LMSE) of volume $i$ is defined as
$$
\begin{aligned}
\phi_i & =\frac{1}{\left|L_i\right|} \sum_{l_k \in L_i}\left(\bar{s}_{l_k}^{\star}-s_{i l_k}^{\star}\right)^2 \\
& =\frac{1}{\left|L_i\right|} \sum_{l_k \in L_i}\left(\frac{\bar{s}_{l_k}-\bar{s}_{l_1}-\left(s_{i l_k}-\bar{s}_{l_1}\right)}{d}\right)^2 \\
& =\frac{1}{\left|L_i\right|} \sum_{l_k \in L_i}\left(\frac{\bar{s}_{l_k}-s_{i l_k}}{d}\right)^2,
\end{aligned}
\tag{4.13}
$$
.