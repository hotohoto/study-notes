# Improving Diffusion Models for Inverse Problems using Manifold Constraints

- https://arxiv.org/abs/2206.00941

- NeurIPS 2022

- Projection Onto Convex Sets (POCS)

  - projects point $\boldsymbol{x}$ to the plane $\boldsymbol{y} = \boldsymbol{H}\boldsymbol{x}$
  - $\boldsymbol{y}$
    - measurement
  - the conventional extra update we did in ILVR or CCDF which we called measurement consistency step
  - it dates back to further in the old days.

- Manifold Constrained Gradient (MCG) 

  - $- \alpha {\frac {\partial} {\partial \boldsymbol{x}_i}} \Vert \boldsymbol{W}(\boldsymbol{y - \boldsymbol{H}\hat{\boldsymbol{x}}_0})\Vert_2^2$
    - $i$
      - current reverse diffusion step index
    - $\boldsymbol{x}_i$
      - the estimate of the current reverse diffusion step $i$
      - it might by noisy when $i > 0$
    - $\hat{\boldsymbol{x}}_0$
      - direct estimate of $\boldsymbol{x}_0$

- theorem 1

  - $\frac{\partial}{\partial \boldsymbol{x}_i}\left\|\boldsymbol{W}\left(\boldsymbol{y}-\boldsymbol{H} \hat{\boldsymbol{x}}_0\right)\right\|_2^2=-2 \boldsymbol{J}_{Q_i}^T \boldsymbol{H}^T \boldsymbol{W}^T \boldsymbol{W}\left(\boldsymbol{y}-\boldsymbol{H} \hat{\boldsymbol{x}}_0\right) \in T_{\hat{\boldsymbol{x}}_0} \mathcal{M}$
    - $Q_i$
      - $\boldsymbol{x}_i \mapsto \hat{\boldsymbol{x}}_0$
      - projection to the estimated data manifold
    - $\boldsymbol{J}$
      - Jacobian of $Q_i$
  - MCG is in the tangent plane of $\hat{\boldsymbol{x}}_0$'s manifold space

- algorithm

$$
\boldsymbol{x}_{i-1}^{\prime}=\boldsymbol{f}\left(\boldsymbol{x}_i, \boldsymbol{s}_\theta\right)-\alpha \frac{\partial}{\partial \boldsymbol{x}_i}\left\|\boldsymbol{W}\left(\boldsymbol{y}-\boldsymbol{H} \hat{\boldsymbol{x}}_0\left(\boldsymbol{x}_i\right)\right)\right\|_2^2+g\left(\boldsymbol{x}_i\right) \boldsymbol{z}, \quad \boldsymbol{z} \sim \mathcal{N}(0, \boldsymbol{I})\tag{14}
$$

$$
\boldsymbol{x}_{i-1}=\boldsymbol{A} \boldsymbol{x}_{i-1}^{\prime}+\boldsymbol{b}\tag{15}
$$



- basic idea

  - points in the noisy data manifolds are mostly in a dense area due to high dimensionality
  - our score network will be trained mostly on these dense area
  - conventional POCS is not enough in that it will move intermediate estimates into a subspace where the score network is not trained on
  - So let's do a gradient step which would move the current estimate around the current noisy manifold toward $\boldsymbol{y} = \boldsymbol{H}\boldsymbol{x}$

- resources

  - https://youtu.be/mjfinYDJTMc
  - https://sang-yun-lee.notion.site/Improving-Diffusion-Models-for-Inverse-Problems-using-Manifold-Constraints-01e82afda989428e8b5faad1c3bbebf2 