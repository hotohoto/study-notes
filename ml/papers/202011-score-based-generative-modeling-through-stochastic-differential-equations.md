# Score-Based Generative Modeling through Stochastic Differential Equations

- https://arxiv.org/abs/2011.13456
- ICLR 2021, Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole
- both SMLD and DDPM can be seen in the perspective of SDE
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-noise-conditioned-score-networks-ncsn
- contributions

  - unified framework generalizing NCSNs and DDPMs
  - flexible sampling
    - general-purpose SDE solvers
    - predictor-corrector samplers
    - deterministic samplers
  - controllerble generation without retraining

# TODO

- Eq. 29. derivation.. 


## 3 Score-based generative modeling with SDEs

### 3.1 Perturbing data with SDEs

- $d\mathbf{x}=\mathbf{f}(\mathbf{x}, t) dt+g(t) d\mathbf{w}$
  - $\mathbf{w}$
    - the standard wiener process
  - $\mathbf{f}(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$
    - the drift coefficient of $\mathbf{x}(t)$
  - $g: \mathbb{R} \to \mathbb{R}$
    - the diffusion coefficient of $\mathbf{x}(t)$
  - This SDE has a unique solution as long as the coefficients are globally Lipschitz in both state and time.
- $d\mathbf{x}=\mathbf{f}(\mathbf{x}, t) dt + \mathbf{G}(\mathbf{x}, t) d\mathbf{w}$
  - (for more general coefficients)
- $p_{st}(\mathbf{x}(t) \vert \mathbf{x}(s))$
  - transition kernel from $\mathbf{x}(s)$ to $\mathbf{x}(t)$

### 3.2 Generating samples by reversing the SDE

- $d\mathbf{x}=\left[\mathbf{f}(\mathbf{x}, t)-g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt+g(t) d\bar{\mathbf{w}}$
  - $\bar{\mathbf{w}}$
    - a standard Wiener process when time flows backwards fro $T$ to $0$
  - $dt$
    - an infinifestimal negative timestep
- $d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} dt+\mathbf{G}(\mathbf{x}, t) d\bar{\mathbf{w}}$

### 3.3 Estimating scores for the SDE

- $\boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \min } \mathbb{E}_t\left\{\lambda(t) \mathbb{E}_{\mathbf{x}(0)} \mathbb{E}_{\mathbf{x}(t) \mid \mathbf{x}(0)}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)-\nabla_{\mathbf{x}(t)} \log p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0))\right\|_2^2\right]\right\}$
  - Assumes the drift and diffusion coefficient of an SDE are affine
- $\boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \min } \mathbb{E}_t\left\{\lambda(t) \mathbb{E}_{\mathbf{x}(0)} \mathbb{E}_{\mathbf{x}(t)} \mathbb{E}_{\mathbf{v} \sim p_{\mathbf{v}}}\left[\frac{1}{2}\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)\right\|_2^2+\mathbf{v}^{\top} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t) \mathbf{v}\right]\right\}$

### 3.4 Examples: VE, VP SDEs and beyond

- VE SDE
  - $d\mathbf{x}=\sqrt{\frac{d\left[\sigma^2(t)\right]}{dt}} ~d\mathbf{w}$
  - Variance exploding SDE
  - A continuous generalization of SMLD
  - e.g.
    - SMLD
      - $d\mathbf{x}=\sigma_{\min }\left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)^t \sqrt{2 \log \frac{\sigma_{\max }}{\sigma_{\min }}} d\mathbf{w}, \quad t \in(0,1]$
      - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0))=\mathcal{N}\left(\mathbf{x}(t) ; \mathbf{x}(0), \sigma_{\min }^2\left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)^{2 t} \mathbf{I}\right), \quad t \in(0,1]$
    - NCSN++
      - their optimal architecture for the VE SDE
    - NCSN++ cont.
      - trained the score network using the continuous loss function in Eq. (7)
- VP SDE
  - $d\mathbf{x}=-\frac{1}{2} \beta(t) \mathbf{x} dt+\sqrt{\beta(t)} d\mathbf{w}$
  - Variance preserving SDE
  - A continuous generalization of DDPM
  - e.g.
    - DDPM
      - $d\mathbf{x}= -\frac{1}{2} (\bar{\beta}_{\min } + t (\bar{\beta}_{\max } - \bar{\beta}_{\min })) \mathbf{x} dt + \sqrt{\bar{\beta}_{\min } + t (\bar{\beta}_{\max } - \bar{\beta}_{\min })}d\mathbf{w}, \quad t \in(0,1]$
      - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0)) =\mathcal{N}\left(\mathbf{x}(t) ; e^{-\frac{1}{4} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-\frac{1}{2} t \bar{\beta}_{\min }} \mathbf{x}(0), \mathbf{I}-\mathbf{I} e^{-\frac{1}{2} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-t \bar{\beta}_{\min }}\right), \quad t \in[0,1]$
    - DDPM++
      - their optimal architecture for the VP SDE
- Sub-VP SDE
  - $d\mathbf{x}=-\frac{1}{2} \beta(t) \mathbf{x} dt+\sqrt{\beta(t)\left(1-e^{-2 \int_0^t \beta(s) ds}\right)} d\mathbf{w}$
  - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0)) =\mathcal{N}\left(\mathbf{x}(t) ; e^{-\frac{1}{4} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-\frac{1}{2} t \bar{\beta}_{\min }} \mathbf{x}(0), \left[1 - e^{-\frac{1}{2} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-t \bar{\beta}_{\min }}\right]^2\mathbf{I}\right), \quad t \in[0,1]$
  - The variance is always bounded by the VP SDE at every intermediate time step
    - (less variance than of VP SDE)
      - the variance of the marginal distributions of Sub-VP SDE goes up slower than the variance of the marginal distributions of VP SDE as $t$ grows
      - the mean of marginal distributions of Sub-VP SDE goes to zero faster than the mean of VP SDE's marginal distribution as $\sigma$ grows
  - seems better in terms of likelihoods
  - especially for low resolution images

## 4 Solving the reverse SDE

### 4.1 General-purpose numerical SDE solvers

- Euler-Maruyama method
- stochastic Runge-Kutta method
- **ancestral sampling**
  - the same as the DDPM sampler
  - just a special discretization of the reverse-time VP/VE SDE
    - can be applied to SMLD
- **reverse diffusion samplers**
  - discretize the reverse-time SDE in the same way as the forward one

### 4.2 Predictor-corrector samplers

- prediction
  - the solution of a numerical SDE solver
- correction
  - score based MCMC

### 4.3 Probability flow and connection to neural ODEs
- Probability flow
  - $d\mathbf{x}=\left[\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$
  - $d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} \nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\boldsymbol{\top}}\right]-\frac{1}{2} \mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} dt$
  - Its trajactories shares the same marginal probability densities $\{p_t(\mathbf{x})\}_{t=0}^T$ with the original SDE
  - Derived via Fokker-Planck equations
    - We find the Fokker-Planck equation from diffusion SDE
    - We find ODE that generates the same Fokker-Plank equation
  - The Fokker-Planck equation can be derived from a general SDE
    - using Itô's lemma and integration by parts
    - references
      - [Fokker Planck Equation Derivation](https://youtu.be/MmcgT6-lBoY)
      - [Ito's calculus](https://en.wikipedia.org/wiki/It%C3%B4_calculus)
      - [Ito's lemma](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma)
  
- Exact likelihood computation
  - Now we can calculate likelihood in a deterministic way
  - how?
    - 1 sample $\mathbf{x}(T)$
    - 2 obtain $\mathbf{x}(t)$ by solving the probability flow ODE
    - 3 calculate the log likelihood
      - $\log p_0(\mathbf{x}(0))=\log p_T(\mathbf{x}(T))+\int_0^T \nabla \cdot \tilde{\mathbf{f}}_{\boldsymbol{\theta}}(\mathbf{x}(t), t) dt$
  - Computing $\nabla \cdot \tilde{\mathbf{f}}_{\boldsymbol{\theta}}$ is expensive
    - estimate it with Skilling-Hutchinson trace estimator
      - $\nabla \cdot \tilde{\mathbf{f}}_{\boldsymbol{\theta}}(\mathbf{x}, t)=\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\boldsymbol{\epsilon}^{\top} \nabla \tilde{\mathbf{f}}_\theta(\mathbf{x}, t) \boldsymbol{\epsilon}\right]$
  
- Manipulating latent representations
  - interpolation
  - temperature rescaling (by modifying norm of embedding)
  
- Uniquely identifiable encoding
  - How?
    - Given a outcome from the dataset
    - Obtain $\mathbf{x}(T)$ using the probability flow ODE
  - For the same inputs, Model A and Model B provide encodings that are close in every dimension
    - despite having different model architectures and training runs
  
- Efficient sampling
  - How?
    - Given
      - $d\mathbf{x}=\mathbf{f}(\mathbf{x}, t) dt+\mathbf{G}(t) d\mathbf{w}$
    - Discretize it and we get
      - $\mathbf{x}_{i+1}=\mathbf{x}_i+\mathbf{f}_i\left(\mathbf{x}_i\right)+\mathbf{G}_i \mathbf{z}_i, \quad i=0,1, \cdots, N-1$
    - Plug the coefficients to the probability flow ODE
      - $\mathbf{x}_i=\mathbf{x}_{i+1}-\mathbf{f}_{i+1}\left(\mathbf{x}_{i+1}\right)+\frac{1}{2} \mathbf{G}_{i+1} \mathbf{G}_{i+1}^{\top} \mathbf{s}_{\theta^*}\left(\mathbf{x}_{i+1}, i+1\right), \quad i=0,1, \cdots, N-1$
    
  - examples
    - SMLD
      - $\mathbf{x}_i=\mathbf{x}_{i+1}+\frac{1}{2}\left(\sigma_{i+1}^2-\sigma_i^2\right) \mathbf{s}_{\theta^*}\left(\mathbf{x}_{i+1}, \sigma_{i+1}\right), \quad i=0,1, \cdots, N-1$
    - DDPM
      - $\mathbf{x}_i=\left(2-\sqrt{1-\beta_{i+1}}\right) \mathbf{x}_{i+1}+\frac{1}{2} \beta_{i+1} \mathbf{s}_{\theta^*}\left(\mathbf{x}_{i+1}, i+1\right), \quad i=0,1, \cdots, N-1$
  
### 4.4 Architecture improvements

- NCSN++
  - Finite Impulse Response (FIR) upsampling/downsampling
  - rescaled skip connection
  - BigGAN-type residual blocks
  - 4 residual blocks per resolution (instead of 2)
  - use residual for input
  - no progressive growing architecture for output
- DDPM++
  - rescaled skip connection
  - BigGAN-type residual blocks
  - 4 residual blocks per resolution (instead of 2)
  - use residual for input
- NCSN++ cont. deep
  - best FID on CIFAR-10
- DDPM++ cont. deep
  - best NLL on CIFAR-10
- (naming)
  - ++
    - for the best architecture they found
  - deep
    - for  doubling the network depth
  - cont.
    - for training the score network via the continuous version of the loss
- more details
  - 1.3M iterations
  - save one checkpoint per 50k iterations
  - FID
    - on 50k samples
  - batch_size=128
  - anti-aliasing based on Finite Impulse Response (FIR)
  - StyleGAN-2 hyper parameters
  - progressive architecture implemented according to StyleGAN-2
  - The Exponential Moving Average (EMA) rate
    - 0.999 for VE
    - 0.9999 for VP
- High resolution images
  - 1024 x 1024 CelebA-HQ
  - NCSN++
  - batch_size=8
  - 2.4 M iterations
  - EMA rate
    - 0.9999

## 5 Controllable generation

$$
d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-g(t)^2\left[\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + \nabla_{\mathbf{x}} \log p_t(\mathbf{y}\vert \mathbf{x})\right]\right\} dt+g(t) d\bar{\mathbf{w}}\tag{14}
$$

$$
d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x}\vert \mathbf{y})\right\} dt+\mathbf{G}(\mathbf{x}, t) d\bar{\mathbf{w}}\tag{48}
$$

$$
\nabla_\mathbf{x}\log p_t(\mathbf{x}\vert \mathbf{y}) = \nabla_\mathbf{x}\log p_t(\mathbf{x}) + \nabla_\mathbf{x}\log p_t(\mathbf{y}\vert \mathbf{x})\tag{49}
$$



- with an auxiliary classifier
  - Class-conditional sampling
- without an auxiliary classifier
  - Imputation
    - separate $\mathbf{x}$, $\mathbf{f}$, $\mathbf{G}$ into known and unknown dimensions
  - Colorization
    - transform the image to map the gray-scale image to a separate channel (known dimension)
      - a 3x3 orthogonal matrix is used to decouple known data dimensions
    - learn the other two channels (unknown dimension)
    - reverse-transform the learned image
- Solving general inverse problem
  - similar to the class-conditional sampling problem
    - requires assumptions
      - $p(\mathbf{y}(t) \vert \mathbf{y})$ is tractable
      - $p_t(\mathbf{x}(t) \vert \mathbf{y}(t), \mathbf{y}) \approx p_t(\mathbf{x}(t) \vert \mathbf{y}(t))$
        - notes
          - for small $t$, these two are almost the same
          - for large $t$, the discrepancy matters less for the final sample


## Resources

- https://www.math.snu.ac.kr/~syha/Lecture-4.pdf
- https://youtu.be/yqF1IkdCQ4Y?t=3459