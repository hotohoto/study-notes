# Optimization

## Method of Lagrange Multiplier

- We want to minimize $f(x)$
- constraint:
  - $h_i(x) = 0$
- define the Lagrange function as
  - $L(x, \lambda) = f(x) + \sum_i\lambda_i\,h_i(x)$
- $x^{*}$ is local minimum $\Leftrightarrow$ $\lambda^{*}$ exists s.t.
  - $\nabla_{x} L(x^{*},\lambda^{*}) = 0$
  - $\nabla_{\lambda_i} L(x^{*},\lambda^{*}) = 0$
  - $y^{T}(\nabla_{xx}^2 L(x^{*}, \lambda^{*}))y \ge 0$
    - $\forall y \text{ s.t. } \nabla_{x} h(x^{*})^{T}y = 0$
    - To be positive definite for the critical points of h(x)

## Lagrangian dual problem

Make a problem from the primary problem for them to be equivalent.

- primary problem
  - We want to minimize $f(x)$
  - constraints (feasible area):
    - $g_i(x^{*}) <= 0$
  - case 1
    - $x^{*}$ in feasible set
      - set the KKT multipliers to 0 to ignore the conditions
  - case 2
    - $x^{*}$ on the border of feasible set
- dual problem
  - $\operatorname{max}_{\mu}\operatorname{min}_x f(x) + \sum_{i}  \mu_i g(x) + \sum_{j} \lambda_j h(x)$
    - dual function
      - $\operatorname{min}_x f(x) + \sum_{i}  \mu_i g(x) + \sum_{j} \lambda_j h(x)$
      - concave funciton with respect to $\mu$
      - that's why we use $\operatorname{max}_\mu$
  - for each condition $g_i(x)$:
    - case 1
      - $x^{*}$ in feasible set
      - $\nabla_x f(x^{*}) = 0$
      - meaning $\mu_i = 0$ in the dual problem
    - case 2
      - $x^{*}$ on the border of feasible set
      - $\mu_i$ for $\nabla_x f(x^{*})$ and $\nabla_x g_i(x^{*})$ to be opposite direction
      - meaning $\mu_i \gt 0$ in the dual problem

Refer to [KKT conditions at Wikipedia](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions).

## Projection-type optimization methods

## Penalty-type optimization methods

## Variational optimization

- Computer vision involves usually ill-posed problems. (There is no deterministic only one solution.)
- Variational approach can solve those ill-posed problems by incorporating prior knowledge.
- can be considered as infinite dimensional problem

- functional
  - a function of functions.
- noise
  - unwanted signal
- TV regularization model

### Tikhonov model

- blur edges

### Total variation minimization

- reserves edges

#### ROF model

- one way of solving TV minimization

####

## Other notes

https://www.youtube.com/playlist?list=PL3940DD956CDF0622

Known Optimization problems

- Least Square problem
- Linear programming
  - https://en.wikipedia.org/wiki/Linear_programming
- Convex optimization
- quadratic programming

Affine Set

- line or plane

Convex Set

- a set which includes 2 points and the segment between them

Convex Combination

Convex hull

Convex Cone

Polyhedra


https://en.wikipedia.org/wiki/Ellipsoid