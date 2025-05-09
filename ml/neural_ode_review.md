paper

- https://arxiv.org/abs/1806.07366

explanation

- https://youtu.be/AD3K8j12EIE
  - https://github.com/llSourcell/Neural_Differential_Equations
- https://blog.acolyer.org/2019/01/09/neural-ordinary-differential-equations/

implementation

- https://github.com/rtqichen/torchdiffeq

etc

- https://mailchi.mp/technologyreview/a-new-type-of-deep-neural-network-that-has-no-layers
- http://akosiorek.github.io/ml/2018/04/03/norm_flows.html
- https://news.ycombinator.com/item?id=18676986
- https://blog.acolyer.org/2019/01/09/neural-ordinary-differential-equations/

TODO

- try to use odeint
- analyze Siraj code
- read paper

Questions

- Why is this model better than LSTN for time series?

Summary

- Oridinary Derivative Equation
  - a differential equation containing one or more functions of one independent variable and its derivatives.
  - The term ordinary is used in contrast with the term partial differential equation which may be with respect to more than one independent variable.
- Partial Derivative Equation
  - a differential equation that contains beforehand unknown multivariable functions and their partial derivatives.
  - A special case is ordinary differential equations (ODEs), which deal with functions of a single variable and their derivatives.
- Euler method
  - approximation using computer to solve (difficult) DE
  - https://www.khanacademy.org/math/ap-calculus-bc/bc-differential-equations-new/bc-7-5/v/eulers-method
  - a numerical method to solve first order first degree differential equation with a given initial value.
  - The Euler method is a first-order method, which means that the local error (error per step) is proportional to the square of the step size
  - and the global error (error at a given time) is proportional to the step size.
