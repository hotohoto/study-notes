# Information theory

"Information is difference that makes a difference."

## units of information

- bit
- nat
  - natural unit of information

## self-information

$$
I_X(x) := -\log p_X(x)
$$

- surprisal or amount of information gained when it is sampled.
- base of logarithms depends on the unit of information
  - use 2 for bit
  - use $e$ for nat

## entropy

$$
H(X)=\mathbb {E} _{X}[I(x)]=-\sum _{x\in \mathbb {X} }p(x)\log p(x).
$$

- expected amount of information
- expected number of questions to ask to differentiate symbols

## joint entropy

$$
{\displaystyle \mathrm {H} (X,Y)=-\sum _{x\in {\mathcal {X}}}\sum _{y\in {\mathcal {Y}}}P(x,y)\log _{2}[P(x,y)]}
$$

- a measure of the uncertainty associated with a set of variables.

## conditional entropy

$$
{\displaystyle \mathrm {H} (Y|X)\ =-\sum _{x\in {\mathcal {X}},y\in {\mathcal {Y}}}p(x,y)\log {\frac {p(x,y)}{p(x)}}}
$$

- the amount of information needed to describe the outcome of a random variable $Y$ given that the value of another random variable $X$ is known.

- (chain rule)

  - $\mathrm{H}(Y|X)=\mathrm{H}(X,Y)−\mathrm{H}(X)$

  

## pointwise mutual information

$$
\operatorname{pmi} (x;y)\equiv \log {\frac {p(x,y)}{p(x)p(y)}}=\log {\frac {p(x|y)}{p(x)}}=\log {\frac {p(y|x)}{p(y)}}
$$

## mutual information

- Amount of information obtained about one random variable through observing the other random variable.
- The mutual information (MI) of the random variables X and Y is the expected value of the PMI (over all possible outcomes).
- range: $[0, \infty)$

$$
I(X;Y)=D_{\mathrm {KL} }(P_{(X,Y)}\|P_{X}\otimes P_{Y})
$$

For 2 discrete distributions and their random variables $X$ and $Y$,
$$
I(X;Y)=\sum _{y\in {\mathcal {Y}}}\sum _{x\in {\mathcal {X}}}{p_{(X,Y)}(x,y)\log {\left({\frac {p_{(X,Y)}(x,y)}{p_{X}(x)\,p_{Y}(y)}}\right)}}
$$
.

For 2 continuous distributions and their random variables,
$$
I(X;Y)=\int _{\mathcal {Y}}\int _{\mathcal {X}}{p_{(X,Y)}(x,y)\log {\left({\frac {p_{(X,Y)}(x,y)}{p_{X}(x)\,p_{Y}(y)}}\right)}}\;dx\,dy
$$
.

## KL-divergence

$$
D_{\text{KL}}(P\parallel Q)=-\sum _{x\in {\mathcal {X}}}P(x)\log \left({\frac {Q(x)}{P(x)}}\right)
$$

or

$$
D_{\text{KL}}(P\parallel Q)=\int _{-\infty }^{\infty }p(x)\log \left({\frac {p(x)}{q(x)}}\right)\,dx
$$

- Not a distance metric
- non-commutative
- find how much it is losing information on average
- left term: target distribution $P$
- right term: estimated distribution $Q$

## cross entropy

$$
H(p,q)=E_{p}[-\log q]
$$

or

$$
H(p,q)=H(p)+D_{\mathrm {KL} }(p\|q)
$$
.

For discrete probability distributions $p$ and $q$ with the same support ${\mathcal {X}}$,

$$
H(p,q)=-\sum _{x\in {\mathcal {X}}}p(x)\,\log q(x)
$$
.

## Rate-distortion theory

- https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory
- the foundation of lossy-compression
- settings
  - $f_n$: encoder
  - $g_n$: decoder
  - $X^n$: input sequence for encoder $f_n$. Each of them $X$ is a random variable.
  - $Y^n$: output sequence for encoder $f_n$. Each of them $Y$ is a random variable.
  - $\hat{X}^n$: the reconstructed output sequence from decoder $g_n$. Each of them $\hat{X}$ is a random variable
  - $R$: minimum number of bits per symbol that should be communicated over a channel. we want to minimize it.
  - $D$: expected distortion between $X_n$ and $\hat{X}_n$ that we're not supposed to exceed
- distortion function
  - Hamming distortion
    - $d(x, \hat{x})$
      - $0$ if $x = \hat{x}$
      - $1$ if $x \neq \hat{x}$
  - Squared-error distortion
    - $d(x, \hat{x}) = (x - \hat{x})^2$
- rate-distortion function
  - $\inf\limits_{Q_{Y \mid X}} I_Q(Y; X)$ subject to $D_Q \le D*$
  - $I_Q(Y; X)$: mutual information between $Y$ and $X$
    - $I(Y; X) = H(Y) - H(Y|X)$
      - min: 0
      - max: $\infty$
  - $D^*$: maximum distortion
  - $D_Q$: distortion between $X_n$ and $Y_n$?? TODO

## References

- https://en.wikipedia.org/wiki/Information_content
- https://en.wikipedia.org/wiki/Mutual_information
- https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
- https://en.wikipedia.org/wiki/Cross_entropy
- [가장 쉬운 KL Divergence 완전정복!](https://youtu.be/Dc0PQlNQhGY)
