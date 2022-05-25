# Information theory

"Information is difference that makes a difference."

## units of information

- bit
- nat
  - natural unit of information

## self-information

$$
\operatorname{I}_X(x) := -\log p_X(x)
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
H(X,Y)
$$

- a measure of the uncertainty associated with a set of variables.

## conditional entropy

$$
H(Y\mid X)
$$

- the amount of information needed to describe the outcome of a random variable $Y$ given that the value of another random variable $X$ is known.

## pointwise mutual information

$$
\operatorname {pmi} (x;y)\equiv \log {\frac {p(x,y)}{p(x)p(y)}}=\log {\frac {p(x|y)}{p(x)}}=\log {\frac {p(y|x)}{p(y)}}
$$

## mutual information

- Amount of information obtained about one random variable through observing the other random variable.
- The mutual information (MI) of the random variables X and Y is the expected value of the PMI (over all possible outcomes).

$$
I(X;Y)=D_{\mathrm {KL} }(P_{(X,Y)}\|P_{X}\otimes P_{Y})
$$

For 2 discrete distributions and their random variables $X$ and $Y$,
$$
\operatorname {I} (X;Y)=\sum _{y\in {\mathcal {Y}}}\sum _{x\in {\mathcal {X}}}{p_{(X,Y)}(x,y)\log {\left({\frac {p_{(X,Y)}(x,y)}{p_{X}(x)\,p_{Y}(y)}}\right)}}
$$
.

For 2 continuous distributions and their random variables,
$$
\operatorname {I} (X;Y)=\int _{\mathcal {Y}}\int _{\mathcal {X}}{p_{(X,Y)}(x,y)\log {\left({\frac {p_{(X,Y)}(x,y)}{p_{X}(x)\,p_{Y}(y)}}\right)}}\;dx\,dy
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

- Not distance metric
- non-commutative
- find how much it is losing information on average
- left term: target distribution $P$
- right term: estimated distribution $Q$

## cross entropy

$$
H(p,q)=\operatorname {E} _{p}[-\log q]
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

## References

- https://en.wikipedia.org/wiki/Information_content
- https://en.wikipedia.org/wiki/Mutual_information
- https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
- https://en.wikipedia.org/wiki/Cross_entropy

(수리통계학 강의)

- 부산대 수리통계학
  - (강의)
    - [2014-2 수리통계학 II (김충락 교수)](https://oer.pusan.ac.kr/2014-2-%EC%88%98%EB%A6%AC%ED%86%B5%EA%B3%84%ED%95%99-II-%EA%B9%80%EC%B6%A9%EB%9D%BD-%EA%B5%90%EC%88%98)
    - [2014-1 수리통계학 I (김충락 교수)](https://oer.pusan.ac.kr/2014-1-%ec%88%98%eb%a6%ac%ed%86%b5%ea%b3%84%ed%95%99-I-%ea%b9%80%ec%b6%a9%eb%9d%bd-%ea%b5%90%ec%88%98)
  - (강의노트)
    - https://crkim.pusan.ac.kr/crkim/24914/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGY3JraW0lMkY0OTc4JTJGODE4NjA4JTJGYXJ0Y2xWaWV3LmRvJTNG
    - https://crkim.pusan.ac.kr/crkim/24914/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGY3JraW0lMkY0OTc4JTJGNzI1NTM0JTJGYXJ0Y2xWaWV3LmRvJTNG
