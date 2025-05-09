# Probabilistic Graphical Models

https://ermongroup.github.io/cs228-notes/

## Preliminaries

### Introduction

#### Probabilistic modeling

#### The difficulties of probabilistic modeling

- probabilities are inherently exponentially-sized objects;
- so, make simplifying assumptions about their structure.

#### Describing probabilities with graphs

#### A bird’s eye overview of the course

- Representation
  - directed: Bayesian networks
  - undirected: Markov Random Fields
- Inference
- Learning

### Probability review

#### Chain rule

$p(x_1, x_2, \cdots, x_n) = p(x_1)p(x_2|x_1)\cdots p(x_n|x_1, x_2, \cdots, x_n)$

### Real-world applications

https://ermongroup.github.io/cs228-notes/preliminaries/applications/

#### Images

p(x): probability distribution

- Generation
  - p(x)
- In-painting
  - p(image| patch)
- Image denoising
  - p(original image| noisy image)

#### Language models

p(x): probability distribution over sequence of words or characters that assignes high probability to proper sententeces.

- Generation
- Translation
  - p(y|x)

#### Audio models

p(x): probability distribution over audio signals that assigns high probability to ones that sound like human speech.

- Upsampling or super-resolution
  - p(I|O)
    - I: intermediate signals
    - O: the observed low-resolution audio signals
- Speech synthesis
- Speech recognition

#### Applications in Science Today

- Error correcting codes
  - e.g.
    - Iterative message passing algorithms
      - belief propagation
- Computational Biology
  - e.g.
    - model of how DNA sequences evolve over time
- Ecology
  - e.g.
    - bird migration model capturing spatial and temporal dependencies
- Economics
  - e.g.
    - estimated daily per capita expenditure capturing spatial dependencies

#### Applications in Health Care and Medicine

- Medical Diagnosis
  - e.g.
    - a Bayesian network for diagnosing pneumonia

## Representation
### 1. Bayesian networks

https://ermongroup.github.io/cs228-notes/representation/directed/

- directed ascylic graphs show causality effectively

#### Probabilistic modeling with Bayesian networks

- $n$: Number of variables
- $k$: Number of ancestor variables that the variable depends on
- $d$: Number of possible values for each variable which is discrete)
  - we assume they are all the same
- $x_{A_i}$: the set of $x_i$'s ancestor variables
- To reduce the number of parameters of a model to be $O(nd^{k+1})$ we want to replace
  - $p(x_i|x_{i-1}, x_{i-2}, ..., x_1)$
  - with $p(x_i|x_{A_i})$
- Otherwise we would need to consider $O(d^n)$ parameters

(Graphical representation)

- node or vertex: variable $x_i$
- edge: dependency relationship
- parent nodes of $x_i$: ancestors of $x_i$

(Formal definition)

a Bayesian network is a directed graph $G = (V, E)$ together with

- A random variable $x_i$ for each node $i \in V$.
- One conditional probability distribution (CPD) $p(x_i|x_{A_i})$ per node, specifying the probability of $x_i$ conditioned on its parents' values.

Note that

- A bayesian network defines a probability distribution p.
- A probability p factorizes over a DAG $G$ if it can be decomposed into a product of factors as specified by $G$
- There can be many ways for a probability p to factorize, and DAG $G$ would specify one of them.

#### The dependencies of a Bayes net

- $I(p)$: the set of all independencies that holds for a joint distribution $p$.
- $I$-map
  - If $p$ factorizes over $G$ then $I(G) \subseteq I(p)$
  - we say that G is an $I$-map for $p$

(Independencies described by directed graphs)

For three variables $X, Y, Z$:

- common parent
  - $X \leftarrow Z \rightarrow Y$
  - $\Longrightarrow X \perp Y | Z$
- cascade
  - $X \rightarrow Z \rightarrow Y$ or $Y \rightarrow Z \rightarrow X$
  - $\Longrightarrow X \perp Y | Z$
- V-structure
  - $X \rightarrow Z \leftarrow Y$
  - $\Longrightarrow X \perp Y$ when $Z$ is not observed

For more variables in Bayesian networks:

- $d$-separation
  - variables that are $d$-separated in $G$ are independent in $p$
  - but note that the converse it not true:
    - a distribution may fatorize over $G$, yet have have independencies that are not captured in $G$

(The representational power of directed graphs)

TODO


### 2. Markov random fields

https://ermongroup.github.io/cs228-notes/representation/undirected/

TODO

- preferable for problems where there is no clear causality between random variables.

## Inference

### 1. Variable elimination

## Learning
