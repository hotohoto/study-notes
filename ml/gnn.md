# GNN (Graph Neural Networks)

## Terminology

- graph or network
- vertex or node
  - V: set of vertices
- edge or link
  - E: set of edges
- complex system
  - brain
  - knowledge graph
  - chemical moleclre
- problems
  - node classification
    - protain interaction
  - link prediction
  - recommendation
  - community detection
    - similar to clustering
  - ranking
    - or information retrieval
  - information cascading or viral marketing
- types
  - undirected graph
  - directed graph
  - unweighted graph
  - weighted graph
  - unipartite graph
  - bipartite graph
    - there are two types of vertices
    - two vertices can be connected when they are different types of vertices
- neighbor
  - vertices connected to a vertex
  - N(v): set of neighbor vertices of vertex v
  - N_in(v): set of incoming neighbor vertices of vertex v
  - N_out(v): set of outgoing neighbor vertices of vertex v
- degree
  - d(v) or |N(v)| or number of neighbors of vertex v
  - d_in(v) = |N_in(v)|
  - d_out(v) = |N_out(v)|
- real graph
  - degree is distributed over heavy tail distribution
  - there is a giant connected component
  - high clustering coefficent
- random graph
  - refers to probability distributions over graphs
  - example
    - Erdős–Rényi random graph
      - G(n,p)
        - n: number of vertices
        - p: probability that two vertices are connected
          - all connections are independent
      - degree is likely to be distributed over normal distribution
      - low clustering coefficent
      - it's likely that there exists a giant connected component when the expected degree is greater than 1
- path
  - sequence of vertices
  - starts from u ends at v moving through edges
- legnth
  - number of edges over a path
  - (number of vertices) - 1
- distance
  - length of shortest path between u and v
- diameter
  - the maximum distance of any two vertices in a graph
- small world effect
  - most nodes can be reached from every other node by a small number of hops or steps
  - examples in which there is no small world efect
    - chain graph
    - cycle graph
    - grid graph
- connected components
- community
  - https://en.wikipedia.org/wiki/Community_structure
  - local clustering coefficient
    - C_i
    - https://en.wikipedia.org/wiki/Clustering_coefficient
  - global clustering coefficient
- homophily
  - 동질성
  - simmilar properties
- transitivity
  - 전이성
  - introduce a friend to another friend

## Page rank

Page rank score
- weighted voting
  - $r_{j}=\sum_{i \in N_{in}(j)} \frac{r_{i}}{d_{out}(i)}$
- the same as random walk converges into stationary distribution
  - $p_{j}=\sum_{i \in N_{in}(j)} \frac{p_{i}}{d_{out}(i)}$

Page rank algorithms
- power iteration
  - initialize r_i with 1 / # of pages
  - iterate until it converges
  - cons
    - may not be converged
    - may not ends up with a good distribution for dead ends
- power iteration with teleport
  - transition to a random page with a teleport damping factor
    - $r_{j}=\sum_{i \in N_{\text {in}}(j)}\left(\alpha \frac{r_{i}}{d_{\text {out}}(i)}\right)+(1-\alpha) \frac{1}{|V|}$
    - $|V| = \text{(number of pages)}$:
    - usually α = 0.8
  - no dead ends issue

## Community Detection

- similar to clustering
- configuration model
  - https://en.wikipedia.org/wiki/Configuration_model
- modularity
  - ${1 \over |E|}\sum\limits_{s \in S}\left(\text{number of edges in s}\right) - \left(\text{number of edges in s under configuration model}\right)$
  - if inner edges are many compared to the configuration models, it's a well detected community
  - in [-1, 1]
  - statistically good when it's in [0.3, 0.7]
- Girvan-Newman algorithm
  - TODO http://www.edwith.org/ai211/lecture/1163371

## Recommender system

- Collaborative filtering

## GNN



## References

- https://www.edwith.org/ai211
-
