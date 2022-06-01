# Graph and recommendation system

[그래프와 추천 시스템](https://www.boostcourse.org/ai211/joinLectures/316864)

## TODO

(papers)
- 2014 DeepWalk: Online Learning of Social Representations
- 2016 node2vec: Scalable Feature Learning for Networks

(Recommended materials by the lecturer for further study)

- Mining of massive datasets (2nd edition)
  - Jure Leskovec, et al.
- Networks crowds and markets
  - David Easley and Jon Kleinberg
- CS224W: Machine Learning with Graphs
  - standford / winter 2021
- CS246: Mining Massive Data Sets
  - standford / winter 2020

## 1. Basics

- https://www.boostcourse.org/ai211/lecture/1157069
- https://www.boostcourse.org/ai211/lecture/1157071

terms:

- directed graph vs undirected graph
- unweighted graph vs weighted graph
- unpartite graph vs bipartite graph
- notation
  - $G = (V, E)$
- neighbor
  - out-neighbor
  - in-neighbor
- path
  - a sequence from $u$ to $v$
  - any adjacent vertices in the sequence are connected by an edge
- distance of $u$ and $v$
  - the length of the shortest path between the two vertices
- diameter of $G$
  - maximum distance in the graph $G$

## 2. Patterns

- https://www.boostcourse.org/ai211/lecture/1163362
- https://www.boostcourse.org/ai211/lecture/1163363

terms:

- Erdős–Rényi random graph
  - $G(n,p)$
    - $n$
      - number of vertices
    - $p$
      - probability that any edge exists
    - the existence of edges are independent
- degree of $v$ or 연결성
  - $|N(v)|$ or $d_v$ or $d(v)$
  - out-degree
    - $|N_\text{out}(v)|$ or $d_\text{out}(v)$
  - in-degree
    - $|N_\text{in}(v)|$ or $d_\text{in}(v)$
- connected component
  - nodes connected by edges
    - there exists a path for any two nodes in a connected component
    - it should not be able to add any additional node into it
- community or 군집
  - there are many edges between nodes in a community
  - there are not many edges between nodes belonging to two different communities.
  - local clustering coefficient
    - $C_i$ of node $v_i$
      - (number of pairs of neighborhoods that have an edge) / ${d(v_i) \choose 2}$
        - note that the node $v_i$ is not taken into account when calcuating the numerator
      - $n$ number of nodes in the clu
  - global clustering coefficient
    - the average of the local clustering coefficient
    - exlucding nodes that the local clustering coefficient is not defined

patterns:

- Small world effect
  - It's likely to exist in the real world graphs
  - exceptions:
    - chain graph, cycle graph, grid graph
- Heavy tailed degree distribution
- giant connected components
  - e.g. in the MSN graph, there exists a connected component including 99.9% of nodes
  - random graph is less likely to have a giant connected component
    - but it depends on the parameter $p$
- high clustering coefficient
  - for real world graphs
  - reasons
    - homophily
      - the group of people in similar ages are likely to be friends
    - transitivity
      - my neighborhood is likely to introduce their neighborhoods
- low clustering coefficient
  - for random graphs

## 3. Page ranks

- https://www.boostcourse.org/ai211/lecture/1157086
- https://www.boostcourse.org/ai211/lecture/1157087
- https://en.wikipedia.org/wiki/PageRank

description:

- A page with many incoming edges are more important and reliable compared to other pages
- incoming edges from the highly ranked nodes have high weights
- power iteration
- algorithm
  - add edges from each dead-end node to all nodes including itself
  - initialize $r_i^{(0)} = 1 / |V|$
  - update ranks using the formula below until it converges
    - $r_j^{(t+1)} = \sum\limits_{i \in N_\text{i}(j)} \left( \alpha {r_i^{(t)} \over d_\text{out}(i)} \right) + (1 - \alpha) {1 \over |V|}$
- teleport
  - to address pitfalls
    - spider trap
    - dead end
  - with damping factor $\alpha$
    - usually it's $0.8$
  - how it looks like
    - if there's no outgoing edges
      - if $u \le \alpha$ where $u \sim \mathcal{U}_{[0, 1]}$
        - pick an outgoing edge and go to the counter node
      - else
        - teleport to a random node
    - else
      - teleport to a random node

## 4. Influence model or 전파 모델

- Decision making based
  - e.g.
    - kakao talk vs naver line
    - early adaptors will chose a product independently
    - the other will choose a product depending on their neighborhoods' decision
  - algorithm
    - linear threshold models
- Probabilistic propagation models
  - e.g.
    - COVID-19 infection
  - algorithm
    - independent cascade models
- Influence Maximization
  - choosing best influencers
  - e.g.
    - shoose seed nodes for viral marketing
  - NP-hard
  - Kate Middleton effect
    - show is a famous influencer
  - algorithms
    - heuristics
      - node centrality
        - page rank score
        - degree centrality or 연결 중심성
        - distance centrality or 근접 중심성
        - betweenness centrality or 매개 중심성
      - greedy algorithm
        - The performance lower bound is guaranteed.

## 5 Community Detection or 군집 탐색

- Configuration model
  - keep the degree for each node
  - randomize the edges
- modularity or 군집성
  - in [-1, 1]
  - 0.3 ~ 0.7 would be of a good community found statistically

(군집 탐색 알고리즘)

- Girvan-Newman Algorithm
  - top-down
  - iteratively removes bridges which have the highest betweeness centrality and connect the other communities
    - how?
      - betweeness centrality: number of cases where the edge belongs to the shortest path of any two nodes
    - modularity
  - assume each node belong to a single community
- Louvain Algorithm
  - bottom up
  - assume each node is a community
  - in each pass we merge communities according to modularity until the modularity is not increased
  - assume each node belong to a single community
- Overlapping communities cases
  - some can be in both these communities at the same time - a group of highschool friends, a squash club, a basketball club
  - MLE
  - assume discrete assignment
    - not able to use gradient descent methods
  - assume continuous assignment
    - 완화된 중첩 군집 모형
    - would be more reallistic since we might feel more dedicated to some communities than others

## 6. Recommendation system (basics)

https://www.boostcourse.org/ai211/lecture/1157094

(content based recommendation)

- item profile
  - one-hot encoding
- user profile
  - weighted average of item profiles
    - weights: likes, # of stars
- matching
  - cosine similarity between a user profile and each item profile
- pros
  - no need to use the data from other users
  - unique users can be applied
  - new item can be applied
  - can provide the reasons why those items are recommended
- cons
  - item attributes are mandatory
  - requires user data of purchasing items
  - overfitting - so the recommendation ranges can be narrow.

(collaborative filtering)

- procedure
  - step1: find similar users in terms of preference
    - calcluate correlation coefficient over the purchased items in common
  - step2: find items that those users preferred
  - step3: recommend those items
- pros
  - available even when there are no attributes for items
- cons
  - requires quite many data
  - new users and new items can not be applied
  - bad at unique users

How to calculate the correlation coefficient?

$$
\operatorname{sim}(x,y) =
{
  {\sum_{s \in S_{xy}}(r_{xs} - \overline{r_x})(r_{ys} - \overline{r_y})}\over
  {
    \sqrt{\sum_{s \in S_{xy}}(r_{xs} - \overline{r_x})^2}
    \sqrt{\sum_{s \in S_{xy}}(r_{ys} - \overline{r_y})^2}
  }
} =
{
  \operatorname{cov}(r_x, r_y)
  \over
  \sigma_{r_x}\sigma_{r_y}
}
$$

- $r_{xs}$
  - the rating score of item $s$ by user $x$
- $r_x$
  - the average rating given by user $x$
- $S_{xy}$
  - all the items purchased/watched/rated by both user $x$ and $y$ in common

How to estimate the rating of item $s$ by user $x$?

$$
\hat{r}_{xs} =
{
  {\sum_{y \in N(x;s)} \operatorname{sim}(x,y) r_{ys}}
  \over
  {\sum_{y \in N(x;s)} \operatorname{sim}(x,y)}
}
$$

- $N(x;s)$
  - $k$ users who have most simlilar preference to user $x$

(evaluation of recommendation system)

- MSE
- RMSE
- Correlation coefficient of between "orders" of ground truth ratings and the estimated ratings
- Actual purchase ratio for the recommended items
  - the recommendation order and the diversity may be taken into account

## 7. Representaion of nodes

https://www.boostcourse.org/ai211/lecture/1157099

- node representation learning
  - learn to represent nodes as vectors in the embedding space
    - $f: V \to \mathbb{R}^d$
  - pros
    - we can use a bunch of vector based algorithms e.g.
      - K-means, DBSCAN
  - similarity
    - in embedding space
      - inner product
    - in graph
- transductive vs inductive methods
  - transductive methods
    - $\text{ENC}(v)$
      - generates $z_v$ using the graph
    - cons
      - if the graph has been changed a bit the entire encodings should be calculated again
      - to use embeddings, all of them should be calculated in advance
      - can not make use of the attributes of nodes
  - inductive methods
    - generates $z_v$ using the graph and the attributes of nodes
    - e.g. graph neural networks
    - pros
      - can provide node embeddings even if the graph has been changed a bit
      - no need to calculate the embeddings in advance and save them
      - can make use of the attributes of nodes
- transductive methods
  - adjacency based approach
    - $\mathcal{L} = \sum\limits_{(u,v) \in V \times V} ||\mathbf{z}_u^T\mathbf{z}_v - \mathbf{A} _{u,v}||^2$
    - SGD can be used
    - cons
      - it doesn't take account of how long the two nodes are far away to each other.
  - distance based approach
  - path base approach
    - $\mathcal{L} = \sum\limits_{(u,v) \in V \times V} ||\mathbf{z}_u^T\mathbf{z}_v - \mathbf{A} _{u,v}^k||^2$
  - overlap based approach
    - $\mathcal{L} = \sum\limits_{(u,v) \in V \times V} ||\mathbf{z}_u^T\mathbf{z}_v - \mathbf{S} _{u,v}||^2$
      - using common neighborhoods
        - $S_{u, v} = |N(u) \cap N(v)| = \sum\limits_{w \in N(u) \cap N(v)} 1$
      - using Jacard similiarity
        - $S_{u, v} = {|N(u) \cap N(v)| \over |N(u) \cup N(v)|}$
      - using Adamic Adar score
        - $S_{u, v} = \sum\limits_{w \in N(u) \cap N(v)} {1 \over d_w}$
        - $d_w$: the number of neighborhodds of the common neighborhood $w$
  - random walk based approach
    - focus on local information but doesn't restrict to use global information
    - $\mathcal{L} = \sum\limits_{u \in V}\sum\limits_{v \in N_R(u)} - \log(P(v|\mathbf{z}_u))$
      - where
        - $N_R(u)$ is the list of nodes approached by random walk from node $u$
        - $P(v|\mathbf{z}_u) = {\exp(\mathbf{z}_u^T\mathbf{z}_v) / \sum\limits_{n \in V} \exp(\mathbf{z}_u^T\mathbf{z}_n)}$
      - Note that it takes $O(n^2)$ so we need approximation
        - $\log({\exp(\mathbf{z}_u^T\mathbf{z}_v) / \sum\limits_{n \in V} \exp(\mathbf{z}_u^T\mathbf{z}_n)}) \approx \log(\sigma(\mathbf{z}_u^T\mathbf{z}_v)) - \sum\limits_{i=1}^k \log(\sigma(\mathbf{z}_u^T\mathbf{z}_{n_i}))$
          - $n_i \sim P_v$
          - $k$: number of negative samples
    - types
      - DeepWalk
        - uses uniform probabilities when selecting a node to walk to
      - Node2Vec
        - uses the second-order biased random walk
          - nodes going far from the previous node
          - nodes staying in the same distance from the previous node
          - going back to the previous nodes

## 8. recommendation system (intensive)

https://www.boostcourse.org/ai211/lecture/1163376

Netflix Chanllenge
- training data
  - 480K users
  - 18K movies
  - 100M reviews
- test data
  - recent 2.8M reviews
- evaluation
  - RMSE
    - mean ratings: 1.1296
    - mean of user ratings: 1.0651
    - mean of item ratings: 1.0533
    - legacy Netflix recommendation system: 0.9514
    - collaborative filtering: 0.94
    - latent factor model: 0.90
    - latent factor model with user/item bias: 0.89
    - latent factor model with temporal user/item bias: 0.876
    - Ensemble: 0.8567

https://www.boostcourse.org/ai211/lecture/1163377

(Latent Factor Model)
- UV decomposition similar to SVD
- uses node representation
- learn latent factor
  - rather than using explicit factors e.g. romance, action, ...
- model:
  - predicts $r_{xi}$ by $p_x^T q_i$
- training:
  - learns $p_x$, and $q_i$
  - $\mathcal{L} = \sum\limits_{(x, i) \in R} (r_{xi} - p_x^T q_i)^2 + \lambda_1 \sum\limits_x ||p_x||^2 + \lambda_2 \sum\limits_i ||q_i||^2$
    - $R$: training set
    - $r_{xi}$: rating
    - $x$: user index
    - $i$: item index
    - $\lambda_1, \lambda_2$: L2 regularization coefficients
  - optimizer: (Stochastic) Gradient Descent

(Latent Factor Model - with user/item bias taken into account)
- model:
  - predicts $r_{xi}$ by $\mu + b_x + b_i + p_x^T q_i$
    - $\mu$: the mean of entire ratings
    - $b_x$: the mean of the ratings of the user
    - $b_i$: the mean of the ratings of the item
- training:
  - learns $b_x$, $b_i$, $p_x$, and $q_i$
  - $\mathcal{L} = \sum\limits_{(x, i) \in R} (r_{xi} - (\mu + b_x + b_i + p_x^T q_i))^2 + \lambda_1 \sum\limits_x ||p_x||^2 + \lambda_2 \sum\limits_i ||q_i||^2 + \lambda_3 \sum\limits_x ||b_x||^2 + \lambda_4 \sum\limits_i ||b_i||^2$
  - optimizer: (Stochastic) Gradient Descent

(Latent Factor Model - with temporal user/item bias taken into account)

- model:
  - predicts $r_{xi}$ by $\mu + b_x(t) + b_i(t) + p_x^T q_i$
    - $\mu$: the mean of entire ratings
    - $b_x(t)$: the mean of the ratings of the user with respect to the days since release
    - $b_i(t)$: the mean of the ratings of the item with respect to the days since release

## GNN

https://www.boostcourse.org/ai211/lecture/1157116

- inductive method
- applicable to both supervised/unsupervised learning

### Vanilla GNN

$$ h_v^0 = x_v$$
$$ h_v^k = \sigma\left((W_k \sum\limits_{n \in N(v)} {h_u^{k-1} \over |N(v)|} + B_k h_v^{k-1}\right)$$
$$z_v = h_v^K$$

- where
  - $k \gt 0$
    - the distance index where is k from the node we're interested in
    - when $k=0$, it's farthest from the node we're intrested in
    - when $k=K$, it's the node itself
    - so we aggregate inputs from the layer $k=0$ until we reach to the node we're intrested in ath the layer $K$
  - $\sigma$
    - a sigmoid function e.g. ReLU, tanh, ...
  - $x_v$
    - attributes of the node $v$
    - for the nodes in the layer k=0, we use their attributes as their hidden states
- inputs:
  - graph of nodes up to those of the distance $K$ from the original node we're interested in
  - attributes of those nodes
- trainable parameters:
  - $W_k$, $B_k$
  - The aggregation function for each layer is shared within the layer
- for each layer, inputs from neighborhood nodes are averaged so that no matter the number of neighborhoods are they become the input of fixed shape
- For a classification task end-to-end models outperformed models trained with separate training steps

### Graph Convolutional Network (GCN)

$$
h_v^k = \sigma\left(W_k \sum\limits_{\{n \in N(v)\} \cup v} {h_u^{k-1} \over \sqrt{|N(u)||N(v)|}}\right)
$$

- contrary to images in computer vision tasks
  - the number of neighborhood nodes are not fixed
  - the neighborhood nodes would be much less likely to have correlation
  - the order of neighborhood nodes has no meaning
- So we should not use CNN for graph but GNN or GCN

### GraphSAGE

$$
h_v^k = \sigma\left(\operatorname{concat}[W_k \cdot \operatorname{AGG}(\{h_u^{k-1}|\forall u \in N(v)\}), B_k h_v^{k-1}]\right)
$$

- AGG
  - mean
    - $\sum\limits_{u \in N(v)} {h_u^{k-1}\over |N(v)|}$
  - element-wise max-pooling
    - $\gamma(\{Q h_u^{k-1}| \forall u \in N(v)\})$
  - LSTM
    - $\operatorname{LSTM}([h_u^{k-1}| \forall u \in N(v)])$

### Graph Attention Network (GAT)

https://www.boostcourse.org/ai211/lecture/1163777

- use multi-head attention instead of averaging neighborhoods' hidden states
- self attention
  - $\alpha_{ij}$
    - weights from the node $i$ to the node $j$
    - step 1
      - $\tilde{h}_i = h_i W$
    - step 2
      - $e_{ij} =  a^T[\operatorname{concat}(\tilde{h}_i, \tilde{h}_j)]$
    - step 3
      - $\alpha_{ij} = \operatorname{softmax}_ j(e_{ij}) = {\exp(e_{ij}) \over \sum_{k \in N_i} \exp(e _{ik})}$
- multi-head attention
  - $h_i^\prime = \operatorname{concat}\limits_{1 \le k \le K} \sigma \left(\sum\limits_{j \in N_i} \alpha_{ij}^k h_j W_k\right)$

### Graph representation learning

https://www.boostcourse.org/ai211/lecture/1163777

- graph to a vector
- Graph pooling
  - node embeddings to a graph embedding
- Differentiable pooling (DiffPool)
  - make use of community structure detected

### Over-smoothing problem

https://www.boostcourse.org/ai211/lecture/1163777

- related to the small world effect
- as we have more and more layers the model performance is decreased
- residual connection doesn't help that much

(JK Network)

- introduce layer aggregation
  - lastly aggregate all the hidden states from all the layers
  - with
    - concatenation
    - max pooling
    - LSTM-attention

(APPNP)

- Use the neural network structure only in the 0th layer
  - the other layers doesn't have $W$ as its parameter

### Data augmentation

https://www.boostcourse.org/ai211/lecture/1163777

- add edge where the two adjacent nodes have similar node embeddings

## papers and articles

- [2020 Feature Extraction for Graphs](https://towardsdatascience.com/feature-extraction-for-graphs-625f4c5fb8cd)
  - node level
    - node degree
    - eigenvector centrality
  - graph level
    - adjacency matrix
    - Laplacian matrix
      - degree matrix - adjacency matrix
    - bag of nodes
    - Weisfeiler-Lehman (WL) Kernel
    - Graphlet kernels
    - path-based kernels
    - etc.
      - GraphHopper kernel
      - neural message passing
      - graph convolution networks
  - neighborhood overlap features (between two nodes)
    - Sorensen index
    - Salton index
    - Hub Promoted index
    - Jaccard index
    - Resource Allocation (RA) index
    - Katz Index
    - Leicht-Holme-Newman (LHN) similarity
      - normalized by the expected value of the Adjacency matrix
