# K-means algorithm

## short comings

- k needs to be set
  - can be addressed by non-parameteric Bayesian method
- depending on the initial parameters it can fall into a local minimum
- limited to Euclidean distance
  - GMM supports ellipsis shaped distance

## EM Algorithm

- Expectation
  - assign latent variable Z with the log-likelihood expectation using current parameters
- Maximization
  - find the best parameters maximizing the expected log-likelihood and update them

## Other sub-topics

- pick k with non-parametric Bayesian methods
  - http://stat.columbia.edu/~porbanz/npb-tutorial.html
  - https://blog.statsbot.co/bayesian-nonparametrics-9f2ce7074b97
  - https://www.stats.ox.ac.uk/~teh/research/npbayes/OrbTeh2010a.pdf
  - 2016 A Bayesian non-parametric method for clustering high-dimensional binary data
    - https://arxiv.org/abs/1603.02494
  - 2011 Revisiting k-means: New Algorithms via Bayesian Nonparametrics
    - https://arxiv.org/abs/1111.0352
- how to initialize first centroids
  - pick random k samples as centroids
  - other methods
    - 2009, An Improved K-means Clustering Algorithm with refined initial centroids
      - http://www.dline.info/jnt/fulltext/v1n3/2.pdf
    - 2017, K-Means Clustering With Initial Centroids Based On Difference Operator
      - http://www.ijircce.com/upload/2017/january/23_K_MEANS.pdf
    - 2010, Enhancing K-means Clustering Algorithm with Improved Initial Center
      - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.7070&rep=rep1&type=pdf
