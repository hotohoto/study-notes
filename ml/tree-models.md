# Tree models

(non-linear)

## Decision Tree

summary

- separates data

good question?

- better than now as much as possible
- increase purity
- decrease impurity

algorithms

- CART
  - https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/
  - Regression
    - use squared error loss function
  - Classification
    - use GINI index
    - https://en.wikipedia.org/wiki/Decision_tree_learning
- ID3
  - for classification
  - maximize Information Gain
  - lower the weighted sum of entropy
    - high entropy = low purity = high impurity
    - low entropy = high purity = low impurity
  - https://sefiks.com/2017/11/20/a-step-by-step-id3-decision-tree-example/
- ID3 with the standard deviation
  - for regression
  - maximize the reduction of the standard deviation
    - low standard deviation = values are less scattered
    - high standard deviation = values are more scattered
- C4.5
  - ID3 successor
  - https://www.quora.com/What-are-the-differences-between-ID3-C4-5-and-CART
- C5.0
  - proprietary
- Chi-square

etc.

- pruning
- we can stop before reaching the end
- stopping criteria
- number of minimum data size for the node
- maximum node

regression tree explained...

- http://www.stat.cmu.edu/~cshalizi/350-2006/lecture-10.pdf

pros

- The human can understand and utilize the structure of the tree

cons

- cannot configure the max depth meticulously when it needs to deal with underfitting and overfitting (there is no depth of either 3.2 or 2.5)

## Random Forest

- the ensemble of decision trees (average)
- how to make different trees from the same dataset
  - resampling (takes different instances from the dataset)
  - select different features (usually randomly)
- we can make parallel inference function
- pros
  - short training time comparatively thanks to the parallel boosting

## GBM

how it works

- make decision tree of a loss function
- and make another decision tree complementing the result of the previously made tree
- and make decision tree complementing the result of the previously made tree
- …

- kind of boosting
- too many steps make overfitting
- shrinkage value
  - the amount of the values that will be applied as supplementation by the following trees

- iteration here is number of trees

- pros
- cons
  - long training time comparatively thanks to the serial boosting
  - long inference time comparatively to the serial boosting

types

- Light GBM
  - made by MS
  - Light means fast
