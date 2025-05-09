# DS project guide

## tips

- Try different split ratios.
- Get test scores for each model.
  - Validation scores are not proper for the final comparison.
  - Put the best models in the report, and put the others in the appendix
- split
  - training set
    - k-folds cross validation
  - validation set
    - having multiple subset provides better understanding about the generalization.
  - test set
- Try to split the dataset with various ratios.
- Recommend two or three models for the best performance and the best stability
- Apply k-folds cross validation only on the training set.
  - We want to have a reliable evaluation score using a holdout set.
  - We don't want models to overfit to the validation set.
- Think who knows the problem best
  - Try to think from the view point of the big picture
  - Find reference and compare it to the model we want to train in the current project
- Model attributes
  - what to predict
    - (e.g.)
      - for a 6-month window
        - by modifying the window, we can make the target less skewed
  - what is the sample
    - per building vs per address vs per facility in a buidling
    - use only subset for securing more features?
      - (e.g.)
        - only for non-residential properties so that inspectable

