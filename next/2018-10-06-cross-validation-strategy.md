# Cross Validation Strategy
In my posts thus far, I mentioned "cross validation" several times as a technique to evaluate encoding schemes, tune parameters and estimate prediction accuracy. In this post, I explain the cross validation strategy I used for my model. This strategy is highly-applicable and useful to other pipelines with stacked models because it ensures minimal information leakage in the testing process.  

# Implementing the Pipeline
One reason I fell in love with Python was the simplicity with which I could develop machine learning pipelines. `sklearn` has a `Pipeline` class that allows you to chain data processing steps and machine learning models together. These `Pipeline` objects ensure that no leakage occurs by fitting data processing and machine learning models with only training data, and applying the required transformations or predictions on test data. The beauty of `Pipeline` is that it allows users to do these things with **just two steps**: the `fit` and `predict` methods.  
  
Hence, to optimise encoding schemes and hyperparameters, I put them into a custom pipeline and ran **nested cross validation (CV)**.
  
## Nested Cross Validation (CV)
What we want to know about the pipeline is: how well does this perform on unseen data? To achieve this, we partition the data into "known" and "unseen" data by means of cross validation. Suppose we slice the data into 5 equal slices or folds: Folds 1 to 5. We effectively have 5 sets of "known" and "unseen" data:  
  
| Set |  Training Data  | Test Data |
|:---:|:---------------:|:---------:|
|  1  | Fold 2, 3, 4, 5 |  Fold 1   |
|  2  | Fold 1, 3, 4, 5 |  Fold 2   |
|  3  | Fold 1, 2, 4, 5 |  Fold 3   |
|  4  | Fold 1, 2, 3, 5 |  Fold 4   |
|  5  | Fold 1, 2, 3, 4 |  Fold 5   |
  
We have yet to address the "nested" part of nested CV. Let's look at Set 1. Treating the training data (Fold 2, 3, 4, 5) as a single training set, we divide this set into yet another 5 folds.  
  
| Inner Set |  Set 1 Training Data  | Test Data |
|:---:|:---------------:|:---------:|
|  A  | Fold B, C, D, E |  Fold A   |
|  B  | Fold A, C, D, E |  Fold B   |
|  C  | Fold A, B, D, E |  Fold C   |
|  D  | Fold A, B, C, E |  Fold D   |
|  E  | Fold A, B, C, D |  Fold E   |
  
We call sets 1 to 5 the *outer CV loop*, and sets A to E (for each of sets 1 to 5) the *inner CV loops*. We perform all optimisation (of encoding schemes and hyperparameters) on the inner CV loops before performing predictions in the outer CV loop. For example, suppose we are at Set 1. We obtain sets 1A to 1E, and use these to test different parameter configurations: 5 pipeline runs per configuration. That gives us a mini distribution of accuracy scores for each configuration, and allows us to choose the best one with some reliability. With the optimal configurations, we take a step back into the outer CV loop. We fit the pipeline to the full training data (Folds 2 to 5), and make predictions on the test data (Fold 1). Hence, we would be fitting many, many pipelines and models - as many as `No. of Models * No. of Parameter Configurations * 5 Outer Loop CV Folds * 5 Inner Loop CV Folds`.
  
Nested CV therefore allows us to make full use of the data we have to generate a reliable estimate of model prediction accuracy. We need a *distribution* of accuracy scores because there is a good deal of inherent randomness within the process. For example, there is randomness in:  
  
1. The way the features are encoded: e.g. decision tree binning
2. Machine learning algorithms: e.g. At each split, Random Forest and Gradient Boosting regressions draw features at random to split the data
3. The splits in cross validation: each 5-fold (or *k*-fold) partition is split at *random*. Repeating the splitting process with a different random seed would result in different 5-fold partitions.
