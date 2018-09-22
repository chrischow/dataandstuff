# Introduction
Thus far, I've written about exploratory data analysis (EDA) and feature engineering in the HDB resale flat dataset. What I should have done earlier was to outline the overall approach for modelling. In this post, I introduce the broad approach to the resale flat price model and provide some preliminary results on the progress made.  
  
# The Pipeline
The objective of this project is to develop something that can accurately predict (or value) resale flats. This thing takes in data provided by HDB through [Data.gov.sg](https://data.gov.sg/), generates new features, cleans them, runs them through a complex machine learning model, and spits out predicted prices to **enable home buyers to assess the value of prospective homes**. This thing is a **pipeline**.
  
We can think of it as a macro model comprised of the following key processes:  
  
1. **Feature engineering.** In this step, we create new features from the limited set of features provided to us. I wrote about engineering [numeric features](https://chrischow.github.io/dataandstuff/2018-09-08-hdb-feature-engineering-i/), [categorical features](http://www.google.com), and [geocoding](https://chrischow.github.io/dataandstuff/2018-09-16-hdb-feature-engineering-ii/) earlier, and commented on how *the way* features are encoded - binning method for numeric features and encoding methods for categorical features - are parameters in the pipeline. In the pipeline, many combinations of encoding techniques are tested to find the optimal encoding method for each machine learning model.
2. **Hyperparameter tuning.** In this step, we optimise several regression models separately. Each model (less OLS regression) has its own set of hyperparameters that governs how the model learns old data and how well it performs on unseen data. The objective of hyperparameter tuning is to reduce *overfitting*, which I will explain the coming sections.
3. **Prediction.** In the testing phase, we use this step to generate predictions on pseudo-unseen data to evaluate the pipeline. Once the pipeline goes live, this phase is for generating actual predictions/valuations.  
  
## The Machine Learning Model
Before we dive into the pipeline, it is important to establish what exactly we're optimising. The machine learning model is the final entity in the pipeline that converts data points into predictions. Although I say "model", I really mean a model of models. That is, I am developing a **stacked regression model**: a model that has several base regression models (first layer), and a single regression model (second layer) that aggregates the predictions of the first-layer models. I selected several regression models for the first layer:  
  
1. OLS Regression
2. Ridge
3. Lasso
4. ElasticNet
5. Support Vector Regression
6. Random Forest Regression (`sklearn`)
7. Gradient Boosting Regression (`xgboost`)
  
These models will also be candidates for the single second layer model. Each of these models (less OLS regression) has configurable hyperparameters which will be optimised in the hyperparameter tuning step. As such, each first-layer model (and the second-layer model) must be optimised separately.  
  
## Feature Engineering
I've demonstrated several feature engineering techniques in my previous posts. However, in the pipeline, I don't perform feature engineering manually. I wrote several functions to automate the application of encoding schemes of my choice to separate features. This enabled me to test out all encoding schemes on several features for all models. Due to the nature of each machine learning algorithm, different feature encodings were used for different models. For example, OLS regression and Ridge regression tended to perform better with one-hot-encoded categorical features, while the tree-based methods (RF and Gradient Boosting regression) tended to work better with stats-encoded and binary-encoded categorical features.
  
## Hyperparameter Tuning
With the encoding schemes sorted out, the next step was to optimise machine learning models separately. The objective of hyperparameter tuning is to reduce overfitting, which is where a model learns the training data too well and performs badly on unseen data. It is equivalent to memorising solution steps for several specific math problems without developing an understanding of the logic involved. When given questions with even minor variations, we would be unable to solve them because we only know how to regurgitate the solutions to the few problems that we encountered.
  
## Prediction
This step is simple. Fit the respective regression models to training data, and perform predictions on test data. Because have the "correct answers" for the test data, we can evaluate how good the predictions (and models) are.
  
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
  
We call sets 1 to 5 the *outer CV loop*, and sets A to E (for each of sets 1 to 5) the *inner CV loops*. We perform all optimisation (of encoding schemes and hyperparameters) on the inner CV loops before performing predictions in the outer CV loop. For example, suppose we are at Set 1. We obtain sets 1A to 1E, and use these to test different parameter configurations: 5 pipeline runs per configuration. That gives us a mini distribution of accuracy scores for each configuration, and allows us to choose the best one with some reliability. With the optimal configurations, we take a step back into the outer CV loop. We fit the pipeline to the full training data (Folds 2 to 5), and make predictions on the test data (Fold 1). Hence, we would be fitting many, many pipelines and models - as many as $No.\ of\ Models\ \times\ No.\ of\ Parameter\ Configurations\ \times\ 5\ Outer\ Loop\ CV\ Folds\ \times\ 5\ Inner\ Loop\ CV\ Folds$.
  
Nested CV therefore allows us to make full use of the data we have to generate a reliable estimate of model prediction accuracy. We need a *distribution* of accuracy scores because there is a good deal of inherent randomness within the process. For example, there is randomness in:  
  
1. The way the features are encoded: e.g. decision tree binning
2. Machine learning algorithms: e.g. At each split, Random Forest and Gradient Boosting regressions draw features at random to split the data
3. The splits in cross validation: each 5-fold (or *k*-fold) partition is split at *random*. Repeating the splitting process with a different random seed would result in different 5-fold partitions.
  

