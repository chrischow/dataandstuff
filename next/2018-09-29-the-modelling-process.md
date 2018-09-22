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
  
Hence, to test out the various combinations
