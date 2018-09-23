# Introduction
Thus far, I've written about exploratory data analysis (EDA) and feature engineering in the HDB resale flat dataset. What I should have done earlier was to outline the overall approach for modelling. In this post, I introduce the broad approach to the resale flat price model and provide some preliminary results on the progress made.  
  
# The Pipeline
The objective of this project is to develop something that can accurately predict (or value) resale flats. This thing takes in data provided by HDB through [Data.gov.sg](https://data.gov.sg/), generates new features, cleans them, runs them through a complex machine learning model, and spits out predicted prices to **enable home buyers to assess the value of prospective homes**. This thing is a **pipeline**.
  
We can think of it as a macro model comprised of three key processes: (1) feature engineering, (2) hyperparameter tuning, and (3) prediction.  
  
## Feature Engineering
In my previous posts, I wrote about engineering [numeric features](https://chrischow.github.io/dataandstuff/2018-09-08-hdb-feature-engineering-i/), [categorical features](http://www.google.com), and [geocoding](https://chrischow.github.io/dataandstuff/2018-09-16-hdb-feature-engineering-ii/). Feature engineering actually isn't as simple as creating new columns in a table. We have to take into account the **encoding scheme** used and **leakage**.  
  
### Encoding Schemes
Essentially, the encoding schemes - *the way features are coded* - affect the final accuracy of the machine learning models we feed the data into. For example, the way we choose to bin numeric features and the type of encoding method we use on categorical features will impact different machine learning models in unique ways. Thus, for each machine learning model, we must test out various combinations of encoding schemes (for the different features).  
  
The table below shows the various ways we can encode each feature. This means that we need to run a total of **24 models** to test out all encoding schemes per model because we can optimise the schemes one feature at a time.  
  
|     Feature     | Numeric? |   Ordinal?  | One-hot? | Binary? | Stats? | Total Ways |
|:---------------:|:--------:|:-----------:|:--------:|:-------:|:------:|:----------:|
| Remaining Lease |    Yes   |     Yes     |    Yes   |   Yes   |   Yes  |      5     |
|    Floor Area   |    Yes   |     Yes     |    Yes   |   Yes   |   Yes  |      5     |
|       Town      |          |             |    Yes   |   Yes   |   Yes  |      3     |
|    Flat Type    |          |             |    Yes   |         |   Yes  |      2     |
|   Storey Range  |          |     Yes     |    Yes   |   Yes   |   Yes  |      4     |
|    Flat Model   |          |             |    Yes   |   Yes   |   Yes  |      3     |
|       Year      |          | Yes (Fixed) |          |         |        |            |
|      Month      |          | Yes (Fixed) |          |         |        |            |
|     Cluster     |          |             |    Yes   |   Yes   |        |      2     |
  
Since we're fitting 7 machine learning models (more on this later), we need a total of **168 models** to find the optimal encoding scheme for each model. Choosing the best schemes for each model was extremely computationally expensive. Hence, I chose to do it once, thereby making the assumption that encoding schemes are unaffected by randomly-drawn samples in cross validation (more on this in the next post).  
  
### Leakage
Previously, I wrote about how [contaminating training data](https://chrischow.github.io/dataandstuff/2018-09-01-preventing-contamination/) would result in overly-optimistic estimates of predictive accuracy. This means that, even though we know the optimal encoding scheme of a specific feature (say stats encoding for flat types), we cannot simply apply stats encoding to the entire dataset. This is because processing training data with test data (in cross validation) will inevitably *leak* information about the target feature (resale price) into the training set. It is like getting a few solution steps of each question in an 'A' level math exam before you actually take the exam: you don't know the full answer or the full question, but you have an advantage because of the bit of information you're getting. Hence, we must do the following in order: (assuming we have three features A, B, and C)  
  
1. Fit the training data to the feature encoding model for feature A
2. Transform the **training data** using the fitted encoding model
3. Transform the **test data** using the fitted encoding model
4. Do the same for the respective feature encoding models for features B and C sequentially
  
This requires us to have a pipeline that processes training data and test data in parallel, not together.
  
## Hyperparameter Tuning
In this step, we optimise the hyperparameters for several (regression) models separately. Each model (less OLS regression) has its own set of hyperparameters that governs how the model learns old data and how well it performs on unseen data. The objective of hyperparameter tuning is to reduce overfitting, which is where a model learns the training data too well and performs badly on unseen data.  
  
### Overfitting
Suppose we memorised solution steps for a book of specific math problems without developing any understanding of the logic involved. If we received a new book of math problems that were simply minor variations of the book we memorised, we might fail miserably because we only know how to regurgitate the solutions to the few problems that we studied. Thus, we can think of overfitting as *causing the model to learn how to solve specific problems only*, and hyperparameter optimisation as *helping the model to develop sound principles for solving the general problem*. This sounds a lot like the enhancement of Singapore's education curriculum, and we should expect as much because the concept of learning is the same for both human learners and machine learners.  
  
### Models?
Until now, I had not been specific about what machine learning model I was developing. To be specific, I am developing a **stacked regression model**: a model that has several base regression models (first layer), and a single regression meta model (second layer) that aggregates the predictions of the first-layer models.  
  
#### The First Layer
Models in the first layer take in processed data and attempt to predict resale prices. The idea for the first layer is to have as many diverse regression models as possible. Hopefully, they each excel in accurately predicting different *groups* of samples, thereby delivering an accurate model when put together. I chose the following regression algorithms:  
  
|           Algorithm          |    Type    |             Remarks             | Mean MAPE |
|:----------------------------:|:----------:|:-------------------------------:|:---------:|
| OLS                          |   Linear   | Non-parameteric; quick and easy |   6.57%   |
| Ridge                        |   Linear   | L2 regularisation               |  Dropped  |
| Lasso                        |   Linear   | L1 regularisation               |  Dropped  |
| ElasticNet                   |   Linear   | L1 and L2 regularisation        |  Dropped  |
| Support Vector               |   Linear   | Kernels                         |   9.18%   |
| Random Forest                | Tree-based | Bagging                         |   4.90%   |
| Gradient Boosting (XGBoost)  | Tree-based | Boosting                        |   4.58%   |
| Gradient Boosting (LightGBM) | Tree-based | Boosting                        |   5.06%   |
  
Thus far, I dropped Ridge regression because the results were extremely similar to linear regression, and Lasso and ElasticNet because they were terrible. Using nested cross validation, I computed estimates of their out-of-sample prediction (last column).   
  
#### The Second Layer
The second layer is a lot simpler: it requires just one (simple) model to aggregate the predictions of the first-layer models. It is also possible to feed the original data into the second layer model along with the first-layer predictions, but testing is required to assess if this helps to improve the overall predictive accuracy. I'm currently working on the second layer model and I have no results to present yet.  
  
## Prediction
This step is simple. Fit the respective regression models to training data, and perform predictions on test data. Because we have the "correct answers" for test data, we can evaluate how good the predictions (and models) are. Then, fit the meta model to the predictions to generate the final predictions. I'll elaborate more on this in the next post on cross validation.  
  
## Putting the Pipeline Together
We start with semi-processed data. The pipeline converts this data into predictions using the following sequence of processes:  
  
1. Generate features according to *specified encoding schemes*
2. Preprocess data using some *specified preprocessing methods*
3. Tune *specified hyperparameters* for a *specified model*
4. Generate predictions
  
Note how there are four major parameters we can play with. There are numerous encoding schemes and preprocessing methods, several models (which we also treat as parameters in the pipeline), each with varying numbers of hyperparameters. This means that we need to run the pipeline many, many times to find the optimal parameters that help the machine learning models generalise well.  
  
The process of optimising these major parameters to generate a single accuracy value on test data is just *one instance* of the pipeline. In the next post, I outline a strategy for cross validation, which will require us to run many instances of pipelines, but will give us reliable estimates of accuracy.  
  

