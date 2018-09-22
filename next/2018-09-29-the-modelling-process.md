# Introduction
Thus far, I've written about exploratory data analysis (EDA) and feature engineering in the HDB resale flat dataset. What I should have done earlier was to outline the overall approach for modelling. In this post, I introduce the broad approach to the resale flat price model and provide some preliminary results on the progress made.  
  
# The End State
The objective of all this work is to develop something that can accurately predict (or value) resale flats. This thing takes in data provided by HDB through [Data.gov.sg](https://data.gov.sg/), generates new features, cleans them, runs them through a complex machine learning model, and spits out predicted prices to **enable home buyers to assess the value of prospective homes**. This thing is a **machine learning pipeline**.
  
# The Pipeline
In my posts, I typically used the term "model" to refer to machine learning algorithms. However, it is important to distinguish between machine learning *models* and the overarching pipeline that contains numerous key steps as part of the machine learning *process*. We can think of the pipeline as a macro model comprised of the following steps:  
  
1. **Feature engineering.** In this step, we create new features from the limited set of features provided to us. I wrote about engineering [numeric features](https://chrischow.github.io/dataandstuff/2018-09-08-hdb-feature-engineering-i/), [categorical features](http://www.google.com), and [geocoding](https://chrischow.github.io/dataandstuff/2018-09-16-hdb-feature-engineering-ii/) earlier, and commented on how *the way* features are encoded - binning method for numeric features and encoding methods for categorical features - are parameters in the pipeline. In the pipeline, many combinations of encoding techniques are tested to find the optimal encoding method for each model.
2. **Hyperparameter tuning.** In this step, we optimise several regression models separately. Each model (less OLS regression) has its own set of hyperparameters that governs how the model learns old data and how well it performs on unseen data. The objective of hyperparameter tuning is to reduce *overfitting*, which I will explain the coming sections.
3. **Prediction.** In this final step, we generate predictions on pseudo-unseen data to evaluate the model.
  
