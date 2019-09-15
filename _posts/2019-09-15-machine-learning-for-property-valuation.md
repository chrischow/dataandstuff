---
type: post  
title: "Machine Learning for Property Valuation"  
bigimg: /img/property_pricing.jpg
image: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/property_pricing_sq.jpg
share-img: /img/property_pricing.jpg
share-img2: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/property_pricing_sq.jpg
tags: [data science, real estate]
---  

# Discovering Property Valuation
Recently, I discussed the property market with a friend who was a real estate agent. I became fascinated with the real estate market: the marketing, the negotiations, the incentives, and the contracting process. Of greatest interest to me was property valuation. I learned that property sellers used various vendors for valuation, from licensed surveyors to free online tools. Generally, these vendors were not transparent about the techniques they employed, besides stating that these were proprietary. Just for fun, I decided to develop an ML model of my own to explore how ML can be used to value properties.

# Existing Valuers
To know how good our model is, we need benchmarks. Here are the main valuers that property sellers go to:  
  
### The Singapore Institute of Surveyors and Valuers (SISV)
First, we have the [SISV](http://www.sisv.org.sg/), defined by the [Straits Times](https://www.straitstimes.com/business/property/srx-operator-sues-major-consultancies-over-its-valuation-service) as *\"a professional body representing mainly land surveyors, quantity surveyors, valuers, real estate agents and property managers\"*. Its valuers are "licensed under the Appraisers Act", and must have ["a relevant educational background and adequate practical experience"](http://www.sisv.org.sg/vgp-about.aspx). Therefore, unless you have a Bachelor's Degree in Real Estate, Property Management, or a similar field, you would not know how exactly your property is being valued, and you would not know how to evaluate the accuracy of the valuation you're given. In fact, there are no open records of how accurate SISV's valuations are. Unfortunately, property sellers have high willingness to pay for SISV's services because SISV is licensed by the government to perform valuations.  
  
### Singapore Real Estate Exchange (SRX)
From [Wikipedia](https://en.wikipedia.org/wiki/Singapore_Real_Estate_Exchange), SRX is *\"a consortium of leading real estate agencies administered by StreetSine Technology Group in Singapore\"*. It offers clients several recommended prices, the most famous of which is "X-Value", a prediction of a property's value, generated using Comparable Market Analysis (CMA) and property characteristics. See their [white paper](https://s3-ap-southeast-1.amazonaws.com/static.streetsine/SPI/spi_white_paper2.pdf) and [statistics on X-Value's accuracy](https://www.srx.com.sg/XValue-performance).  
  
### UrbanZoom
UrbanZoom is a property research tool that employs artificial intelligence (AI) to generate accurate property valuations. Their aim is to bring more transparency to the real estate market, because they believe that [*everyone should be able to buy or sell their homes without any fear of misinformation*](https://www.urbanzoom.com/about). UrbanZoom also shares some [accuracy statistics](https://www.urbanzoom.com/explanation) on its valuation tool, Zoom Value.
  
## On Transparency
I applaud SRX and UrbanZoom for using modern technical methods to generate valuations, and for openly publishing the accuracy statistics on their predictions. This is a fresh move away from the traditional approach, which employs opaque valuation methods that are protected by legislation. I couldn't agree more with UrbanZoom's philosophy, because the negative effects of information asymmetry are amplified in real estate, where each transaction involves hundreds of thousands of dollars. Mispricing a property could mean forgone savings for a child's university education, or a substantial amount of retirement funds.  
  
In this post, I plan to take transparency one step further by providing a detailed walkthrough of my ML model, which comes close to matching SRX's X-Value and UrbanZoom's Zoom Value.  
  
**Disclaimer:** This post represents only the ML perspective. I used basic ML techniques on open data to generate all findings in this post. I have close to no experience in the property market, and have had no consultations with anyone working in SRX or UrbanZoom.

# Executive Summary [TL;DR]
  
* I modelled the prices of private non-landed and landed properties, and resale HDBs. I called this prediction service "C-Value".
* C-Value could not beat X-Value's and Zoom Value's accuracy, with accuracy measured as the *median error* and the *proportion of predictions within given margins of error*.
* C-Value arguably provides a good-enough valuation of private non-landed properties and resale HDBs. Based on the median transaction prices for each property category:
    * The error difference (from X-Value / Zoom Value) of **0.04%** for private non-landed properties corresponds to **\$480**.
    * The error difference of **0.1%** for resale HDBs corresponds to **\$410**.
    * The error difference of **0.3%** for private landed properties corresponds to **\$9,000**.
    
* The chosen algorithms were:  
    * **Private Non-Landed:** K-Nearest Neighbours Regression
    * **Resale HDB:** Gradient Boosting Regression (LightGBM implementation)
    * **Private Landed:** Gradient Boosting Regression (LightGBM implementation)

## SRX's X-Value
SRX uses 4 main metrics to evaluate X-Value: (a) Purchase Price Deviation (PPD), and percentage of price deviations within (b) 5%, (c) 10%, and (d) 20% of X-Value. Metric (a) is simply `Transacted Price - X-Value`, while the percentage price deviations are computed as `( Transacted Price - X-Value ) / X-Value`. To see the statistics for X-Value, see SRX's webpage [here](https://www.srx.com.sg/XValue-performance). As shown in the table below, C-Value came close to matching X-Value in generating accurate predictions.  





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Private Non-Landed</th>
      <th colspan="2" halign="left">HDB Resale</th>
      <th colspan="2" halign="left">Private Landed</th>
    </tr>
    <tr>
      <th>Metric</th>
      <th>X-Value</th>
      <th>C-Value</th>
      <th>X-Value</th>
      <th>C-Value</th>
      <th>X-Value</th>
      <th>C-Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Median PPD</td>
      <td>2.7%</td>
      <td>2.74%</td>
      <td>3.1%</td>
      <td>3.2%</td>
      <td>7.7%</td>
      <td>8.0%</td>
    </tr>
    <tr>
      <td>Within 5%</td>
      <td>70.5%</td>
      <td>69.6%</td>
      <td>70.5%</td>
      <td>69.2%</td>
      <td>36.7%</td>
      <td>34.6%</td>
    </tr>
    <tr>
      <td>Within 10%</td>
      <td>89.1%</td>
      <td>87.7%</td>
      <td>93.0%</td>
      <td>93.2%</td>
      <td>59.8%</td>
      <td>58.3%</td>
    </tr>
    <tr>
      <td>Within 20%</td>
      <td>97.7%</td>
      <td>97.3%</td>
      <td>99.2%</td>
      <td>99.7%</td>
      <td>82.8%</td>
      <td>83.2%</td>
    </tr>
  </tbody>
</table>
</div>



## UrbanZoom's Zoom Value
UrbanZoom provides [similar statistics](https://www.urbanzoom.com/explanation) for Zoom Value: (a) Mean Absolute Percentage Error (MAPE) and the percentages of predictions that fell within (a) 5%, (b) 10%, and (c) 20% of actual price. MAPE is computed as `( Zoom Value - Transacted Price ) / Transacted Price`. Note that Zoom Value combines predictions for only Condominiums (Private Non-Landed) and resale HDBs. See a comparison of Zoom Value and C-Value in the table below.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Metric</th>
      <th>Zoom Value</th>
      <th>C-Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Median Error</td>
      <td>3.0%</td>
      <td>3.03%</td>
    </tr>
    <tr>
      <td>Within 5%</td>
      <td>70%</td>
      <td>69.3%</td>
    </tr>
    <tr>
      <td>Within 10%</td>
      <td>90%</td>
      <td>91.0%</td>
    </tr>
    <tr>
      <td>Within 20%</td>
      <td>98%</td>
      <td>98.7%</td>
    </tr>
  </tbody>
</table>
</div>



# Private Non-Landed and Private Landed Property

## The Data
The models for private non-landed and landed property were developed using [URA caveat data](https://www.ura.gov.sg/realEstateIIWeb/transaction/search.action) from Aug 2016 to Aug 2019. As you can see in the table below, there were 16 features, including a manual tagging of Non-Landed / Landed under the `category` feature, and excluding the serial number of each entry.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>S/N</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Project Name</th>
      <td>WALLICH RESIDENCE</td>
      <td>QUEENS</td>
      <td>ONE PEARL BANK</td>
    </tr>
    <tr>
      <th>Street Name</th>
      <td>WALLICH STREET</td>
      <td>STIRLING ROAD</td>
      <td>PEARL BANK</td>
    </tr>
    <tr>
      <th>Type</th>
      <td>Apartment</td>
      <td>Condominium</td>
      <td>Apartment</td>
    </tr>
    <tr>
      <th>Postal District</th>
      <td>02</td>
      <td>03</td>
      <td>03</td>
    </tr>
    <tr>
      <th>Market Segment</th>
      <td>CCR</td>
      <td>RCR</td>
      <td>RCR</td>
    </tr>
    <tr>
      <th>Tenure</th>
      <td>99 yrs lease commencing from 2011</td>
      <td>99 yrs lease commencing from 1998</td>
      <td>99 yrs lease commencing from 2019</td>
    </tr>
    <tr>
      <th>Type of Sale</th>
      <td>Resale</td>
      <td>Resale</td>
      <td>New Sale</td>
    </tr>
    <tr>
      <th>No. of Units</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Price ($)</th>
      <td>4.3e+06</td>
      <td>1.55e+06</td>
      <td>992000</td>
    </tr>
    <tr>
      <th>Nett Price ($)</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Area (Sqft)</th>
      <td>1313</td>
      <td>1184</td>
      <td>431</td>
    </tr>
    <tr>
      <th>Type of Area</th>
      <td>Strata</td>
      <td>Strata</td>
      <td>Strata</td>
    </tr>
    <tr>
      <th>Floor Level</th>
      <td>51 to 55</td>
      <td>26 to 30</td>
      <td>01 to 05</td>
    </tr>
    <tr>
      <th>Unit Price ($psf)</th>
      <td>3274</td>
      <td>1309</td>
      <td>2304</td>
    </tr>
    <tr>
      <th>Date of Sale</th>
      <td>Aug-2019</td>
      <td>Aug-2019</td>
      <td>Aug-2019</td>
    </tr>
    <tr>
      <th>category</th>
      <td>NonLanded</td>
      <td>NonLanded</td>
      <td>NonLanded</td>
    </tr>
  </tbody>
</table>
</div>



### Data Cleaning
The data was generally clean, with the exception of the `Tenure` feature. Here were the key issues and my steps to resolve them:  
* **Missing entries for `Tenure`:** Filled in the information using a simple Google search on the relevant properties.
* **Tenure without start year:** I found out that these were properties that were still under construction. I filled in the completion year for these properties through Google.
* **Freehold properties had no starting year for tenure:** I made the assumption that all Freehold properties were constructed in 1985. On hindsight, this could have affected C-Value's accuracy.
  
### Feature Engineering
I performed simple feature engineering to extract more value from the dataset. Prior to model training:
* I expanded `Tenure` into three new features to capture more characteristics about the properties:  
    * Remaining Lease
    * Age
    * Freehold? (binary feature)
* I converted Floor Level into a numeric feature by taking the upper floor within each range. By doing so, we allowed price to be positively correlated to the differences in level between any two given units. On the other hand, one-hot encoding (OHE) would not have allowed us to do this. 
* I also converted all non-numeric features into binary features, and dropped unused features like Price, Nett Price, and Date of Sale. The date of sale was dropped because I assumed that there were no major price changes from Aug 2016 to Aug 2019. This may not have been the best assumption.
  
In model training, I generated a whole set of binary features by applying Term Frequency-Inverse Document Frequency (TF-IDF) to the Project Name and Street Name. Natural Language Processing (NLP) enabled me to make full use of the dataset. In fact, these features turned out to be extremely useful. The intuition behind this was to provide the model with greater fidelity on the properties' location. The **project** captured properties in the same development, while the **street** captured properties in the same neighbourhood. These were created during model training to avoid leakage from incorporating project names and street names that were in the test sets.

## Model Training
### Features
For both models, I used the same set of features:  

* Type*
* Type of Sale*
* Type of Area*
* Market Segment*
* Postal District*
* Age
* Area (Sqft)
* Floor
* Street Name (in binary TF-IDF features)
* Project Name (in binary TF-IDF features)
* Remaining Lease
* Freehold

The features with an asterix were encoded using OHE.

### Algorithms
First, I chose K-Nearest Neighbours Regression (K-NN) because the way this algorithm works is extremely similar to how we price properties. In property pricing, we ask "how much did that similar property sell for?" K-NN is effectively the ML implementation of this approach. "Similar" is defined in terms of the features (characteristics of the property) that we put into the model.  

Second, I chose Gradient Boosting Regression (LightGBM implementation) simply because it is generally a good algorithm. It is fast, effective, flexible, and can model non-linear relationships.

I did not perform any hyperparameter optimisation for both algorithms in any of the models. This was because I wanted a quick and dirty gauge on how useful ML could be. Only after seeing the models' results did I see a point in further optimisation.

### Evaluation
To evaluate the models, I used 20 repeats of 5-fold cross validation (CV) to generate distributions (*n = 100*) of each evaluation metric. The metrics were:  
  
* Median Purchase Price Deviation (PPD)
* (Absolute) Price Deviations within 5% of C-Value
* (Absolute) Price Deviations within 10% of C-Value
* (Absolute) Price Deviations within 20% of C-Value
* Mean Absolute Percentage Error (MAPE)
* Mean Absolute Error (MAE)
* R-squared (R2)
* Root Mean Squared Error (RMSE)

I wrote a custom function to run the repeated CV with the following steps for each iteration:
1. Create binary TF-IDF features for Project Name and Street Name on the training set and transform the same features in the test set
2. Normalise the data if required (only for K-NN)
3. Fit a model
4. Compute and save metrics
5. Go to next iteration and start from Step 1

### Model for Private Non-Landed Properties
In the code below, I configured the cross validation object and the data, and ran the K-NN and LightGBM algorithms using my custom function. I reported the mean of the relevant metrics as the final result.


```python
# Configure CV object
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=100)

# Configure data
all_vars = ['Type', 'Type of Sale', 'Type of Area', 'Market Segment', 'Postal District', 'age', 'Area (Sqft)', 'floor', 'Street Name', 'Project Name', 'remaining_lease', 'freehold']
dum_vars = ['Type', 'Type of Sale', 'Type of Area', 'Market Segment', 'Postal District']

# Landed data
df_landed = df[df.category == 'NonLanded']

# Configure data
X_data = pd.get_dummies(df_landed[all_vars],
                        columns=dum_vars)
y_data = df_landed['Unit Price ($psf)']

# Run K-NN model
kn = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=4)
kn_cv = custom_cv(kn, X_data, y_data, cv, norm=True, ppd=True)

# Run LightGBM model
lg = LGBMRegressor(n_estimators=500, max_depth=10, random_state=123, n_jobs=4, reg_alpha=0.1, reg_lambda=0.9)
lg_cv = custom_cv(lg, X_data, y_data, cv, norm=False, early=False, ppd=True)
```

From the results below, we see that K-NN was the better algorithm. However, overall, C-Value did not match up to X-Value across all metrics. Therefore, C-Value is not as robust an ML estimate as X-Value is. However, the difference in median error (0.04%) at the median non-landed price of \$1.2M corresponded to a price difference of only \$480.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Metric</th>
      <th>X-Value</th>
      <th>K-NN</th>
      <th>LightGBM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Median PPD</td>
      <td>2.7%</td>
      <td>2.74%</td>
      <td>3.60%</td>
    </tr>
    <tr>
      <td>Within 5%</td>
      <td>70.5%</td>
      <td>69.61%</td>
      <td>62.57%</td>
    </tr>
    <tr>
      <td>Within 10%</td>
      <td>89.1%</td>
      <td>87.71%</td>
      <td>85.15%</td>
    </tr>
    <tr>
      <td>Within 20%</td>
      <td>97.7%</td>
      <td>97.29%</td>
      <td>96.95%</td>
    </tr>
    <tr>
      <td>MAPE</td>
      <td>-</td>
      <td>4.78%</td>
      <td>5.53%</td>
    </tr>
    <tr>
      <td>MAE</td>
      <td>-</td>
      <td>$64</td>
      <td>$75</td>
    </tr>
    <tr>
      <td>R2</td>
      <td>-</td>
      <td>95.33%</td>
      <td>94.46%</td>
    </tr>
    <tr>
      <td>RMSE</td>
      <td>-</td>
      <td>$111</td>
      <td>$121</td>
    </tr>
  </tbody>
</table>
</div>


![](../graphics/2019-09-15-machine-learning-for-property-pricing/output_19_0.png)


### Model for Private Landed Properties
The approach taken was the same as before. 


```python
# Configure CV object
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=100)

# Configure data
all_vars = ['Type', 'Type of Sale', 'Type of Area', 'Market Segment', 'Postal District', 'age', 'Area (Sqft)', 'floor', 'Street Name', 'Project Name', 'remaining_lease', 'freehold']
dum_vars = ['Type', 'Type of Sale', 'Type of Area', 'Market Segment', 'Postal District']

# Landed data
df_landed = df[df.category == 'Landed']

# Configure data
X_data = pd.get_dummies(df_landed[all_vars],
                        columns=dum_vars)
y_data = df_landed['Unit Price ($psf)']

# Run K-NN model
kn = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=4)
kn_cv = custom_cv(kn, X_data, y_data, cv, norm=True, ppd=True)

# Run LightGBM model
lg = LGBMRegressor(n_estimators=500, max_depth=8, random_state=123, n_jobs=4, reg_alpha=0.1, reg_lambda=0.9)
lg_cv = custom_cv(lg, X_data, y_data, cv, norm=False, early=False, ppd=True)
```

This time, LightGBM was the better algorithm. However, once again, C-Value did not match up to X-Value across all metrics, except in predicting prices within a 20% margin of error. The difference in median error (0.31%) at the median landed price of \$3M corresponded to a price difference of \$9.3k.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Metric</th>
      <th>X-Value</th>
      <th>K-NN</th>
      <th>LightGBM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Median PPD</td>
      <td>7.7%</td>
      <td>8.28%</td>
      <td>8.01%</td>
    </tr>
    <tr>
      <td>Within 5%</td>
      <td>36.7%</td>
      <td>35.86%</td>
      <td>34.64%</td>
    </tr>
    <tr>
      <td>Within 10%</td>
      <td>59.8%</td>
      <td>55.81%</td>
      <td>58.26%</td>
    </tr>
    <tr>
      <td>Within 20%</td>
      <td>82.8%</td>
      <td>79.00%</td>
      <td>83.21%</td>
    </tr>
    <tr>
      <td>MAPE</td>
      <td>-</td>
      <td>13.57%</td>
      <td>12.07%</td>
    </tr>
    <tr>
      <td>MAE</td>
      <td>-</td>
      <td>$157</td>
      <td>$141</td>
    </tr>
    <tr>
      <td>R2</td>
      <td>-</td>
      <td>68.87%</td>
      <td>$75.81%</td>
    </tr>
    <tr>
      <td>RMSE</td>
      <td>-</td>
      <td>$242</td>
      <td>$213</td>
    </tr>
  </tbody>
</table>
</div>



![](../graphics/2019-09-15-machine-learning-for-property-pricing/output_24_0.png)


# Resale HDBs

## The Data
The model for resale HDBs was developed using [resale flat price data from HDB](https://data.gov.sg/dataset/resale-flat-prices), from Jan 2015 to Aug 2019. This dataset comprised 11 features, and we used all of them except the transaction month.


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>S/N</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>month</th>
      <td>2015-01</td>
      <td>2015-01</td>
      <td>2015-01</td>
    </tr>
    <tr>
      <th>town</th>
      <td>ANG MO KIO</td>
      <td>ANG MO KIO</td>
      <td>ANG MO KIO</td>
    </tr>
    <tr>
      <th>flat_type</th>
      <td>3 ROOM</td>
      <td>3 ROOM</td>
      <td>3 ROOM</td>
    </tr>
    <tr>
      <th>block</th>
      <td>174</td>
      <td>541</td>
      <td>163</td>
    </tr>
    <tr>
      <th>street_name</th>
      <td>ANG MO KIO AVE 4</td>
      <td>ANG MO KIO AVE 10</td>
      <td>ANG MO KIO AVE 4</td>
    </tr>
    <tr>
      <th>storey_range</th>
      <td>07 TO 09</td>
      <td>01 TO 03</td>
      <td>01 TO 03</td>
    </tr>
    <tr>
      <th>floor_area_sqm</th>
      <td>60</td>
      <td>68</td>
      <td>69</td>
    </tr>
    <tr>
      <th>flat_model</th>
      <td>Improved</td>
      <td>New Generation</td>
      <td>New Generation</td>
    </tr>
    <tr>
      <th>lease_commence_date</th>
      <td>1986</td>
      <td>1981</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>remaining_lease</th>
      <td>70</td>
      <td>65</td>
      <td>64</td>
    </tr>
    <tr>
      <th>resale_price</th>
      <td>255000</td>
      <td>275000</td>
      <td>285000</td>
    </tr>
  </tbody>
</table>
</div>



### Data Cleaning
The dataset was extremely clean. I only made three changes:  
1. **Convert `floor area` from square metres to square feet:** This allows us to compare MAE and RMSE with the other models, since they were computed using square feet. 
2. **Convert `price` to `price per square foot (PSF)`:** For alignment with the other models.
3. **Convert `remaining lease` into years:** This was originally included as a string. I simply removed all text except the number of years.

### Feature Engineering
Prior to model training, I created the `age` and `floor` features, as per the private property dataset. During model training, I applied the same NLP concept for street names (binary TF-IDF to capture more location data).

The one thing I did differently was to generate new features for the units' block. I separated block numbers from block letters, and created binary features for each block number and letter. The idea here was to add more location information. Flats previously sold within the same block should have some influence on the price of any given flat in that block.

## Model Training

### Algorithms and Evaluation
I used the same approach as before.

### Model for Resale HDBs


```python
# Configure CV object
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=100)

# Read data
resale = pd.read_csv('HDB/resale_final.csv')

# FE for blocks
resale['block_num'] = resale.block.str.replace('[A-Z]', '')
resale['block_letter'] = resale.block.str.replace('[0-9]', '')
resale = resale.drop('block', axis=1)

# Configure features
hdb_vars = ['town', 'flat_type', 'block_num', 'block_letter', 'street_name', 'sqft', 'flat_model', 'remaining_lease', 'price', 'age', 'floor']
hdb_dum_vars = ['town', 'flat_type', 'flat_model', 'block_num', 'block_letter']

# Configure data
X_data = pd.get_dummies(resale[hdb_vars],
                        columns=hdb_dum_vars).drop('price', axis=1)
y_data = resale['price']

# Run K-NN model
kn = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=4)
kn_cv = custom_cv(kn, X_data, y_data, cv, norm=True, ppd=True)

# Run LightGBM model
lg = LGBMRegressor(n_estimators=5000, max_depth=25, random_state=123, n_jobs=4, reg_alpha=0.1, reg_lambda=0.9)
lg_cv = custom_cv(lg, X_data, y_data, cv, norm=False, early=False, ppd=True)
```

Once again, LightGBM was the better algorithm. Yet, C-Value fared worse than X-Value in (1) the median error and in (2) predicting prices within a 5% margin of error. The difference in median error (0.1%) at the median resale HDB price of \$410k corresponded to a price difference of only \$410.


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Metric</th>
      <th>X-Value</th>
      <th>K-NN</th>
      <th>LightGBM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Median PPD</td>
      <td>3.1%</td>
      <td>3.93%</td>
      <td>3.20%</td>
    </tr>
    <tr>
      <td>Within 5%</td>
      <td>70.5%</td>
      <td>60.04%</td>
      <td>69.14%</td>
    </tr>
    <tr>
      <td>Within 10%</td>
      <td>93.0%</td>
      <td>87.73%</td>
      <td>93.27%</td>
    </tr>
    <tr>
      <td>Within 20%</td>
      <td>99.2%</td>
      <td>99.19%</td>
      <td>99.72%</td>
    </tr>
    <tr>
      <td>MAPE</td>
      <td>-</td>
      <td>5.08%</td>
      <td>4.12%</td>
    </tr>
    <tr>
      <td>MAE</td>
      <td>-</td>
      <td>$21</td>
      <td>$17</td>
    </tr>
    <tr>
      <td>R2</td>
      <td>-</td>
      <td>92.87%</td>
      <td>95.29%</td>
    </tr>
    <tr>
      <td>RMSE</td>
      <td>-</td>
      <td>$29</td>
      <td>$23</td>
    </tr>
  </tbody>
</table>
</div>


![](../graphics/2019-09-15-machine-learning-for-property-pricing/output_36_0.png)


# Zoom Value: Private Non-Landed and Resale HDBs
I combined the best models for private non-landed properties (K-NN) and resale HDBs (LightGBM) to create the C-Value equivalent to Zoom Value. Overall, C-Value couldn't match Zoom Value in terms of the median error and the proportion of predictions within 5% accuracy. There was no comparable price difference resulting from the difference in median error, because UrbanZoom did not break down the accuracy statistics by the type of property.


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Metric</th>
      <th>Zoom Value</th>
      <th>C-Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Median Error</td>
      <td>3%</td>
      <td>3.03%</td>
    </tr>
    <tr>
      <td>Within 5%</td>
      <td>70%</td>
      <td>69.33%</td>
    </tr>
    <tr>
      <td>Within 10%</td>
      <td>90%</td>
      <td>91.02%</td>
    </tr>
    <tr>
      <td>Within 20%</td>
      <td>98%</td>
      <td>98.74%</td>
    </tr>
  </tbody>
</table>
</div>


![](../graphics/2019-09-15-machine-learning-for-property-pricing/output_39_0.png)


# Improving the Model
Although the dollar price differences due to the median error differences were small, C-Value failed to beat X-Value and Zoom Value in terms of hard ML performance metrics. Here are some possible reasons why:  
  
1. **Too little data.** The models were trained on relatively recent data, as opposed to a large bank of data on housing prices. The good thing about this approach is that it avoids using old data that may have become irrelevant. The tradeoff is that *some* of the older data could be useful. For example, 5 years could have been an acceptable time frame for the dataset.
2. **No temporal effects.** The models assumed that relationships in the data were stable during the time frames in the respective dataset. However, one month's prices could be related to the following month's prices. Including temporal features like lagged prices for a given type of property could improve prediction.
3. **Simple algorithms.** K-NN was a good model because it accurately models business thinking on valuation, and LightGBM is a good model in general. However, other sophisticated approaches could have been tested. For example, stacked regression can be used to control outlying predictions and thereby improve MAPE.

# Conclusion
In this post, I showed that basic ML algorithms could produce acceptable valuations (C-Value) of Singapore properties. Although these could not match up to the predictive accuracy of X-Value and Zoom Value in terms of median error and the proportion of predictions within given margins of error, these were good enough for providing a rough estimate of value.

More generally, I showed that there is value in using ML for quantifying relationships between characteristics of a property (and/or its similar properties) and its market price. This also means that ML can be used to quantify and recommend a fair **listing** price. This would be an alternative to [SRX's X-Listing Price](https://www.srx.com.sg/ask-home-prof/13952/streetsine-launches-x-listing-price-to-improve-real-estate-pricing), the only all-in-one recommendation service provided to sellers that is available on the market. I'll save this for next time :)

---
Credits for image: [Nanalyze](https://www.nanalyze.com/)
