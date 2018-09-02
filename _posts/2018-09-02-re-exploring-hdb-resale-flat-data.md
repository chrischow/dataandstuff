---
type: post
title: (Re-)Exploring HDB Resale Flat Data in 17 Graphs
tags: [real estate pricing, eda]
bigimg: /img/hdb_img.jpg
image: https://www.psd.gov.sg/heartofpublicservice/our-people/2-service/houses-to-homes/media/19980006213_-_0110-mr_0oizmon.jpg
share-img: /img/hdb_img.jpg
---
  
# A New Approach
About a year ago, I published [my first post on data&stuff](https://dataandstuff.wordpress.com/2017/08/14/hdb-resale-flat-prices-in-singapore/). I applied econometric techniques to develop three least squares regression models to explain HDB resale flat prices. A year on, I'm re-visiting the expanded dataset (now includes an additional year of data) with new skills and knowledge. This time, I intend to apply proper data science techniques to accurately predict prices.  
  
In this first post, I perform exploratory data analysis (EDA) on the dataset. In subsequent posts, I will develop a more complex regression model to predict resale flat prices.


```python
# Import
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# Settings
%matplotlib inline
warnings.filterwarnings('ignore')

# Read data
hdb = pd.read_csv('resale-flat-prices-based-on-registration-date-from-jan-2015-onwards.csv')
```
  
## Target: Resale Prices
As we can see, resale prices are right-skewed (mean is to the right of the median). The mean resale price transacted was a whopping $440,000. Singaporeans must be crazy rich to afford a resale flat in this era.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot1.png)
  
## Date and Month Purchased
First, note that the `month` feature combines both the month and the year. Let's split these up while preserving the original notation.
  
```python
# Rename month variable
hdb = hdb.rename(columns={'month': 'year_mth'})

# Add variables for month and year
hdb['year'] = pd.to_numeric(hdb.year_mth.str[:4])
hdb['month'] = pd.to_numeric(hdb.year_mth.str[5:])
```
  
From the graph below, we find that there are "hot" and "cold" periods for buying resale flats, with a surge in recent months. We note how lots of transactions take place on a regular basis: at least 1,000 per month. At the median price, that's approximately $4.4 billion transacted per month.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot2.png)
  
### Relation with Target
Plotting the median resale price from 2015 onwards, we find that the median price has remained stable over time. In addition, the variation in prices has remained relatively wide. Hence, as in my [first post](https://dataandstuff.wordpress.com/2017/08/14/hdb-resale-flat-prices-in-singapore/) on HDB resale flat prices, we will assume that the relationship between the flat characteristics and resale flat prices is stable for all transactions in the dataset. In other words, we treat the transactions as having occurred within a single, stable time period.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot3.png)

# Town
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot4.png)
  
### Relation with Target
We find high variability in resale flat prices across the respective towns. This tells us that towns are an important factor in predicting resale flat prices.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot5.png)
  
## Flat Type
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot6.png)
  
### Relation to Target
Naturally, we would expect flats that are "high SES" to have a higher resale price:
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot7.png)
  
## Storey Range
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot8.png)
  
### Relation to Target
Conventional wisdom would tell us that the higher the storey, the nicer the view. The nicer the view, the higher the resale price. The data appears to agree.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot9.png)
  
## Floor Area
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot10.png)
  
### Relation to Target
Conventional wisdom would also suggest a positive relationship between floor area and price. Yet again, the data appears to agree.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot11.png)
  
## Flat Model
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot12.png)
  
### Relation to Target
There appears to be high variability in resale prices across flat types. This suggests that flat types will be useful for prediction.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot13.png)
  
## Lease Commencement Date
Although we expect a higher price for later lease commencement dates, the relationship is not all that clear. Perhaps remaining lease is a bigger factor.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot14.png)
  
### Relation to Target
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot15.png)
  
## Remaining Lease
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot16.png)
  
### Relation to Target
We find a positive relationship between resale price and the remaining years in lease from 50 to 90 years. However, from 90 years onwards (referring to Build-to-Order (BTO) flats sold in the last 5 years), the relationship weakens substantially, and the variation increases substantially as well. This suggests that we could create a special category for transactions of flats with 90 or more years remaining in their leases to predict resale flat prices.
  
![](../graphics/2018-09-02-re-exploring-hdb-resale-flat-data-plot17.png)
  
---
  
Click [here](http://nbviewer.jupyter.org/github/chrischow/dataandstuff/blob/725111d6d9eca9a525fcfff2f7a36d39a6500db8/notebooks/2018-09-02-re-exploring-hdb-resale-flat-data.ipynb){:target="_blank"} for the full Jupyter notebook.
