---
type: post  
title: "How (Not) to Invest III: Beating the Market with Clenow Momentum"  
bigimg: /img/hnti_logo_sq.jpg
image: http://financeandmarkets.com/wp-content/uploads/2018/06/cryptotrading.original.jpg
share-img: http://financeandmarkets.com/wp-content/uploads/2018/06/cryptotrading.original.jpg
share-img2: /img/hnti_logo_sq.jpg
tags: [finance, investing, how not to invest, algorithmic trading]
---  
  
# Introduction
In the first two posts in this series, I wrote about the MACD, RSI, and Stochastic oscillator, all of which are momentum indicators. In this third post, I write about a less-known indicator: Clenow momentum. In his brilliant book *Stocks on the Move: Beating the Market with Hedge Fund Momentum Strategies*, Andreas Clenow outlines a framework for measuring momentum. His approach involves using an exponential regression to measure the direction and quality of trends. He combines these metrics into a ranking system for choosing the highest momentum stocks at any point in time, and proposes a volatility-based approach to portfolio optimisation.  
  
Although some claim that Clenow Momentum beat the market, in theory, any publicly available indicator should not. In this post, we evaluate the effectiveness of Clenow Momentum as a trading strategy.

# Clenow Momentum (CM)
  
## The Core: Exponential Regression
Typically, we would run a simple linear regression of prices on time to estimate how quickly a stock is growing over time. However, that would not be feasible for comparisons across stocks. Hence, Clenow proposes an *exponential* regression in which we run a regression of *log prices* on time. That relates time to *change* in price - and we know that percentage change makes sense for comparisons.  
  
Clenow proposes a rolling period of 90 days for the exponential regression. Given stock traders' abilities to rapidly react to new information, our measure of CM would need to be much more responsive. Hence, I propose a rolling period of 60 days instead.  
  
### Component 1: Coefficient of Exponential Regression
The first component of Clenow Momentum (CM) measures the trend. In a linear regression of log prices on time, the coefficient  on time essentially tells us the percentage increase in returns for a one-unit change in time (e.g. one day for daily data, 15 minutes for 15-minute data). Clearly, the larger the coefficient, the stronger the trend. To make the coefficient more intuitive, we *annualise* the coefficient.  
  
### Component 2: R-squared (R2) Measure of Exponential Regression
Second, Clenow proposes the R-squared (R2) of the exponential regression as a measure of the quality of the trend. Consider the two scenarios below:


![](../graphics/2018-11-10-how-not-to-invest-clenow-momentum/plot1.png)



![](../graphics/2018-11-10-how-not-to-invest-clenow-momentum/plot2.png)


Assuming we have daily data, the slope is 0.30 for Stock A, which suggests that prices are increasing. However, the R2 is extremely low at **9.00%** due to volatility, especially on Day 2. For Stock B, the slope is even steeper, suggesting that prices are rising even faster than Stock A. It also has a lower R2. This means that the line was a much better fit to the data, due to the lower volatility of the stock price *around the general upward trend*. Thus, Clenow suggests that a stock that has a better R2 fit has a lower volatility, and has a higher probability of continuing its trend.

### Combined Component
Clenow posits that both the upward trend and goodness-of-fit are important. Hence, he combines both these measures by **multiplying them**:  
  
```
    Clenow Momentum (CM) = Annualised Slope of Regression Coefficient (%) * R2 of Regression (%)
```  

## Additional Conditions
Clenow slaps on two additional conditions for stock selection. We'll keep this in mind until we test a CM-based trading strategy.
  
1. **Never invest during a bear market.** The first of Clenow's (declared) additional conditions is to never invest when the market is declining. He does not believe in shorting stocks. Hence, he sets the rule that *if the S&P 500 is trading below its 200-day SMA, do not buy any stocks*.  
2. **The stock should be trending upward.** The second additional condition is that the stock should be trading above its 100-day SMA. This is used to quantitatively define an uptrend.  
3. **The stock price should be relatively stable.** The third of Clenow's additional conditions is that a stock should not have experienced a 15% or greater jump in price during the lookback period. Although this is implicitly captured under the goodness-of-fit (R2), Clenow implements this condition to ensure that prices are increasing gradually and not experiencing sudden jumps.  
  
## Stock Selection
Clenow proposes updating of portfolios on a weekly basis. On a fixed day each week, CM is calculated for a basket of stocks (the universe), and the stocks are ranked by their CM value. The greater the CM value, the higher the rank. Then, the portfolio is re-balanced. For now, the important thing to note is that the portfolio is updated **weekly** - this helps us to define the look-forward period for our tests: **5 days.**

# Testing Clenow Momentum I: Statistical Relationships
Clenow posits that the higher the CM score, the higher the probability of continued stock increases over the next 5 days. In this section, we test that assumption by computing the correlation between 5-day forward returns and CM, the R2 of exponential rolling-window regressions, and the slope coefficients of the regressions.  
  
In our computations, we use an adjusted version of CM (ACM), which incorporates the additional conditions stated above. In any time period where these conditions are not met, ACM is set to zero. ACM therefore represents a simplified version of Clenow's strategy.

From the graphs below, we see that on average, there was **no correlation between 5-day forward returns and ACM (which reflects the Clenow Momentum strategy) or the components of ACM: the regression R2 and slope**. The maximum size of the correlation coefficient in either direction (positive or negative) was at most 0.20, which is still relatively small.
  
![](../graphics/2018-11-10-how-not-to-invest-clenow-momentum/plot3.png)



![](../graphics/2018-11-10-how-not-to-invest-clenow-momentum/plot4.png)



![](../graphics/2018-11-10-how-not-to-invest-clenow-momentum/plot5.png)


Thus, we may conclude that statistically, there is no strong relationship between ACM or its components with forward 5-day returns.

# Testing Clenow Momentum II: Trading Returns
We run a trading simulation using a simplified version of Clenow's approach to test if the strategy beats the buy-and-hold benchmark. The simplified strategy involves the following:  
  
1. Maintaining an equally-weighted portfolio that is reset every week
2. Choosing the top 10 stocks by ACM instead of the top *N* stocks by standard deviation up to a fixed amount in principal
  
Note that for the CM-based strategy, we only run one simulation because the approach applies to a universe of stocks (in this case the S&P 500). If we wanted confirmation on different universes of stocks, we could easily run the same simulation on the S&P Mid-Cap 400 or the Russell 1000.  
  
Overall, **the CM-based strategy beat the market**. There were periods where the strategy performed worse than the S&P 500, but the overall drawdowns were significantly lower.
  
![](../graphics/2018-11-10-how-not-to-invest-clenow-momentum/plot6.png)
  
The performance of the CM-based strategy and the S&P 500 buy-and-hold benchmark are: (assuming a risk-free rate of 3.00%)
  
<div style="overflow-x:auto; width: 100%;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Strategy</th>
      <th>Returns</th>
      <th>Volatility</th>
      <th>Sharpe Ratio</th>
      <th>Max Drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CM</td>
      <td>8.73%</td>
      <td>8.45%</td>
      <td>0.677899</td>
      <td>-11.67%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Buy-and-Hold</td>
      <td>8.62%</td>
      <td>15.03%</td>
      <td>0.373798</td>
      <td>-27.33%</td>
    </tr>
  </tbody>
</table>
</div>



# How Did Clenow Momentum Beat the Market?!
Thus far, we found no statistically significant relationship between ACM and forward returns. However, we also found that Clenow's strategy beats the market! I propose two theories/potential reasons that explain these results.  
  
## 1. Momentum Works
Momentum is a sustained increase in price because demand outpaces supply. Hence, to make money from momentum, it makes sense to get in on the "demand side" as early as possible on a stock that other traders in the market will eventually buy. As long as demand continues to increase, returns for the early buyers will increase. Therefore, **momentum traders must buy stocks that other traders want, before the other traders want them**. This principal incentivises traders to make predictions on the next "hot stock", to monitor one another, and jump on the bandwagon as soon as a stock has shown some evidence of positive momentum. Therefore, **momentum works for traders who are able to identify stocks with momentum early enough**.  
  
This explanation may not have given credit to fundamental factors, but I recognise that fundamental analysis is equally important. The past financial crises have taught us that it is not enough for a stock to have price momentum; the company must have *value-creation momentum* as well. Only fundamental analysis can help us filter stocks that create value.  
  
## 2. Statistics Assumes a Single, Stable Underlying Relationship
In the trading simulation, note how the S&P 500 absolutely slayed the CM-based strategy. Only after the Dot-Com crash (first peak in the green graph) did the strategy start to match the S&P 500. Next, after the 2007 financial crisis, CM destroyed the S&P 500. I am not suggesting the strategy will become even more effective in the years to come. My point is that the relationship between indicators and stock prices change over time. Traditional indicators like MACD and RSI may have worked in the past, but don't work in the modern era. Newer indicators like CM failed in the past, but can now be used to beat the market.  
  
This phenomenon is related to investor psychology, which has been researched in depth by experts like [Robert Shiller](https://www.nytimes.com/2007/08/26/business/yourmoney/26view.html). On a broader level, stocks demonstrate how people think and survive. We observe how some practices translate to a reward, and we adapt. As more of us start to adapt, we crowd out the rewards. Some of us create new practices that are equally or more rewarding, and others adapt. This cycle causes old relationships (between practices and rewards) to break down, and new relationships to form. Consequently, there is no equilibrium; only efforts to strive toward one. We *invent* an equilibrium retrospectively (using past data) and *then* shape a narrative of it.  
  
In summary: relationships between indicators and stock prices are never the same...  
  
* In *different* time periods for the *same* stock
* In the *same* time period for *different* stocks
* In *different* time periods for *different* stocks
  
Consequently, traditional statistical tests (which assume some stable underlying relationship) will not be able to identify these relationships.

# Limitations
This study was limited in several ways. First, we employed a simplified version of Clenow's approach. We did not use Clenow's formula for portfolio re-balancing. Doing so would certainly have affected the portfolio weights. Second, we tested only one configuration: a lookback period of 60 days. Given that trading occurs at a faster pace, it may have been more appropriate to use a shorter lookback period and higher-resolution data. Third, we used only one set of data. I chose the S&P 500 as the universe of stocks that were input into the algorithm. Other more volatile indices like the S&P 400 would have produced different results. I also used daily returns for convenience. The strategy may or may not have been more effective on higher-resolution data. Although a more extensive study would be useful for validating Clenow's approach, the fact that it worked on the S&P 500 shows that it has massive potential.

# Conclusion [TLDR]
In this post, we showed that a strategy using Clenow Momentum **beat the S&P 500 buy-and-hold benchmark**, even though statistical tests showed no significant relation between 5-day forward returns and Clenow Momentum and it's individual components. I propose two explanations for this phenomenon: (1) momentum works and (2) relationships between indicators and stock returns change over time.  
  
---
Click [here](http://nbviewer.ipython.org/github/chrischow/dataandstuff/blob/3bb5473ee589962f95d4931d96356c5676dada90/notebooks/2018-11-10-how-not-to-invest-clenow-momentum.ipynb){:target="_blank"} for the full Jupyter notebook.
  
Credits for images: [FinanceAndMarkets.com](https://financeandmarkets.com/)
