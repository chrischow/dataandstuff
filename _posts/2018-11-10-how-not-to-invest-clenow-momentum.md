
# How (Not) to Invest III - Momentum that Beats the Market
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


```python
# Import required modules
import fix_yahoo_finance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings
from yahoo_finance import Share

# Settings
warnings.filterwarnings('ignore')

# Override pdr
yf.pdr_override()

# Import stocklist
sp500 = pd.read_csv('sp500.csv')
```


```python
# Modify settings
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.axis'] = 'y'
mpl.rcParams['grid.color'] = '#e8e8e8'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.color'] = '#494949'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.color'] = '#494949'
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.edgecolor'] = '#494949'
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.labelpad'] = 15
mpl.rcParams['axes.labelcolor'] = '#494949'
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Raleway'
mpl.rcParams['scatter.marker'] = 'h'

# Colours
def get_cols():
    
    print('[Colours]:')
    print('Orange:     #ff9966')
    print('Navy Blue:  #133056')
    print('Light Blue: #b1ceeb')
    print('Green:      #6fceb0')
    print('Red:        #f85b74')

    return
```


```python
# Create fake data: low R2
low_r2 = pd.DataFrame(
    {
        'a': [1,2,3,4,5],
        'b': [1,5,2,4,3]
    }
)

# Create fake data: high R2
high_r2 = pd.DataFrame(
    {
        'a': [1,2,3,4,5],
        'b': [1,3,2,3.5,3]
    }
)

a1,b1 = np.polyfit(low_r2.a, low_r2.b, 1)
a2,b2 = np.polyfit(high_r2.a, high_r2.b, 1)

# Calculate R2 for low R2 data
lm_lowr2 = LinearRegression()
lm_lowr2.fit(low_r2[['a']], low_r2.b)
r2_low = lm_lowr2.score(low_r2[['a']], low_r2.b)
coef_low = lm_lowr2.coef_[0]

# Calculate R2 for high R2 data
lm_hir2 = LinearRegression()
lm_hir2.fit(high_r2[['a']], high_r2.b)
r2_high = lm_hir2.score(high_r2[['a']], high_r2.b)
coef_high = lm_hir2.coef_[0]
```


```python
plt.figure(figsize=(10, 8))
plt.plot(low_r2.a, low_r2.b)
plt.plot(low_r2.a, b1 + a1*low_r2.a)
plt.title('Stock A - R2: ' + '{0:.2f}%'.format(r2_low*100) + ' | Slope: ' + '{0:.2f}'.format(coef_low), fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(high_r2.a, high_r2.b)
plt.plot(high_r2.a, b2 + a2*high_r2.a)
plt.title('Stock B - R2: ' + '{0:.2f}%'.format(r2_high*100) + ' | Slope: ' + '{0:.2f}'.format(coef_high), fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.show()
```


![png](output_5_0.png)



![png](output_5_1.png)


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


```python
# # Fix start date and end date
# start_date = '1979-01-01'
# end_date = '2018-06-01'

# # Pull data
# alldata = pdr.get_data_yahoo(list(sp500.Symbol), start_date, end_date, progress=False)

# # Pull S&P500 data
# sp500_index = pdr.get_data_yahoo('^GSPC', start_date, end_date, progress = False)

# # Save data
# sp500_index.to_csv('sp500_index.csv')
```


```python
# # Save data
# alldata.to_csv('sp500_full.csv')
# alldata.Close.to_csv('sp500_close.csv')
```

We investigate the statistical relationships between  using the **correlation coefficient**. 

### Function for Computing ACM Thresholds


```python
def compute_acm(stk, lookback, full_df):
    
    # Pull data from main data frame
    stock_data = full_df[['days', 'sp500']].copy()
    stock_data['close'] = full_df[stk].copy()
    
    # Remove NaNs
    # stock_data.dropna(axis = 0, inplace = True)
    
    # Run rolling regression with 30-day windows
    temp_lm = pd.stats.ols.MovingOLS(
        y=np.log(stock_data.close), x=stock_data[['days']],
        window_type='rolling', window=lookback, intercept=True
    )
    
    # Compute Clenow Momentum
    stock_data['slope'] = ((temp_lm.beta.days + 1) ** 250 - 1) * 100
    stock_data['r2'] = temp_lm.r2 * 100
    stock_data['cm'] = stock_data.slope * stock_data.r2

    # Compute forward returns
    stock_data['ret'] = stock_data.close.shift(-5) / stock_data.close - 1

    # Compute F1: S&P500 trading above 200-day SMA
    stock_data['f1'] = (stock_data.sp500 >= pd.rolling_mean(stock_data.sp500, 200)).astype(int)

    # Compute F2: Stock is trading above 100-day SMA
    stock_data['f2'] = (stock_data.close >= pd.rolling_mean(stock_data.close, 100)).astype(int)

    # Compute F3: Stock has not gapped more than 15% within lookback period
    stock_data['roll_max'] = pd.rolling_max(stock_data.close, lookback)
    stock_data['roll_min'] = pd.rolling_min(stock_data.close, lookback)
    stock_data['inc'] = (stock_data.roll_max > stock_data.roll_min).astype(int)
    stock_data['max_gap'] = stock_data.inc * (stock_data.roll_max / stock_data.roll_min - 1) + \
        (stock_data.inc - 1) * (stock_data.roll_min / stock_data.roll_max - 1)
    stock_data['f3'] = (stock_data.max_gap < 0.15).astype(int)
    
    # Compute ACM
    stock_data['acm'] = stock_data.cm * stock_data.f1 * stock_data.f2 * stock_data.f3
    
    # Remove NaNs
    stock_data.dropna(axis = 0, inplace = True)
    
    # Calculate correlation between returns and ACM, R2, and slope
    corr_acm = stock_data[['ret', 'acm']].corr().iloc[0,1]
    corr_r2 = stock_data[['ret', 'r2']].corr().iloc[0,1]
    corr_slope = stock_data[['ret', 'slope']].corr().iloc[0,1]
    
    # Regression
    temp_lm = LinearRegression(n_jobs = 3)
    temp_lm.fit(stock_data[['acm', 'r2', 'slope']], stock_data.ret)
    
    # Extract coefficients
    coef_acm = temp_lm.coef_[0]
    coef_r2 = temp_lm.coef_[1]
    coef_slope = temp_lm.coef_[2]
    reg_r2 = temp_lm.score(stock_data[['acm', 'r2', 'slope']], stock_data.ret)
    
    # Configure output
    output = (corr_acm, corr_r2, corr_slope, reg_r2, coef_acm, coef_r2, coef_slope)
    
    # Output
    return output
```


```python
# # Reload data
# alldata = pd.read_csv('sp500_close.csv')
# sp500_index = pd.read_csv('sp500_index.csv')

# # Add days
# alldata['days'] = np.arange(alldata.shape[0])

# # Add S&P500
# alldata['sp500'] = sp500_index.Close

# # Initialise results list
# all_pred = []

# # Configure parameters
# LOOKBACK = 60

# # Subset symbols
# symbol_subset = [x for x in list(sp500.Symbol) if x in list(alldata.columns)]

# # Loop through stocks
# for i in np.arange(0, len(symbol_subset)):
    
#     # Update
#     print('Computing statistics for ' + str(symbol_subset[i]) + ' ' + str(i) + ' of ' + str(len(symbol_subset)) + '...', end = '', flush = True)
    
#     # Compute thresholds
#     temp_thresh = compute_acm(symbol_subset[i], LOOKBACK, alldata)
    
#     # Append data
#     all_pred.append(temp_thresh)
    
#     # Print results
#     print('Done!')
```

    Computing statistics for MCO 312 of 496...Done!
    Computing statistics for MS 313 of 496...Done!
    Computing statistics for MSI 314 of 496...Done!
    Computing statistics for MYL 315 of 496...Done!
    Computing statistics for NDAQ 316 of 496...Done!
    Computing statistics for NOV 317 of 496...Done!
    Computing statistics for NAVI 318 of 496...Done!
    Computing statistics for NKTR 319 of 496...Done!
    Computing statistics for NTAP 320 of 496...Done!
    Computing statistics for NFLX 321 of 496...Done!
    Computing statistics for NWL 322 of 496...Done!
    Computing statistics for NFX 323 of 496...Done!
    Computing statistics for NEM 324 of 496...Done!
    Computing statistics for NWSA 325 of 496...Done!
    Computing statistics for NWS 326 of 496...Done!
    Computing statistics for NEE 327 of 496...Done!
    Computing statistics for NLSN 328 of 496...Done!
    Computing statistics for NKE 329 of 496...Done!
    Computing statistics for NI 330 of 496...Done!
    Computing statistics for NBL 331 of 496...Done!
    Computing statistics for JWN 332 of 496...Done!
    Computing statistics for NSC 333 of 496...Done!
    Computing statistics for NTRS 334 of 496...Done!
    Computing statistics for NOC 335 of 496...Done!
    Computing statistics for NCLH 336 of 496...Done!
    Computing statistics for NRG 337 of 496...Done!
    Computing statistics for NUE 338 of 496...Done!
    Computing statistics for NVDA 339 of 496...Done!
    Computing statistics for ORLY 340 of 496...Done!
    Computing statistics for OXY 341 of 496...Done!
    Computing statistics for OMC 342 of 496...Done!
    Computing statistics for OKE 343 of 496...Done!
    Computing statistics for ORCL 344 of 496...Done!
    Computing statistics for PCAR 345 of 496...Done!
    Computing statistics for PKG 346 of 496...Done!
    Computing statistics for PH 347 of 496...Done!
    Computing statistics for PAYX 348 of 496...Done!
    Computing statistics for PYPL 349 of 496...Done!
    Computing statistics for PNR 350 of 496...Done!
    Computing statistics for PBCT 351 of 496...Done!
    Computing statistics for PEP 352 of 496...Done!
    Computing statistics for PKI 353 of 496...Done!
    Computing statistics for PRGO 354 of 496...Done!
    Computing statistics for PFE 355 of 496...Done!
    Computing statistics for PCG 356 of 496...Done!
    Computing statistics for PM 357 of 496...Done!
    Computing statistics for PSX 358 of 496...Done!
    Computing statistics for PNW 359 of 496...Done!
    Computing statistics for PXD 360 of 496...Done!
    Computing statistics for PNC 361 of 496...Done!
    Computing statistics for RL 362 of 496...Done!
    Computing statistics for PPG 363 of 496...Done!
    Computing statistics for PPL 364 of 496...Done!
    Computing statistics for PX 365 of 496...Done!
    Computing statistics for PFG 366 of 496...Done!
    Computing statistics for PG 367 of 496...Done!
    Computing statistics for PGR 368 of 496...Done!
    Computing statistics for PLD 369 of 496...Done!
    Computing statistics for PRU 370 of 496...Done!
    Computing statistics for PEG 371 of 496...Done!
    Computing statistics for PSA 372 of 496...Done!
    Computing statistics for PHM 373 of 496...Done!
    Computing statistics for PVH 374 of 496...Done!
    Computing statistics for QRVO 375 of 496...Done!
    Computing statistics for QCOM 376 of 496...Done!
    Computing statistics for PWR 377 of 496...Done!
    Computing statistics for DGX 378 of 496...Done!
    Computing statistics for RRC 379 of 496...Done!
    Computing statistics for RJF 380 of 496...Done!
    Computing statistics for RTN 381 of 496...Done!
    Computing statistics for O 382 of 496...Done!
    Computing statistics for RHT 383 of 496...Done!
    Computing statistics for REG 384 of 496...Done!
    Computing statistics for REGN 385 of 496...Done!
    Computing statistics for RF 386 of 496...Done!
    Computing statistics for RSG 387 of 496...Done!
    Computing statistics for RMD 388 of 496...Done!
    Computing statistics for RHI 389 of 496...Done!
    Computing statistics for ROK 390 of 496...Done!
    Computing statistics for COL 391 of 496...Done!
    Computing statistics for ROP 392 of 496...Done!
    Computing statistics for ROST 393 of 496...Done!
    Computing statistics for RCL 394 of 496...Done!
    Computing statistics for SPGI 395 of 496...Done!
    Computing statistics for CRM 396 of 496...Done!
    Computing statistics for SBAC 397 of 496...Done!
    Computing statistics for SCG 398 of 496...Done!
    Computing statistics for SLB 399 of 496...Done!
    Computing statistics for STX 400 of 496...Done!
    Computing statistics for SEE 401 of 496...Done!
    Computing statistics for SRE 402 of 496...Done!
    Computing statistics for SHW 403 of 496...Done!
    Computing statistics for SPG 404 of 496...Done!
    Computing statistics for SWKS 405 of 496...Done!
    Computing statistics for SLG 406 of 496...Done!
    Computing statistics for SNA 407 of 496...Done!
    Computing statistics for SO 408 of 496...Done!
    Computing statistics for LUV 409 of 496...Done!
    Computing statistics for SWK 410 of 496...Done!
    Computing statistics for SBUX 411 of 496...Done!
    Computing statistics for STT 412 of 496...Done!
    Computing statistics for SRCL 413 of 496...Done!
    Computing statistics for SYK 414 of 496...Done!
    Computing statistics for STI 415 of 496...Done!
    Computing statistics for SIVB 416 of 496...Done!
    Computing statistics for SYMC 417 of 496...Done!
    Computing statistics for SYF 418 of 496...Done!
    Computing statistics for SNPS 419 of 496...Done!
    Computing statistics for SYY 420 of 496...Done!
    Computing statistics for TROW 421 of 496...Done!
    Computing statistics for TTWO 422 of 496...Done!
    Computing statistics for TPR 423 of 496...Done!
    Computing statistics for TGT 424 of 496...Done!
    Computing statistics for TEL 425 of 496...Done!
    Computing statistics for FTI 426 of 496...Done!
    Computing statistics for TXN 427 of 496...Done!
    Computing statistics for TXT 428 of 496...Done!
    Computing statistics for BK 429 of 496...Done!
    Computing statistics for CLX 430 of 496...Done!
    Computing statistics for COO 431 of 496...Done!
    Computing statistics for HSY 432 of 496...Done!
    Computing statistics for MOS 433 of 496...Done!
    Computing statistics for TRV 434 of 496...Done!
    Computing statistics for DIS 435 of 496...Done!
    Computing statistics for TMO 436 of 496...Done!
    Computing statistics for TIF 437 of 496...Done!
    Computing statistics for TJX 438 of 496...Done!
    Computing statistics for TMK 439 of 496...Done!
    Computing statistics for TSS 440 of 496...Done!
    Computing statistics for TSCO 441 of 496...Done!
    Computing statistics for TDG 442 of 496...Done!
    Computing statistics for TRIP 443 of 496...Done!
    Computing statistics for FOXA 444 of 496...Done!
    Computing statistics for FOX 445 of 496...Done!
    Computing statistics for TSN 446 of 496...Done!
    Computing statistics for USB 447 of 496...Done!
    Computing statistics for UDR 448 of 496...Done!
    Computing statistics for ULTA 449 of 496...Done!
    Computing statistics for UAA 450 of 496...Done!
    Computing statistics for UA 451 of 496...Done!
    Computing statistics for UNP 452 of 496...Done!
    Computing statistics for UAL 453 of 496...Done!
    Computing statistics for UNH 454 of 496...Done!
    Computing statistics for UPS 455 of 496...Done!
    Computing statistics for URI 456 of 496...Done!
    Computing statistics for UTX 457 of 496...Done!
    Computing statistics for UHS 458 of 496...Done!
    Computing statistics for UNM 459 of 496...Done!
    Computing statistics for VFC 460 of 496...Done!
    Computing statistics for VLO 461 of 496...Done!
    Computing statistics for VAR 462 of 496...Done!
    Computing statistics for VTR 463 of 496...Done!
    Computing statistics for VRSN 464 of 496...Done!
    Computing statistics for VRSK 465 of 496...Done!
    Computing statistics for VZ 466 of 496...Done!
    Computing statistics for VRTX 467 of 496...Done!
    Computing statistics for VIAB 468 of 496...Done!
    Computing statistics for V 469 of 496...Done!
    Computing statistics for VNO 470 of 496...Done!
    Computing statistics for VMC 471 of 496...Done!
    Computing statistics for WMT 472 of 496...Done!
    Computing statistics for WBA 473 of 496...Done!
    Computing statistics for WM 474 of 496...Done!
    Computing statistics for WAT 475 of 496...Done!
    Computing statistics for WEC 476 of 496...Done!
    Computing statistics for WFC 477 of 496...Done!
    Computing statistics for WELL 478 of 496...Done!
    Computing statistics for WDC 479 of 496...Done!
    Computing statistics for WU 480 of 496...Done!
    Computing statistics for WRK 481 of 496...Done!
    Computing statistics for WY 482 of 496...Done!
    Computing statistics for WHR 483 of 496...Done!
    Computing statistics for WMB 484 of 496...Done!
    Computing statistics for WLTW 485 of 496...Done!
    Computing statistics for WYNN 486 of 496...Done!
    Computing statistics for XEL 487 of 496...Done!
    Computing statistics for XRX 488 of 496...Done!
    Computing statistics for XLNX 489 of 496...Done!
    Computing statistics for XL 490 of 496...Done!
    Computing statistics for XYL 491 of 496...Done!
    Computing statistics for YUM 492 of 496...Done!
    Computing statistics for ZBH 493 of 496...Done!
    Computing statistics for ZION 494 of 496...Done!
    Computing statistics for ZTS 495 of 496...Done!
    


```python
# # Convert to data frame
# corr_data = pd.DataFrame(all_pred, columns = [
#     'corr_acm', 'corr_r2', 'corr_slope', 'reg_r2', 'coef_acm', 'coef_r2', 'coef_slope'
# ])

# # Save
# corr_data.to_csv('corr_data.csv', index = False)
```

From the graphs below, we see that on average, there was **no correlation between 5-day forward returns and ACM (which reflects the Clenow Momentum strategy) or the components of ACM: the regression R2 and slope**. The maximum size of the correlation coefficient in either direction (positive or negative) was at most 0.20, which is still relatively small.


```python
# Read
corr_data = pd.read_csv('corr_data.csv')

# Plot
plt.figure(figsize = (10, 8))
corr_data.corr_acm.plot.hist(bins = 100, color = '#133056', alpha = 0.8)
plt.title('Correlation: Returns vs. ACM', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.show()

# Plot
plt.figure(figsize = (10, 8))
corr_data.corr_r2.plot.hist(bins = 100, color = '#6fceb0', alpha = 0.8)
plt.title('Correlation: Returns vs. R2', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.show()

# Plot
plt.figure(figsize = (10, 8))
corr_data.corr_slope.plot.hist(bins = 100, color = '#b1ceeb', alpha = 0.8)
plt.title('Correlation: Returns vs. Slope Coefficient', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.show()
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)


Thus, we may conclude that statistically, there is no strong relationship between ACM or its components with forward 5-day returns.

# Testing Clenow Momentum II: Trading Returns
We run a trading simulation using a simplified version of Clenow's approach to test if the strategy beats the buy-and-hold benchmark. The simplified strategy involves the following:  
  
1. Maintaining an equally-weighted portfolio that is reset every week
2. Choosing the top 10 stocks by ACM instead of the top *N* stocks by standard deviation up to a fixed amount in principal
  
Note that for the CM-based strategy, we only run one simulation because the approach applies to a universe of stocks (in this case the S&P 500). If we wanted confirmation on different universes of stocks, we could easily run the same simulation on the S&P Mid-Cap 400 or the Russell 1000.

### Function for Executing Trades


```python
def trade_acm(stk, lookback, full_df):
    
    # Pull data from main data frame
    stock_data = full_df[['days', 'sp500']].copy()
    stock_data['close'] = full_df[stk].copy()
    
    # Run rolling regression with 30-day windows
    temp_lm = pd.stats.ols.MovingOLS(
        y=np.log(stock_data.close), x=stock_data[['days']],
        window_type='rolling', window=lookback, intercept=True
    )
    
    # Compute Clenow Momentum
    stock_data['slope'] = ((temp_lm.beta.days + 1) ** 250 - 1) * 100
    stock_data['r2'] = temp_lm.r2 * 100
    stock_data['cm'] = stock_data.slope * stock_data.r2

    # Compute forward returns
    stock_data['ret'] = stock_data.close.shift(-5) / stock_data.close - 1

    # Compute F1: S&P500 trading above 200-day SMA
    stock_data['f1'] = (stock_data.sp500 >= pd.rolling_mean(stock_data.sp500, 200)).astype(int)

    # Compute F2: Stock is trading above 100-day SMA
    stock_data['f2'] = (stock_data.close >= pd.rolling_mean(stock_data.close, 100)).astype(int)

    # Compute F3: Stock has not gapped more than 15% within lookback period
    stock_data['roll_max'] = pd.rolling_max(stock_data.close, lookback)
    stock_data['roll_min'] = pd.rolling_min(stock_data.close, lookback)
    stock_data['max_min'] = abs(pd.rolling_max(stock_data.close, lookback) / pd.rolling_min(stock_data.close, lookback)) - 1
    stock_data['min_max'] = abs(pd.rolling_min(stock_data.close, lookback) / pd.rolling_max(stock_data.close, lookback)) - 1
    stock_data['max_gap'] = stock_data[['max_min', 'min_max']].max(axis = 1)
    stock_data['f3'] = (abs(stock_data.max_gap) < 0.15).astype(int)
    
    # Compute ACM
    stock_data['acm'] = stock_data.cm * stock_data.f1 * stock_data.f2 * stock_data.f3
    
    # Output
    return (stock_data[['ret']], stock_data.acm)
```


```python
# # Reload data
# alldata = pd.read_csv('sp500_close.csv')
# sp500_index = pd.read_csv('sp500_index.csv')

# # Add days
# alldata['days'] = np.arange(alldata.shape[0])

# # Add S&P500
# alldata['sp500'] = sp500_index.Close

# # Initialise results list
# all_ret = pd.DataFrame()
# all_acm = pd.DataFrame()

# # Configure parameters
# LOOKBACK = 60

# # Subset symbols
# symbol_subset = [x for x in list(sp500.Symbol) if x in list(alldata.columns)]

# # Loop through stocks
# for i in np.arange(0, len(symbol_subset)):
    
#     # Update
#     print('Computing ACM for ' + str(symbol_subset[i]) + ' - ' + str(i) + ' of ' + str(len(symbol_subset)) + '...', end = '', flush = True)
    
#     # Compute ACM
#     temp_acm = trade_acm(symbol_subset[i], LOOKBACK, alldata)
    
#     # Append data
#     all_ret[symbol_subset[i]] = temp_acm[0]
#     all_acm[symbol_subset[i]] = temp_acm[1]
    
#     # Print results
#     print('Done!')
```

    Computing ACM for MMM - 0 of 496...Done!
    Computing ACM for AOS - 1 of 496...Done!
    Computing ACM for ABT - 2 of 496...Done!
    Computing ACM for ABBV - 3 of 496...Done!
    Computing ACM for ACN - 4 of 496...Done!
    Computing ACM for ATVI - 5 of 496...Done!
    Computing ACM for AYI - 6 of 496...Done!
    Computing ACM for ADBE - 7 of 496...Done!
    Computing ACM for AAP - 8 of 496...Done!
    Computing ACM for AMD - 9 of 496...Done!
    Computing ACM for AES - 10 of 496...Done!
    Computing ACM for AET - 11 of 496...Done!
    Computing ACM for AMG - 12 of 496...Done!
    Computing ACM for AFL - 13 of 496...Done!
    Computing ACM for A - 14 of 496...Done!
    Computing ACM for APD - 15 of 496...Done!
    Computing ACM for AKAM - 16 of 496...Done!
    Computing ACM for ALK - 17 of 496...Done!
    Computing ACM for ALB - 18 of 496...Done!
    Computing ACM for ARE - 19 of 496...Done!
    Computing ACM for ALXN - 20 of 496...Done!
    Computing ACM for ALGN - 21 of 496...Done!
    Computing ACM for ALLE - 22 of 496...Done!
    Computing ACM for AGN - 23 of 496...Done!
    Computing ACM for ADS - 24 of 496...Done!
    Computing ACM for LNT - 25 of 496...Done!
    Computing ACM for ALL - 26 of 496...Done!
    Computing ACM for GOOGL - 27 of 496...Done!
    Computing ACM for GOOG - 28 of 496...Done!
    Computing ACM for MO - 29 of 496...Done!
    Computing ACM for AMZN - 30 of 496...Done!
    Computing ACM for AEE - 31 of 496...Done!
    Computing ACM for AAL - 32 of 496...Done!
    Computing ACM for AEP - 33 of 496...Done!
    Computing ACM for AXP - 34 of 496...Done!
    Computing ACM for AIG - 35 of 496...Done!
    Computing ACM for AMT - 36 of 496...Done!
    Computing ACM for AWK - 37 of 496...Done!
    Computing ACM for AMP - 38 of 496...Done!
    Computing ACM for ABC - 39 of 496...Done!
    Computing ACM for AME - 40 of 496...Done!
    Computing ACM for AMGN - 41 of 496...Done!
    Computing ACM for APH - 42 of 496...Done!
    Computing ACM for APC - 43 of 496...Done!
    Computing ACM for ADI - 44 of 496...Done!
    Computing ACM for ANDV - 45 of 496...Done!
    Computing ACM for ANSS - 46 of 496...Done!
    Computing ACM for ANTM - 47 of 496...Done!
    Computing ACM for AON - 48 of 496...Done!
    Computing ACM for APA - 49 of 496...Done!
    Computing ACM for AIV - 50 of 496...Done!
    Computing ACM for AAPL - 51 of 496...Done!
    Computing ACM for AMAT - 52 of 496...Done!
    Computing ACM for APTV - 53 of 496...Done!
    Computing ACM for ADM - 54 of 496...Done!
    Computing ACM for ARNC - 55 of 496...Done!
    Computing ACM for AJG - 56 of 496...Done!
    Computing ACM for AIZ - 57 of 496...Done!
    Computing ACM for T - 58 of 496...Done!
    Computing ACM for ADSK - 59 of 496...Done!
    Computing ACM for ADP - 60 of 496...Done!
    Computing ACM for AZO - 61 of 496...Done!
    Computing ACM for AVB - 62 of 496...Done!
    Computing ACM for AVY - 63 of 496...Done!
    Computing ACM for BHGE - 64 of 496...Done!
    Computing ACM for BLL - 65 of 496...Done!
    Computing ACM for BAC - 66 of 496...Done!
    Computing ACM for BAX - 67 of 496...Done!
    Computing ACM for BBT - 68 of 496...Done!
    Computing ACM for BDX - 69 of 496...Done!
    Computing ACM for BBY - 70 of 496...Done!
    Computing ACM for BIIB - 71 of 496...Done!
    Computing ACM for BLK - 72 of 496...Done!
    Computing ACM for HRB - 73 of 496...Done!
    Computing ACM for BA - 74 of 496...Done!
    Computing ACM for BKNG - 75 of 496...Done!
    Computing ACM for BWA - 76 of 496...Done!
    Computing ACM for BXP - 77 of 496...Done!
    Computing ACM for BSX - 78 of 496...Done!
    Computing ACM for BHF - 79 of 496...Done!
    Computing ACM for BMY - 80 of 496...Done!
    Computing ACM for AVGO - 81 of 496...Done!
    Computing ACM for CHRW - 82 of 496...Done!
    Computing ACM for CA - 83 of 496...Done!
    Computing ACM for COG - 84 of 496...Done!
    Computing ACM for CDNS - 85 of 496...Done!
    Computing ACM for CPB - 86 of 496...Done!
    Computing ACM for COF - 87 of 496...Done!
    Computing ACM for CAH - 88 of 496...Done!
    Computing ACM for KMX - 89 of 496...Done!
    Computing ACM for CCL - 90 of 496...Done!
    Computing ACM for CAT - 91 of 496...Done!
    Computing ACM for CBOE - 92 of 496...Done!
    Computing ACM for CBRE - 93 of 496...Done!
    Computing ACM for CBS - 94 of 496...Done!
    Computing ACM for CELG - 95 of 496...Done!
    Computing ACM for CNC - 96 of 496...Done!
    Computing ACM for CNP - 97 of 496...Done!
    Computing ACM for CTL - 98 of 496...Done!
    Computing ACM for CERN - 99 of 496...Done!
    Computing ACM for CF - 100 of 496...Done!
    Computing ACM for SCHW - 101 of 496...Done!
    Computing ACM for CHTR - 102 of 496...Done!
    Computing ACM for CVX - 103 of 496...Done!
    Computing ACM for CMG - 104 of 496...Done!
    Computing ACM for CB - 105 of 496...Done!
    Computing ACM for CHD - 106 of 496...Done!
    Computing ACM for CI - 107 of 496...Done!
    Computing ACM for XEC - 108 of 496...Done!
    Computing ACM for CINF - 109 of 496...Done!
    Computing ACM for CTAS - 110 of 496...Done!
    Computing ACM for CSCO - 111 of 496...Done!
    Computing ACM for C - 112 of 496...Done!
    Computing ACM for CFG - 113 of 496...Done!
    Computing ACM for CTXS - 114 of 496...Done!
    Computing ACM for CME - 115 of 496...Done!
    Computing ACM for CMS - 116 of 496...Done!
    Computing ACM for KO - 117 of 496...Done!
    Computing ACM for CTSH - 118 of 496...Done!
    Computing ACM for CL - 119 of 496...Done!
    Computing ACM for CMCSA - 120 of 496...Done!
    Computing ACM for CMA - 121 of 496...Done!
    Computing ACM for CAG - 122 of 496...Done!
    Computing ACM for CXO - 123 of 496...Done!
    Computing ACM for COP - 124 of 496...Done!
    Computing ACM for ED - 125 of 496...Done!
    Computing ACM for STZ - 126 of 496...Done!
    Computing ACM for GLW - 127 of 496...Done!
    Computing ACM for COST - 128 of 496...Done!
    Computing ACM for COTY - 129 of 496...Done!
    Computing ACM for CCI - 130 of 496...Done!
    Computing ACM for CSX - 131 of 496...Done!
    Computing ACM for CMI - 132 of 496...Done!
    Computing ACM for CVS - 133 of 496...Done!
    Computing ACM for DHI - 134 of 496...Done!
    Computing ACM for DHR - 135 of 496...Done!
    Computing ACM for DRI - 136 of 496...Done!
    Computing ACM for DVA - 137 of 496...Done!
    Computing ACM for DE - 138 of 496...Done!
    Computing ACM for DAL - 139 of 496...Done!
    Computing ACM for XRAY - 140 of 496...Done!
    Computing ACM for DVN - 141 of 496...Done!
    Computing ACM for DLR - 142 of 496...Done!
    Computing ACM for DFS - 143 of 496...Done!
    Computing ACM for DISCA - 144 of 496...Done!
    Computing ACM for DISCK - 145 of 496...Done!
    Computing ACM for DISH - 146 of 496...Done!
    Computing ACM for DG - 147 of 496...Done!
    Computing ACM for DLTR - 148 of 496...Done!
    Computing ACM for D - 149 of 496...Done!
    Computing ACM for DOV - 150 of 496...Done!
    Computing ACM for DWDP - 151 of 496...Done!
    Computing ACM for DTE - 152 of 496...Done!
    Computing ACM for DUK - 153 of 496...Done!
    Computing ACM for DRE - 154 of 496...Done!
    Computing ACM for DXC - 155 of 496...Done!
    Computing ACM for ETFC - 156 of 496...Done!
    Computing ACM for EMN - 157 of 496...Done!
    Computing ACM for ETN - 158 of 496...Done!
    Computing ACM for EBAY - 159 of 496...Done!
    Computing ACM for ECL - 160 of 496...Done!
    Computing ACM for EIX - 161 of 496...Done!
    Computing ACM for EW - 162 of 496...Done!
    Computing ACM for EA - 163 of 496...Done!
    Computing ACM for EMR - 164 of 496...Done!
    Computing ACM for ETR - 165 of 496...Done!
    Computing ACM for EVHC - 166 of 496...Done!
    Computing ACM for EOG - 167 of 496...Done!
    Computing ACM for EQT - 168 of 496...Done!
    Computing ACM for EFX - 169 of 496...Done!
    Computing ACM for EQIX - 170 of 496...Done!
    Computing ACM for EQR - 171 of 496...Done!
    Computing ACM for ESS - 172 of 496...Done!
    Computing ACM for EL - 173 of 496...Done!
    Computing ACM for RE - 174 of 496...Done!
    Computing ACM for ES - 175 of 496...Done!
    Computing ACM for EXC - 176 of 496...Done!
    Computing ACM for EXPE - 177 of 496...Done!
    Computing ACM for EXPD - 178 of 496...Done!
    Computing ACM for ESRX - 179 of 496...Done!
    Computing ACM for EXR - 180 of 496...Done!
    Computing ACM for XOM - 181 of 496...Done!
    Computing ACM for FFIV - 182 of 496...Done!
    Computing ACM for FB - 183 of 496...Done!
    Computing ACM for FAST - 184 of 496...Done!
    Computing ACM for FRT - 185 of 496...Done!
    Computing ACM for FDX - 186 of 496...Done!
    Computing ACM for FIS - 187 of 496...Done!
    Computing ACM for FITB - 188 of 496...Done!
    Computing ACM for FE - 189 of 496...Done!
    Computing ACM for FISV - 190 of 496...Done!
    Computing ACM for FLIR - 191 of 496...Done!
    Computing ACM for FLS - 192 of 496...Done!
    Computing ACM for FLR - 193 of 496...Done!
    Computing ACM for FMC - 194 of 496...Done!
    Computing ACM for FL - 195 of 496...Done!
    Computing ACM for F - 196 of 496...Done!
    Computing ACM for FTV - 197 of 496...Done!
    Computing ACM for FBHS - 198 of 496...Done!
    Computing ACM for BEN - 199 of 496...Done!
    Computing ACM for FCX - 200 of 496...Done!
    Computing ACM for GPS - 201 of 496...Done!
    Computing ACM for GRMN - 202 of 496...Done!
    Computing ACM for IT - 203 of 496...Done!
    Computing ACM for GD - 204 of 496...Done!
    Computing ACM for GE - 205 of 496...Done!
    Computing ACM for GIS - 206 of 496...Done!
    Computing ACM for GM - 207 of 496...Done!
    Computing ACM for GPC - 208 of 496...Done!
    Computing ACM for GILD - 209 of 496...Done!
    Computing ACM for GPN - 210 of 496...Done!
    Computing ACM for GS - 211 of 496...Done!
    Computing ACM for GT - 212 of 496...Done!
    Computing ACM for GWW - 213 of 496...Done!
    Computing ACM for HAL - 214 of 496...Done!
    Computing ACM for HBI - 215 of 496...Done!
    Computing ACM for HOG - 216 of 496...Done!
    Computing ACM for HRS - 217 of 496...Done!
    Computing ACM for HIG - 218 of 496...Done!
    Computing ACM for HAS - 219 of 496...Done!
    Computing ACM for HCA - 220 of 496...Done!
    Computing ACM for HCP - 221 of 496...Done!
    Computing ACM for HP - 222 of 496...Done!
    Computing ACM for HSIC - 223 of 496...Done!
    Computing ACM for HES - 224 of 496...Done!
    Computing ACM for HPE - 225 of 496...Done!
    Computing ACM for HLT - 226 of 496...Done!
    Computing ACM for HOLX - 227 of 496...Done!
    Computing ACM for HD - 228 of 496...Done!
    Computing ACM for HON - 229 of 496...Done!
    Computing ACM for HRL - 230 of 496...Done!
    Computing ACM for HST - 231 of 496...Done!
    Computing ACM for HPQ - 232 of 496...Done!
    Computing ACM for HUM - 233 of 496...Done!
    Computing ACM for HBAN - 234 of 496...Done!
    Computing ACM for HII - 235 of 496...Done!
    Computing ACM for IDXX - 236 of 496...Done!
    Computing ACM for INFO - 237 of 496...Done!
    Computing ACM for ITW - 238 of 496...Done!
    Computing ACM for ILMN - 239 of 496...Done!
    Computing ACM for INCY - 240 of 496...Done!
    Computing ACM for IR - 241 of 496...Done!
    Computing ACM for INTC - 242 of 496...Done!
    Computing ACM for ICE - 243 of 496...Done!
    Computing ACM for IBM - 244 of 496...Done!
    Computing ACM for IP - 245 of 496...Done!
    Computing ACM for IPG - 246 of 496...Done!
    Computing ACM for IFF - 247 of 496...Done!
    Computing ACM for INTU - 248 of 496...Done!
    Computing ACM for ISRG - 249 of 496...Done!
    Computing ACM for IVZ - 250 of 496...Done!
    Computing ACM for IPGP - 251 of 496...Done!
    Computing ACM for IQV - 252 of 496...Done!
    Computing ACM for IRM - 253 of 496...Done!
    Computing ACM for JBHT - 254 of 496...Done!
    Computing ACM for JEC - 255 of 496...Done!
    Computing ACM for SJM - 256 of 496...Done!
    Computing ACM for JNJ - 257 of 496...Done!
    Computing ACM for JCI - 258 of 496...Done!
    Computing ACM for JPM - 259 of 496...Done!
    Computing ACM for JNPR - 260 of 496...Done!
    Computing ACM for KSU - 261 of 496...Done!
    Computing ACM for K - 262 of 496...Done!
    Computing ACM for KEY - 263 of 496...Done!
    Computing ACM for KMB - 264 of 496...Done!
    Computing ACM for KIM - 265 of 496...Done!
    Computing ACM for KMI - 266 of 496...Done!
    Computing ACM for KLAC - 267 of 496...Done!
    Computing ACM for KSS - 268 of 496...Done!
    Computing ACM for KHC - 269 of 496...Done!
    Computing ACM for KR - 270 of 496...Done!
    Computing ACM for LB - 271 of 496...Done!
    Computing ACM for LLL - 272 of 496...Done!
    Computing ACM for LH - 273 of 496...Done!
    Computing ACM for LRCX - 274 of 496...Done!
    Computing ACM for LEG - 275 of 496...Done!
    Computing ACM for LEN - 276 of 496...Done!
    Computing ACM for LLY - 277 of 496...Done!
    Computing ACM for LNC - 278 of 496...Done!
    Computing ACM for LKQ - 279 of 496...Done!
    Computing ACM for LMT - 280 of 496...Done!
    Computing ACM for L - 281 of 496...Done!
    Computing ACM for LOW - 282 of 496...Done!
    Computing ACM for LYB - 283 of 496...Done!
    Computing ACM for MTB - 284 of 496...Done!
    Computing ACM for MAC - 285 of 496...Done!
    Computing ACM for M - 286 of 496...Done!
    Computing ACM for MRO - 287 of 496...Done!
    Computing ACM for MPC - 288 of 496...Done!
    Computing ACM for MAR - 289 of 496...Done!
    Computing ACM for MMC - 290 of 496...Done!
    Computing ACM for MLM - 291 of 496...Done!
    Computing ACM for MAS - 292 of 496...Done!
    Computing ACM for MA - 293 of 496...Done!
    Computing ACM for MAT - 294 of 496...Done!
    Computing ACM for MKC - 295 of 496...Done!
    Computing ACM for MCD - 296 of 496...Done!
    Computing ACM for MCK - 297 of 496...Done!
    Computing ACM for MDT - 298 of 496...Done!
    Computing ACM for MRK - 299 of 496...Done!
    Computing ACM for MET - 300 of 496...Done!
    Computing ACM for MTD - 301 of 496...Done!
    Computing ACM for MGM - 302 of 496...Done!
    Computing ACM for KORS - 303 of 496...Done!
    Computing ACM for MCHP - 304 of 496...Done!
    Computing ACM for MU - 305 of 496...Done!
    Computing ACM for MSFT - 306 of 496...Done!
    Computing ACM for MAA - 307 of 496...Done!
    Computing ACM for MHK - 308 of 496...Done!
    Computing ACM for TAP - 309 of 496...Done!
    Computing ACM for MDLZ - 310 of 496...Done!
    Computing ACM for MNST - 311 of 496...Done!
    Computing ACM for MCO - 312 of 496...Done!
    Computing ACM for MS - 313 of 496...Done!
    Computing ACM for MSI - 314 of 496...Done!
    Computing ACM for MYL - 315 of 496...Done!
    Computing ACM for NDAQ - 316 of 496...Done!
    Computing ACM for NOV - 317 of 496...Done!
    Computing ACM for NAVI - 318 of 496...Done!
    Computing ACM for NKTR - 319 of 496...Done!
    Computing ACM for NTAP - 320 of 496...Done!
    Computing ACM for NFLX - 321 of 496...Done!
    Computing ACM for NWL - 322 of 496...Done!
    Computing ACM for NFX - 323 of 496...Done!
    Computing ACM for NEM - 324 of 496...Done!
    Computing ACM for NWSA - 325 of 496...Done!
    Computing ACM for NWS - 326 of 496...Done!
    Computing ACM for NEE - 327 of 496...Done!
    Computing ACM for NLSN - 328 of 496...Done!
    Computing ACM for NKE - 329 of 496...Done!
    Computing ACM for NI - 330 of 496...Done!
    Computing ACM for NBL - 331 of 496...Done!
    Computing ACM for JWN - 332 of 496...Done!
    Computing ACM for NSC - 333 of 496...Done!
    Computing ACM for NTRS - 334 of 496...Done!
    Computing ACM for NOC - 335 of 496...Done!
    Computing ACM for NCLH - 336 of 496...Done!
    Computing ACM for NRG - 337 of 496...Done!
    Computing ACM for NUE - 338 of 496...Done!
    Computing ACM for NVDA - 339 of 496...Done!
    Computing ACM for ORLY - 340 of 496...Done!
    Computing ACM for OXY - 341 of 496...Done!
    Computing ACM for OMC - 342 of 496...Done!
    Computing ACM for OKE - 343 of 496...Done!
    Computing ACM for ORCL - 344 of 496...Done!
    Computing ACM for PCAR - 345 of 496...Done!
    Computing ACM for PKG - 346 of 496...Done!
    Computing ACM for PH - 347 of 496...Done!
    Computing ACM for PAYX - 348 of 496...Done!
    Computing ACM for PYPL - 349 of 496...Done!
    Computing ACM for PNR - 350 of 496...Done!
    Computing ACM for PBCT - 351 of 496...Done!
    Computing ACM for PEP - 352 of 496...Done!
    Computing ACM for PKI - 353 of 496...Done!
    Computing ACM for PRGO - 354 of 496...Done!
    Computing ACM for PFE - 355 of 496...Done!
    Computing ACM for PCG - 356 of 496...Done!
    Computing ACM for PM - 357 of 496...Done!
    Computing ACM for PSX - 358 of 496...Done!
    Computing ACM for PNW - 359 of 496...Done!
    Computing ACM for PXD - 360 of 496...Done!
    Computing ACM for PNC - 361 of 496...Done!
    Computing ACM for RL - 362 of 496...Done!
    Computing ACM for PPG - 363 of 496...Done!
    Computing ACM for PPL - 364 of 496...Done!
    Computing ACM for PX - 365 of 496...Done!
    Computing ACM for PFG - 366 of 496...Done!
    Computing ACM for PG - 367 of 496...Done!
    Computing ACM for PGR - 368 of 496...Done!
    Computing ACM for PLD - 369 of 496...Done!
    Computing ACM for PRU - 370 of 496...Done!
    Computing ACM for PEG - 371 of 496...Done!
    Computing ACM for PSA - 372 of 496...Done!
    Computing ACM for PHM - 373 of 496...Done!
    Computing ACM for PVH - 374 of 496...Done!
    Computing ACM for QRVO - 375 of 496...Done!
    Computing ACM for QCOM - 376 of 496...Done!
    Computing ACM for PWR - 377 of 496...Done!
    Computing ACM for DGX - 378 of 496...Done!
    Computing ACM for RRC - 379 of 496...Done!
    Computing ACM for RJF - 380 of 496...Done!
    Computing ACM for RTN - 381 of 496...Done!
    Computing ACM for O - 382 of 496...Done!
    Computing ACM for RHT - 383 of 496...Done!
    Computing ACM for REG - 384 of 496...Done!
    Computing ACM for REGN - 385 of 496...Done!
    Computing ACM for RF - 386 of 496...Done!
    Computing ACM for RSG - 387 of 496...Done!
    Computing ACM for RMD - 388 of 496...Done!
    Computing ACM for RHI - 389 of 496...Done!
    Computing ACM for ROK - 390 of 496...Done!
    Computing ACM for COL - 391 of 496...Done!
    Computing ACM for ROP - 392 of 496...Done!
    Computing ACM for ROST - 393 of 496...Done!
    Computing ACM for RCL - 394 of 496...Done!
    Computing ACM for SPGI - 395 of 496...Done!
    Computing ACM for CRM - 396 of 496...Done!
    Computing ACM for SBAC - 397 of 496...Done!
    Computing ACM for SCG - 398 of 496...Done!
    Computing ACM for SLB - 399 of 496...Done!
    Computing ACM for STX - 400 of 496...Done!
    Computing ACM for SEE - 401 of 496...Done!
    Computing ACM for SRE - 402 of 496...Done!
    Computing ACM for SHW - 403 of 496...Done!
    Computing ACM for SPG - 404 of 496...Done!
    Computing ACM for SWKS - 405 of 496...Done!
    Computing ACM for SLG - 406 of 496...Done!
    Computing ACM for SNA - 407 of 496...Done!
    Computing ACM for SO - 408 of 496...Done!
    Computing ACM for LUV - 409 of 496...Done!
    Computing ACM for SWK - 410 of 496...Done!
    Computing ACM for SBUX - 411 of 496...Done!
    Computing ACM for STT - 412 of 496...Done!
    Computing ACM for SRCL - 413 of 496...Done!
    Computing ACM for SYK - 414 of 496...Done!
    Computing ACM for STI - 415 of 496...Done!
    Computing ACM for SIVB - 416 of 496...Done!
    Computing ACM for SYMC - 417 of 496...Done!
    Computing ACM for SYF - 418 of 496...Done!
    Computing ACM for SNPS - 419 of 496...Done!
    Computing ACM for SYY - 420 of 496...Done!
    Computing ACM for TROW - 421 of 496...Done!
    Computing ACM for TTWO - 422 of 496...Done!
    Computing ACM for TPR - 423 of 496...Done!
    Computing ACM for TGT - 424 of 496...Done!
    Computing ACM for TEL - 425 of 496...Done!
    Computing ACM for FTI - 426 of 496...Done!
    Computing ACM for TXN - 427 of 496...Done!
    Computing ACM for TXT - 428 of 496...Done!
    Computing ACM for BK - 429 of 496...Done!
    Computing ACM for CLX - 430 of 496...Done!
    Computing ACM for COO - 431 of 496...Done!
    Computing ACM for HSY - 432 of 496...Done!
    Computing ACM for MOS - 433 of 496...Done!
    Computing ACM for TRV - 434 of 496...Done!
    Computing ACM for DIS - 435 of 496...Done!
    Computing ACM for TMO - 436 of 496...Done!
    Computing ACM for TIF - 437 of 496...Done!
    Computing ACM for TJX - 438 of 496...Done!
    Computing ACM for TMK - 439 of 496...Done!
    Computing ACM for TSS - 440 of 496...Done!
    Computing ACM for TSCO - 441 of 496...Done!
    Computing ACM for TDG - 442 of 496...Done!
    Computing ACM for TRIP - 443 of 496...Done!
    Computing ACM for FOXA - 444 of 496...Done!
    Computing ACM for FOX - 445 of 496...Done!
    Computing ACM for TSN - 446 of 496...Done!
    Computing ACM for USB - 447 of 496...Done!
    Computing ACM for UDR - 448 of 496...Done!
    Computing ACM for ULTA - 449 of 496...Done!
    Computing ACM for UAA - 450 of 496...Done!
    Computing ACM for UA - 451 of 496...Done!
    Computing ACM for UNP - 452 of 496...Done!
    Computing ACM for UAL - 453 of 496...Done!
    Computing ACM for UNH - 454 of 496...Done!
    Computing ACM for UPS - 455 of 496...Done!
    Computing ACM for URI - 456 of 496...Done!
    Computing ACM for UTX - 457 of 496...Done!
    Computing ACM for UHS - 458 of 496...Done!
    Computing ACM for UNM - 459 of 496...Done!
    Computing ACM for VFC - 460 of 496...Done!
    Computing ACM for VLO - 461 of 496...Done!
    Computing ACM for VAR - 462 of 496...Done!
    Computing ACM for VTR - 463 of 496...Done!
    Computing ACM for VRSN - 464 of 496...Done!
    Computing ACM for VRSK - 465 of 496...Done!
    Computing ACM for VZ - 466 of 496...Done!
    Computing ACM for VRTX - 467 of 496...Done!
    Computing ACM for VIAB - 468 of 496...Done!
    Computing ACM for V - 469 of 496...Done!
    Computing ACM for VNO - 470 of 496...Done!
    Computing ACM for VMC - 471 of 496...Done!
    Computing ACM for WMT - 472 of 496...Done!
    Computing ACM for WBA - 473 of 496...Done!
    Computing ACM for WM - 474 of 496...Done!
    Computing ACM for WAT - 475 of 496...Done!
    Computing ACM for WEC - 476 of 496...Done!
    Computing ACM for WFC - 477 of 496...Done!
    Computing ACM for WELL - 478 of 496...Done!
    Computing ACM for WDC - 479 of 496...Done!
    Computing ACM for WU - 480 of 496...Done!
    Computing ACM for WRK - 481 of 496...Done!
    Computing ACM for WY - 482 of 496...Done!
    Computing ACM for WHR - 483 of 496...Done!
    Computing ACM for WMB - 484 of 496...Done!
    Computing ACM for WLTW - 485 of 496...Done!
    Computing ACM for WYNN - 486 of 496...Done!
    Computing ACM for XEL - 487 of 496...Done!
    Computing ACM for XRX - 488 of 496...Done!
    Computing ACM for XLNX - 489 of 496...Done!
    Computing ACM for XL - 490 of 496...Done!
    Computing ACM for XYL - 491 of 496...Done!
    Computing ACM for YUM - 492 of 496...Done!
    Computing ACM for ZBH - 493 of 496...Done!
    Computing ACM for ZION - 494 of 496...Done!
    Computing ACM for ZTS - 495 of 496...Done!
    

### Compute Returns


```python
# Save data
# all_ret.to_csv('ret_data.csv', index = False)
# all_acm.to_csv('acm_data.csv', index = False)

# Load data
all_ret = pd.read_csv('ret_data.csv')
all_acm = pd.read_csv('acm_data.csv')

# Truncate data by availability: at least 150 stocks to choose from
avail_acm = (~all_acm.isnull()).sum(axis = 1)
all_ret = all_ret.iloc[364:]
all_acm = all_acm.iloc[364:]

# Select data for every 5 days
all_ret = all_ret.iloc[::5,]
all_acm = all_acm.iloc[::5,]

# Replace NaNs with zeroes
all_acm[all_acm.isnull()] = 0

# Rank by row
rank_acm = all_acm.rank(axis = 1)

# Check for available rankings (at least 1 to 8)
rank_avail = rank_acm.apply(lambda x: (x <= 10).sum(), axis = 1) < 8

# Convert all rows with unavailable rankings to zero
rank_acm[rank_avail] = 0

# Convert all ranked stocks (1 to 10) to 1
rank_acm[(rank_acm > 0) & (rank_acm <= 10)] = 1

# Convert all values above 10 to zero
rank_acm[rank_acm > 10] = 0

# Get ranks available
n_stocks = rank_acm.sum(axis = 1)

# Convert zero values to 1
n_stocks[n_stocks == 0] = 1

# Divide returns by portfolio weight (equal) and add 1
all_ret = all_ret.divide(n_stocks, axis = 0)

# Convert missing values to zero
all_ret[all_ret.isnull()] = 0

# Multiply matrices
all_profits = all_ret * rank_acm

# Sum profits
all_profits['total'] = all_profits.sum(axis = 1) + 1

# Cumulative profits
all_profits['ctotal'] = np.cumprod(all_profits.total) * 100

# Add S&P 500
all_profits['sp500'] = sp500_index.iloc[364:,].Close[::5] / list(sp500_index.iloc[364:,].Close[::5])[0] * 100

# Add date
all_profits['Date'] = pd.to_datetime(sp500_index.iloc[364:,].Date[::5])

# Change index
all_profits.set_index('Date', inplace = True)
```

Overall, **the CM-based strategy beat the market**. There were periods where the strategy performed worse than the S&P 500, but the overall drawdowns were significantly lower.


```python
# Plot
plt.figure(figsize = (10, 8))
all_profits.ctotal.plot(linewidth = 1, color = '#133056')
all_profits.sp500.plot(linewidth = 1, color = '#6fceb0')
plt.xlabel('Time (5 Day Intervals)')
plt.ylabel('Returns (Base = 100)')
plt.title('Clenow Momentum Strategy vs. S&P 500', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.show()
```


![png](output_26_0.png)


The performance of the CM-based strategy and the S&P 500 buy-and-hold benchmark are: (assuming a risk-free rate of 3.00%)


```python
# Compute annualised returns
annret_cm = (all_profits.ctotal.iloc[-1] / 100) ** (50 / all_profits.shape[0]) - 1
annret_sp = (all_profits.sp500.iloc[-1] / all_profits.sp500.iloc[0]) ** (50 / all_profits.shape[0]) - 1

# Compute volatility
std_cm = (all_profits.total - 1).std() * np.sqrt(all_profits.shape[0] / 50)
std_sp = all_profits.sp500.pct_change().std() * np.sqrt(all_profits.shape[0] / 50)

# Compute max drawdown
maxdd_cm = np.min(all_profits.total - 1)
maxdd_sp = all_profits.sp500.pct_change().min()

# Configure table
sharpe_ratios = pd.DataFrame(
    [
        {'Strategy': 'CM',
         'Sharpe Ratio': (annret_cm - 0.03) / std_cm,
         'Returns': '{0:.2f}'.format(annret_cm * 100) + '%',
         'Volatility': '{0:.2f}'.format(std_cm * 100) + '%',
         'Max Drawdown': '{0:.2f}'.format(maxdd_cm * 100) + '%'},
        {'Strategy': 'Buy-and-Hold',
         'Sharpe Ratio': (annret_sp - 0.03) / std_sp,
         'Returns': '{0:.2f}'.format(annret_sp * 100) + '%',
         'Volatility': '{0:.2f}'.format(std_sp * 100) + '%',
         'Max Drawdown': '{0:.2f}'.format(maxdd_sp * 100) + '%'}
    ]
)

# Re-order and print
sharpe_ratios = sharpe_ratios[['Strategy', 'Returns', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']]
sharpe_ratios
```




<div>
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
