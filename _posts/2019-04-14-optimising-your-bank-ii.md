---
type: post  
title: "Optimising Your Bank II: OCBC vs. DBS vs. UOB vs. MayBank"  
bigimg: /img/banks.jpg
image: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/banks.jpg
share-img: /img/banks_sq.jpg
share-img2: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/banks_sq.jpg
tags: [finance, bank accounts]
---  
  
# It's Time for Re-optimisation  
Back in Oct 18, I wrote about [optimising bank interest](https://chrischow.github.io/dataandstuff/2018-10-13-optimising-your-bank/). It was a shootout between OCBC's 360 account, UOB's One account, and MayBank's SaveUp account, in which the **SaveUp** was the first choice. The **One** account complemented the SaveUp nicely. However, OCBC and MayBank changed their policies with effect from 1 Apr 19, invalidating my previous analysis. Hence, it is timely for a new exercise in optimisation. In addition to the SaveUp, 360, and One accounts, I will be adding the DBS **Multiplier** account to the analysis.


# Meet the Contenders
As in the first post, I summarise the interest rates on the 360, One, SaveUp, and Multiplier accounts.

## OCBC 360 Account
The 360 account offers the following with effect from 1 Apr 2019:  
  
| Action                                    | First \$35,000     | \$35,001 to \$70,000 |
|-------------------------------------------|--------------------|----------------------|
| Base Interest                             | 0.05%              | 0.05%                |
| Credit Salary via GIRO                    | 1.20%              | 2.00%                |
| Spend \$500 on OCBC Credit Cards          | 0.30%              | 0.60%                |
| Insure or Invest with OCBC                | 0.60%              | 1.20%                |
| Increase Monthly Account Balance by \$500 | 0.30%              | 0.60%                |
| Increase Monthly Account Balance          | 1.00% on Increment | 1.00% on Increment   |
| Account Balance is \$200,000 & Above      | 1.00%              | 1.00%                |  
  
The only change is in the increase in bonus from crediting salary from 1.50% for the second \$35,000 in bank balances to a whopping **2.00%**.  
  
*Source: [OCBC](https://www.ocbc.com/personal-banking/accounts/360-account.html)*

## UOB One Account
There has been no change to the One account's interest:  
  
![](https://www.uob.com.sg/web-resources/common/images/column-tiles/rates-table.jpg)

*Source: [UOB](https://www.uob.com.sg/personal/save/chequeing/one-account.page)*

## MayBank SaveUp Account
The SaveUp account now pays bonus interest only on the first **\$50,000** in your account instead of **\$60,000** previously, and has combined the GIRO payments and salary credit criteria into **a single criterion**. This makes it more difficult to meet the criteria for 3 products or services.
  
| Action                 | Up to \$50,000 |   |
|------------------------|----------------|---|
| Base Interest          | 0.3125%        |   |
| 1 Product or Service   | 0.30%          |   |
| 2 Products or Services | 0.80%          |   |
| 3 Products or Services | 2.75%          |   |  
  
The products and services include:  
  
1. Spend \$500 on the Platinum Visa Card or Horizon Visa Signature Card
2. Bill payments of \$300 by GIRO
3. Minimum \$2,000 salary credited via GIRO **OR** Minimum education loan of \$10,000
4. Minimum education loan of \$10,000
5. Minimum home loan of \$200,000
6. Minimum car loan of \$35,000
7. Life insurance with a minimum annual premium of \$5,000
8. Renovation loan of \$10,000
9. Minimum investment of \$25,000 in unit trusts or \$30,000 in structured deposits  
  
*Source: [MayBank](http://info.maybank2u.com.sg/saveup/)*

## DBS Multiplier Account
The Multiplier account rewards you for (1) crediting your salary and (2) making at least one more transaction in the following categories:  
  
1. Spending on DBS/POSB credit cards
2. Home loan instalments
3. Insurance
4. Investments (unit trusts, equities, or dividends crediting)
  
An interesting feature of the Multiplier is that bonus interest is awarded based on the **total transaction value per month**. The interest schedule is as follows:  
  


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
      <th></th>
      <th>Eligible Transactions</th>
      <th>Salary + 1 Criterion</th>
      <th>Salary + 2 or more Criteria</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Less than \$2,000</td>
      <td>0.50%</td>
      <td>0.50%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>\$2,000 - \$2,500</td>
      <td>1.55%</td>
      <td>1.80%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>\$2,500 - \$5,000</td>
      <td>1.85%</td>
      <td>2.00%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>\$5,000 - \$15,000</td>
      <td>1.90%</td>
      <td>2.20%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>\$15,000 - \$30,000</td>
      <td>2.00%</td>
      <td>2.30%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>\$30,000 or more</td>
      <td>2.08%</td>
      <td>3.50%</td>
    </tr>
  </tbody>
</table>
</div>



*Source: [DBS](https://www.dbs.com.sg/personal/landing/dbs-multiplier/)*

## The Competing Demands
In summary, the competing demands are:  
  
| Transaction  | 360  | One      | SaveUp  | Multiplier |
|:------------:|:----:|:--------:|:-------:|:----------:|
| Salary       | Yes  | Optional | Yes     | Yes        |
| GIRO         | No   | Optional | Yes     | No         |
| Credit Cards | Yes  | Yes      | Yes     | Yes        |
  
This means that for salary crediting, we can only choose one account, unless you have multiple sources of steady salary. For GIRO transactions, we can have both the One **and** SaveUp accounts. For credit card transactions, it depends entirely on your spending. If your monthly expenditure is high enough, you can meet the criteria for any combination or all of these accounts.

# Yield Curves
In this section, I compute the "yield curves" for the three banks with the assumption that you are able to capture the full bonus interest from performing all three of the abovementioned transactions under each bank account, separately.
  
## OCBC 360 Account
First, we set up a table with savings of \$2,000 to \$90,000:  


```python
# Set up table
ocbc = pd.DataFrame(np.arange(2000, 91000, 1000), columns = ['savings'])
```

Next, assuming we (1) credit our salary, (2) spend \$500 on OCBC credit cards, and (3) increase our monthly balance, we compute the interest for each level of savings.


```python
# Compute base interest
ocbc['base_interest'] = ocbc.savings * 0.0005

# Set up columns for bonus interest
ocbc['salary'] = 0
ocbc['credit_cards'] = 0
ocbc['monthly_balance'] = 0
ocbc['thresh0'] = 0
ocbc['thresh1'] = 35000
ocbc['thresh2'] = 70000
ocbc['diff'] = ocbc.savings - ocbc.thresh1
ocbc['diff'] = ocbc[['thresh0', 'diff']].copy().max(axis = 1)
ocbc['diff2'] = ocbc.savings - ocbc.thresh2
ocbc['diff2'] = ocbc[['thresh0', 'diff2']].copy().max(axis = 1)

# Copy data
ocbc_old = ocbc.copy()

# Compute interest for salary
ocbc.salary = ocbc[['savings', 'thresh1']].min(axis = 1) * 0.012 + \
    ocbc[['thresh1', 'diff']].min(axis = 1) * 0.02 + \
    ocbc.diff2 * 0.0005

# Compute interest for credit cards
ocbc.credit_cards = ocbc[['savings', 'thresh1']].min(axis = 1) * 0.003 + \
    ocbc[['thresh1', 'diff']].min(axis = 1) * 0.006 + \
    ocbc.diff2 * 0.0005

# Compute interest for monthly balance
ocbc.monthly_balance = ocbc[['savings', 'thresh1']].min(axis = 1) * 0.003 + \
    ocbc[['thresh1', 'diff']].min(axis = 1) * 0.006 + \
    ocbc.diff2 * 0.0005

# Compute total interest
ocbc['total_interest'] = ocbc.base_interest + ocbc.salary + ocbc.credit_cards + \
    ocbc.monthly_balance

# Compute effective interest rate (EIR)
ocbc['eir'] = ocbc.total_interest / ocbc.savings

# Delete unnecessary columns
ocbc.drop(['thresh0', 'thresh1', 'thresh2', 'diff', 'diff2'], axis = 1, inplace = True)

# Do all of the above to compute the old OCBC rates
ocbc_old.salary = ocbc_old[['savings', 'thresh1']].min(axis = 1) * 0.012 + \
    ocbc_old[['thresh1', 'diff']].min(axis = 1) * 0.015 + \
    ocbc_old.diff2 * 0.0005
ocbc_old.credit_cards = ocbc_old[['savings', 'thresh1']].min(axis = 1) * 0.003 + \
    ocbc_old[['thresh1', 'diff']].min(axis = 1) * 0.006 + \
    ocbc_old.diff2 * 0.0005
ocbc_old.monthly_balance = ocbc_old[['savings', 'thresh1']].min(axis = 1) * 0.003 + \
    ocbc_old[['thresh1', 'diff']].min(axis = 1) * 0.006 + \
    ocbc_old.diff2 * 0.0005
ocbc_old['total_interest'] = ocbc_old.base_interest + ocbc_old.salary + ocbc_old.credit_cards + \
    ocbc_old.monthly_balance
ocbc_old['eir'] = ocbc_old.total_interest / ocbc_old.savings
ocbc_old.drop(['thresh0', 'thresh1', 'thresh2', 'diff', 'diff2'], axis = 1, inplace = True)
```


![png](output_15_0.png)


From the graph above, we see that the revised criteria for bonus interest on the 360 account result in higher interest at every level above \$35,000. The peak interest rate at \$70,000 is now **2.55%**.

## UOB One Account
Next, we use the same technique as in the first post to compute the yield curve for the One account.


![png](output_19_0.png)


## MayBank SaveUp Account
With the changes in the SaveUp account criteria, the overall interest rate is substantially lower. Under the new rules, the SaveUp account hits a measly maximum of **1.065%** with savings of \$50,000, assuming we continue to meet the same 3 criteria. This is approximately one third the interest rate from before.



![png](output_23_0.png)


## DBS Multiplier
How much you save (interest rate) on the Multiplier account depends entirely on how much you spend - ironic isn't it? I omitted the Multiplier account from my first post because I thought that it was not useful for normal households. To capitalise on the high interest, clients would need to have a large bank balance, and must credit and spend a lot of money in a single month. In other words, ~~they must be rich to benefit from the account~~ DBS is targeting the customer segment with high net worth using this account.
  
However, after seeing how MayBank altered its policy and consequently, its standing in my books, I saw the value in including the Multiplier account as another option. For our computations, we only need to consider transactions per month from crediting salary and spending on credit cards. We will consider two brackets that I figure would apply to most households: (1) \$2,500 to \$5,000 and (2) \$5,000 to \$15,000. Assuming the transaction amounts are constant, the Multiplier simply pays consistent interest.



![png](output_26_0.png)


# Recommendations [TLDR]
  
## Optimal Bank Accounts



![png](output_28_0.png)


#### Bank Balance of \$22,000 and Below
The first choice account is the **Multiplier**. If your monthly transactions (including salary crediting) exceed \$5,000, you will receive the highest interest rate in the market. If not, you would match the interest rate for the other bank accounts. The second choice is the **One** account because (1) it does not require you to credit your salary to earn a competitive rate, and (2) this account is the recommended option for the next bracket, which you probably will move into at some point.  
  
#### Bank Balance of \$22,000 to \$38,000
The first choice is the **One** account. Since this account does not require you to credit your salary, we have a conditional secondary recommendation.  
  
1. **Monthly Transactions of \$5,000 to \$15,000:** Get the *Multiplier* account for the 1.90% interest rate. 
2. **Monthly Transactions of \$2,500 to \$5,000:** Get the *360* account, because the interest rate becomes optimal as your bank balance approaches \$38,000.  
  
#### Bank  Balance of \$38,000 to \$74,000
The **360** account is the first choice. The peak interest rate is 2.55% when your bank balance hits \$70,000. The **One** account is the secondary choice because it complements the 360 accounts nicely through the GIRO transaction criterion.  
  
#### Bank Balance of \$74,000 and Above
The **One** account becomes the preferred account, while the **360** account becomes the secondary account. If you had both accounts open, no change in setup is required because the criteria from the two accounts are complementary.  
  
## Optimise Based on the Steady State
The banks entice customers with higher interest rates for finite periods of time. Here are some of the following criteria that typically award bonus interest for up to 12 months only:  
  
1. Insurance plans
2. Investment plans
3. Loans: education, car, or renovation
  
The only type of loan worth pursuing is the DBS Home Loan. There is no restriction on the amount of time that bonus interest is awarded. In addition, the Multiplier's interest rate for salary crediting plus 2 criteria is quite competitive for balances up to \$47,000.  
  
## Keep Your Options Open
Just 6 months ago, the SaveUp account was the optimal account to deposit your money, regardless of bank balance. Now, the SaveUp account does not feature as a primary or secondary choice under any circumstances. This shows how non-committal the banks can be. All banks "*may at their own discretion/reserve the right to vary the base interest rate and bonus interest rates"*. Yet, this does not mean that we should close our SaveUp accounts. Since the rates can change at any time, keep multiple accounts open for when these changes take place, and re-optimise accordingly.
  
---
Click [here](http://nbviewer.ipython.org/github/chrischow/dataandstuff/blob/65e677cd746cadaa1c349d6ab96e7280d695db7d/notebooks/2018-10-13-optimising-your-bank.ipynb){:target="_blank"} for the full Jupyter notebook.
  
Credits for images: [Financial Times](https://www.ft.com/)
  