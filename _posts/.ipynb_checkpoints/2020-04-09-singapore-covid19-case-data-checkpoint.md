---
type: post  
title: "Singapore COVID-19 Case Data"  
bigimg: /img/covid19.jpg
image: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/covid19_sq.jpg
share-img: /img/covid19.jpg
share-img2: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/covid19_sq.jpg
tags: [exploratory data analysis, COVID-19]
---  

On Monday, I came down with flu (better now) and was issued a 5-day MC. Stuck at home, I thought it would be interesting to explore data on Singapore's COVID-19 cases to see what insights could be drawn. This post contains some exploratory data analysis (EDA) on data from the [Gov.sg COVID-19 Dashboard](https://www.gov.sg/article/covid-19-cases-in-singapore) and the [Against COVID-19 Singapore Dashboard](https://www.againstcovid19.com/singapore/dashboard).

## TL;DR
The key observations:
* The age group with the most cases is the 20-to-30 group.
* There are more males (59%) than females (41%) infected with COVID-19.
* Currently, there are more imported (54%) than local (46%) cases.
* Most cases came from a single destination before contracting COVID-19.
* About half of all cases had a hospital stay of 11 days.
* Singapore citizens make up a majority (60%) of the cases.
* About 73% of cases are still in hospital, while a good 27% of cases have been discharged.
* Approximately 58% (695 of 1189) of cases had no link to known cases.
* Of those who received the virus from at least one person (291), 70% (204) did not transmit the virus further, while 20% (56) transmitted to only one other person.
* If we consider only cases that have reached some conclusion (recovery/death), the recovery rate is **98%**. *GO HEALTHCARE WORKERS.*

We couldn't predict discharges well, possibly because:
1. **There was insufficient data.** We did not have enough personal medical data.
2. **There will always be insufficient data.** Data will not be generated fast enough for us to have a workable sample size because (a) the time from infection to discharge/death is rather long and (b) our population is just too small.
3. **Changing relationship between predictors and outcomes.** As hospitals become more capable and the government introduces more enhanced measures to combat COVID-19, the relationships in the data will change over time.


# The Data
As mentioned earlier, we use data from the [Gov.sg COVID-19 Dashboard](https://www.gov.sg/article/covid-19-cases-in-singapore) and the [Against COVID-19 Singapore Dashboard](https://www.againstcovid19.com/singapore/dashboard).

## Gov.sg COVID-19 Dashboard
We extract data from [this page](https://experience.arcgis.com/experience/7e30edc490a5441a874f9efe67bd8b89), the actual dashboard on ArcGIS, and perform some basic data cleaning.


```python
# Load data
with open('sg_covid19_raw.txt') as json_file:
    cv_json = json.load(json_file)

# Extract attributes
raw_data = [x['attributes'] for x in cv_json]

# Convert to data frame
df = pd.DataFrame(raw_data)

# Configure columns to drop
drop_cols = ['CASE_NEGTV', 'LAT', 'LONG', 'DT_CAS_TOT', 'POST_CODE', 'TOT_COUNT', 'CASE_PENDG', 'ObjectId', 'Case_Count', 'CNFRM_CLR', 'Case_total', 'Confirmed', 'DEATH', 'DISCHARGE', 'Date_of_Di', 'PLAC_VISTD', 'RESPOSTCOD', 'RES_LOC', 'Suspct_Cas', 'Tot_ICU', 'Tot_Impotd', 'Tot_NonICU', 'Tot_local', 'UPD_AS_AT']

# Drop columns
df = df.drop(drop_cols, axis=1)

# Extract date
df['date'] = df.apply(lambda x: str(datetime.datetime.fromtimestamp(x.Date_of_Co/1000).date()), axis=1)

# Convert to date
df['date'] = pd.to_datetime(df.date)

# Compute date since identified
df['days'] = (pd.to_datetime('2020-04-07') - df.date).dt.days

# Drop old date
df = df.drop('Date_of_Co', axis=1)

# Display data
display(df.head())
```


<div style="overflow-x:auto; width: 100%;">
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
      <th>Age</th>
      <th>Case_ID</th>
      <th>Cluster</th>
      <th>Current_Lo</th>
      <th>Gender</th>
      <th>Imported_o</th>
      <th>Nationalit</th>
      <th>PHI</th>
      <th>Place</th>
      <th>PlanningAr</th>
      <th>Prs_rl_URL</th>
      <th>Region_201</th>
      <th>Status</th>
      <th>date</th>
      <th>days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>1189</td>
      <td>0</td>
      <td>KTPH</td>
      <td>F</td>
      <td>Local</td>
      <td>Singapore Citizen</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54</td>
      <td>1188</td>
      <td>0</td>
      <td>KTPH</td>
      <td>M</td>
      <td>Local</td>
      <td>Singapore Citizen</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>1187</td>
      <td>0</td>
      <td>KTPH</td>
      <td>M</td>
      <td>Local</td>
      <td>Bangladeshi (Singapore Work Pass holder)</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>1186</td>
      <td>0</td>
      <td>NCID</td>
      <td>M</td>
      <td>Local</td>
      <td>Bangladeshi (Long Term Pass holder)</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>1185</td>
      <td>0</td>
      <td>NCID</td>
      <td>M</td>
      <td>Local</td>
      <td>Indian (Long Term Pass holder)</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


## Against COVID-19 Dashboard
There wasn't a quick and easy way to extract the data from this website. Hence, I downloaded the data page by page into 12 CSV files.


```python
# Get file list
filelist = os.listdir('Against COVID-19')

# Create dataframe
df2 = pd.DataFrame()
for f in filelist:
    df2 = df2.append(pd.read_csv('Against COVID-19/' + f))

# Remove null entries
df2['Recovered At'] = df2['Recovered At'].str.replace('-', '')

# Convert to date
df2['Recovered At'] = pd.to_datetime(df2['Recovered At'])

# Display data
display(df.head())
```


<div style="overflow-x:auto; width: 100%;">
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
      <th>Age</th>
      <th>Case_ID</th>
      <th>Cluster</th>
      <th>Current_Lo</th>
      <th>Gender</th>
      <th>Imported_o</th>
      <th>Nationalit</th>
      <th>PHI</th>
      <th>Place</th>
      <th>PlanningAr</th>
      <th>Prs_rl_URL</th>
      <th>Region_201</th>
      <th>Status</th>
      <th>date</th>
      <th>days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>1189</td>
      <td>0</td>
      <td>KTPH</td>
      <td>F</td>
      <td>Local</td>
      <td>Singapore Citizen</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54</td>
      <td>1188</td>
      <td>0</td>
      <td>KTPH</td>
      <td>M</td>
      <td>Local</td>
      <td>Singapore Citizen</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>1187</td>
      <td>0</td>
      <td>KTPH</td>
      <td>M</td>
      <td>Local</td>
      <td>Bangladeshi (Singapore Work Pass holder)</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>1186</td>
      <td>0</td>
      <td>NCID</td>
      <td>M</td>
      <td>Local</td>
      <td>Bangladeshi (Long Term Pass holder)</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>1185</td>
      <td>0</td>
      <td>NCID</td>
      <td>M</td>
      <td>Local</td>
      <td>Indian (Long Term Pass holder)</td>
      <td>0</td>
      <td>Singapore</td>
      <td>0</td>
      <td>https://www.moh.gov.sg/news-highlights/details...</td>
      <td>0</td>
      <td>Hospitalised</td>
      <td>2020-04-04</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


There was also a [network graph](https://www.againstcovid19.com/singapore/cases) indicating the linkages between cases. From this, we extract:

* `rx`: The number of people that each case possibly received the virus from
* `tx`: The number of people that each case possibly transmitted the virus to
* `Cluster X`: The various clusters as identified by MOH


```python
# Load graph data
with open('Against COVID-19/network_graph.json') as json_file:
    graph = json.load(json_file)

# Convert to data frame
nodes = pd.DataFrame(graph['nodes'])
edges = pd.DataFrame(graph['edges'])

# Drop unused
nodes = nodes.drop(['borderWidth', 'color', 'shape', 'size', 'x', 'y'], axis=1)
edges = edges.drop(['arrows', 'color', 'dashes', 'font'], axis=1)

# Partition datasets into clusters and nodes
case_clusters = nodes[nodes.id.str.contains('Cluster')]
case_nodes = nodes[~nodes.id.str.contains('Cluster')]

# Extract case ID
case_nodes['label'] = pd.to_numeric(case_nodes.label.str.replace('Case ', '').str.replace(' .*', ''))

# Map case ID
edges['case_id_from'] = edges['from'].map(case_nodes[['label', 'id']].set_index('id').to_dict()['label']).fillna(0).astype(int)
edges['case_id_to'] = edges['to'].map(case_nodes[['label', 'id']].set_index('id').to_dict()['label']).fillna(0).astype(int)

# List of clusters
clusters = sorted(case_clusters.id.unique())

# Initialise features
df2['rx'] = 0
df2['tx'] = 0

for clust in clusters:
    df2[clust] = 0

# Extract features
for case in tqdm(df2.Case):
    
    # Pull data
    temp_dat = edges[(edges.case_id_to == case) | (edges.case_id_from == case)]
    
    # Received
    df2['rx'].loc[df2.Case == case] = temp_dat[(temp_dat['from'].str.contains('Case')) & (temp_dat.case_id_to == case)].shape[0]
    
    # Transmitted
    df2['tx'].loc[df2.Case == case] = temp_dat[(temp_dat['to'].str.contains('Case')) & (temp_dat.case_id_from == case)].shape[0]
    
    # Extract clusters
    temp_clust = temp_dat['from'].loc[temp_dat['from'].str.contains('Cluster')].to_list() + temp_dat['to'].loc[temp_dat['to'].str.contains('Cluster')].to_list()
    
    if len(temp_clust) > 0:
        df2.loc[df2.Case == case, temp_clust] = 1

# Count number of clusters
df2['n_clusters'] = df2[clusters].sum(axis=1)

# Save
df2.to_csv('against_covid_clean.csv', index=False)

# Load
df2 = pd.read_csv('against_covid_clean.csv')
df2['Recovered At'] = pd.to_datetime(df2['Recovered At'])
```


## Data Cleaning
First, we merge the datasets. I was particularly interested in the date that cases recovered (`Recovered At`) and the network features created above, since these were lacking in the Gov.sg dataset.


```python
# Merge
df = df.merge(df2[['Case', 'Patient', 'Recovered At', 'tx', 'rx', 'n_clusters'] + clusters], how='left', left_on='Case_ID', right_on='Case')

# Update case status
df['Status'].loc[df['Recovered At'].notnull()] = 'Discharged'
```

Next, we clean up the string features and create binary variables for each country that each individual visited prior to contracting COVID-19. We also create three features:

1. `n_sources`: Number of countries that the individual visited.
2. `day_year`: Day of the year that the individual was identified as a case.
3. `days`: No. of days that the individual spent in hospital. The start date was always the date that the individual was identified as a case. If the individual is still hospitalised, we used today's date as the end date. If the individual was discharged, we used the recovery date as the end date.


```python
# Gender
df['Gender'] = df.Gender.str.strip().str.lower()

# Nationality
df['nationality'] = df.Nationalit.str.replace('\(.*\)', '').str.strip().str.lower()
df['nationality'].loc[df.nationality == 'india'] = 'indian'
df['nationality'].loc[df.nationality == 'switzerland'] = 'swiss'
df['nationality'].loc[df.nationality == 'myanmar'] = 'burmese'
df['nationality'].loc[df.nationality == 'uk'] = 'british'

# Place
df['Place'].loc[df.Place=='0'] = 'Unknown'
df['Place'] = df.Place.str.replace('United Arab Emirates', 'UAE')
df['Place'] = df.Place.str.replace('Sri Lanka', 'srilanka')
df['Place'] = df.Place.str.replace('South Africa', 'southafrica')
df['Place'] = df.Place.str.replace('Czech Republic', 'czechrepublic')
df['Place'] = df.Place.str.replace('Eastern Europe', 'europe')
df['Place'] = df.Place.str.replace('Moscow', '')
df['Place'] = df.Place.str.replace('Copenhagen', '')
df['Place'] = df.Place.str.replace('London', 'uk')

# Create dummy variables for each location
# Initialised and fit TF-IDF Vectorizer
tf = CountVectorizer()
tf.fit(df.Place)

# Create data frame of source
source_df = pd.DataFrame(tf.transform(df.Place).toarray(), columns=['from_' + x for x in tf.get_feature_names()])
for col in source_df:
    source_df[col] = source_df[col].astype(int)
    
# Merge
df = pd.concat([df, source_df], axis=1)

# Add column
df['n_sources'] = source_df.sum(axis=1)

# Days since 1st Jan
df['day_year'] = (df['date'] - pd.to_datetime('2020-01-01')).dt.days

# No. of days in hospital
df['days'].loc[df['Recovered At'].notnull()] = (df['Recovered At'].loc[df['Recovered At'].notnull()] - df['date'].loc[df['Recovered At'].notnull()]).dt.days

# Fill missing
df[clusters] = df[clusters].fillna(0)
df['rx'] = df.rx.fillna(0)
df['tx'] = df.tx.fillna(0)
df['n_clusters'] = df.n_clusters.fillna(0)
```

# Exploratory Data Analysis
Now, we dive into exploratory data analysis (EDA), first for **individual features**, and then for the **relationship between status (discharged/deceased/hospitalised) and individual features**.

## Individual Features
In this section, we visualise the key features in the dataset: (1) Age, (2) Gender, (3) Imported/Local, (4) No. of Sources, (5) Days in Hospital, and (6) Status. I omit the other features for the following reasons:

* `Region_201` and `PlanningAr`: These features contained information on the location of cases in Singapore e.g. Toa Payoh/Bishan, East/West. A majority of the data was missing (~67%).
* `PHI`, `Current_Lo`, and `Cluster`: These features contained information on the hospitals that cases were treated/held at. A large proportion of the data was missing (63-67%). Also, I didn't think it would be very fair to compare the discharge/death rates among the various hospitals because there is a lot of randomness in the kind of cases that get resolved.  


![](../graphics/2020-04-09-singapore-covid19-case-data/output_15_0.png)



![](../graphics/2020-04-09-singapore-covid19-case-data/output_15_1.png)


Here are the key observations, summarised:

* **Age:** The age group with the most cases is 20 to 30 years of age.
* **Gender:** There are more males (59%) than females (41%) infected with COVID-19.
* **Imported/Local:** Currently, there are more imported (54%) than local (46%) cases.
* **Sources:** Most cases came from a single destination before contracting COVID-19.
* **Days in Hospital:** About half of all cases had a hospital stay of 11 days.
* **Nationality:** Singapore citizens make up a majority (60%) of the cases.
* **Linkages:** The data suggests that most people did not receive or transmit the virus from anyone. This requires some investigation.
* **Outcomes:**
    * About 73% of cases are still in hospital, while a good 27% of cases have been discharged.
    * If we consider only cases that have reached some conclusion (recovery/death), the recovery rate is **98%**. That's 320 of 326 cases. *GO HEALTHCARE WORKERS.*

# Linkages
The network data is rather interesting. Here are some observations:

* Approximately 58% (695 of 1189) of the cases had no link to known cases.
* Of those who received the virus from at least one person (291), 70% (204) did not transmit the virus further, while 20% (56) transmitted to only one other person.


```python
# Two-way table
txrx = pd.crosstab(df.rx, df.tx, margins=True)
display(txrx)
```


<div style="overflow-x:auto; width: 100%;">
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
      <th>tx</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>3.0</th>
      <th>4.0</th>
      <th>5.0</th>
      <th>9.0</th>
      <th>All</th>
    </tr>
    <tr>
      <th>rx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>695</td>
      <td>164</td>
      <td>25</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>898</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>153</td>
      <td>36</td>
      <td>11</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>210</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>30</td>
      <td>14</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>51</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>10</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>All</th>
      <td>899</td>
      <td>220</td>
      <td>44</td>
      <td>16</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>1189</td>
    </tr>
  </tbody>
</table>
</div>


The data shows that the government's enhanced measures to combat COVID-19 appear to be working based on the no-transmission rate over time. This rate measures the percentage of people who did not transmit COVID-19 after contracting it.

Initially, the no-transmission rate (dark blue line) was high because cases were primarily imported (red line). Subsequently, the no-transmission rate fell alongside the number of imported cases fell, indicating that local transmissions rose. The no-transmission rate dropped below 50% shortly after DORSCON was raised to Orange on 7 Feb 20. Thereafter, the no-transmission rate has risen steadily as the government introduced enhanced precautionary measures. 


```python
# Get no-transmission frequencies over time
freqs = []
impts = []

for i in df.date.unique():
    temp_df = df[df.date <= i]
    freqs.append(np.mean(temp_df.tx.loc[temp_df.rx > 0] == 0))
    impts.append(np.mean(temp_df.Imported_o == 'Imported'))

# Convert to data frame
freq_ts = pd.DataFrame({'date': df.date.unique(), 'tx_pct': freqs, 'imported': impts})
```


![](../graphics/2020-04-09-singapore-covid19-case-data/output_20_0.png)


## Relationship Between Status and Individual Features
Next, we breakdown the distribution of each individual feature by the case status (Hospitalised/Discharged/Deceased). Note that there were only 6 deaths, so take the statistics and observations on this group may not generalise well.

### Status vs. Age
Thus far, most of the people who passed on due to COVID-19 were older on average (and the data was relatively tightly grouped around this mean). Meanwhile, there wasn't a big difference between the distribution of age for those currently hospitalised and those who were discharged.


![](../graphics/2020-04-09-singapore-covid19-case-data/output_23_0.png)


### Status vs. Days
Unlike Age, there was no clear pattern for the number of days spent in hospital. On average, deceased cases spent more time in hospital, but the variance was high. Meanwhile, there was no discernible difference in the distributions of days spent in hospital for those currently hospitalised and those who were discharged.



![](../graphics/2020-04-09-singapore-covid19-case-data/output_25_0.png)


### Status vs. Gender
4 of the 6 (66%) deceased cases were male. Once again, there was no distinct difference in gender proportions for the hospitalised and discharged groups. Refer to the table below the graph for the actual numbers.


```python
# Create dummy variable for males
df['male'] = (df.Gender == 'm').astype(int)
```


![](../graphics/2020-04-09-singapore-covid19-case-data/output_27_0.png)



<div style="overflow-x:auto; width: 100%;">
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
      <th>Status</th>
      <th>Deceased</th>
      <th>Discharged</th>
      <th>Hospitalised</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>f</th>
      <td>2</td>
      <td>132</td>
      <td>346</td>
    </tr>
    <tr>
      <th>m</th>
      <td>4</td>
      <td>188</td>
      <td>517</td>
    </tr>
  </tbody>
</table>
</div>


### Status vs. Imported/Local
4 of the 6 (66%) deceased cases were local. There wasn't a big difference in proportion of locals for the hospitalised and discharged groups. Refer to the table below the graph for the actual numbers.


```python
# Create dummy variable for males
df['local'] = (df.Imported_o == 'Local').astype(int)
```


![](../graphics/2020-04-09-singapore-covid19-case-data/output_29_0.png)



<div style="overflow-x:auto; width: 100%;">
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
      <th>Status</th>
      <th>Deceased</th>
      <th>Discharged</th>
      <th>Hospitalised</th>
    </tr>
    <tr>
      <th>local</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>137</td>
      <td>407</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>183</td>
      <td>456</td>
    </tr>
  </tbody>
</table>
</div>


### Status vs. No. of Cases That Individuals Contracted COVID-19 From
Here, we examine the proportions of those hospitalised and discharged who contracted COVID-19 from *X* cases. Strangely, the discharged cases had stronger linkages to past cases.


```python
# Compute proportions
rx_disc = df.rx.loc[df.Status=='Discharged'].value_counts() / df.rx.loc[df.Status=='Discharged'].shape[0]
rx_hosp = df.rx.loc[df.Status=='Hospitalised'].value_counts() / df.rx.loc[df.Status=='Hospitalised'].shape[0]
```


![](../graphics/2020-04-09-singapore-covid19-case-data/output_31_1.png)


### Status vs. Nationality
The graph below plots the proportions of individuals from each nationality who were hospitalised/discharged/deceased. For quick reference, the teal bars represent discharged cases, red represents deceased cases, and light blue represents hospitalised cases.


```python
# Cross-tabulate data
stat_nat = pd.crosstab(df.Status, df.nationality).T

# Compute proportions
stat_nat['totals'] = stat_nat.sum(axis=1)
stat_nat['Deceased'] = stat_nat.Deceased / stat_nat.totals
stat_nat['Discharged'] = stat_nat.Discharged / stat_nat.totals
stat_nat['Hospitalised'] = stat_nat.Hospitalised / stat_nat.totals

# Drop totals
stat_nat = stat_nat.drop('totals', axis=1)
```


![](../graphics/2020-04-09-singapore-covid19-case-data/output_33_0.png)


# Factors for Discharge
Finally, we try our luck to model the factors contributing to discharge. Our target is a binary feature for discharged vs. not discharged today, 08 Apr 2020. I should probably state upfront that there are severe limitations from modelling cases that will be discharged in this manner. This is because (a) most cases have not concluded yet, and (b) we are effectively ignoring deaths by lumping it with hospitalised cases. Let's see what we get anyway, without any high hopes.


```python
df['Recovered At'] = df['Recovered At'].fillna(pd.to_datetime('2020-04-08'))

df['prior_disc'] = 0
for i in df.date.unique():
    df['prior_disc'].loc[df.date==i] = df[(df['Recovered At'] < i)].shape[0]
```

## Statistical Results

In our logistic regression model, we include the following features:

1. Age
2. Gender (male)
3. Source (local)
4. Number of sources
5. Day of the year
6. An interaction variable between age and source (local)
7. An interaction variable between age and gender (male).


```python
# Prepare data
X_dat = pd.get_dummies(df[['Age', 'male', 'local', 'n_sources', 'day_year', 'rx', 'tx', 'n_clusters']])
X_dat['const'] = 1.0
y_dat = (df.Status == 'Discharged').astype(int)

# Create interactions
X_dat['age_local'] = X_dat.Age * X_dat.local
X_dat['age_male'] = X_dat.Age * X_dat.male

# Fit model
logreg = Logit(y_dat, X_dat)
logreg_fitted = logreg.fit(maxiter=10000)

# Print summary
print(logreg_fitted.summary())
```

    Optimization terminated successfully.
             Current function value: 0.350936
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                 Status   No. Observations:                 1189
    Model:                          Logit   Df Residuals:                     1178
    Method:                           MLE   Df Model:                           10
    Date:                Thu, 09 Apr 2020   Pseudo R-squ.:                  0.3974
    Time:                        11:19:42   Log-Likelihood:                -417.26
    converged:                       True   LL-Null:                       -692.47
    Covariance Type:            nonrobust   LLR p-value:                7.323e-112
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Age            0.0094      0.010      0.976      0.329      -0.009       0.028
    male           0.9801      0.477      2.055      0.040       0.045       1.915
    local          0.0140      0.521      0.027      0.979      -1.008       1.036
    n_sources     -0.0052      0.198     -0.026      0.979      -0.393       0.383
    day_year      -0.1834      0.013    -13.592      0.000      -0.210      -0.157
    rx            -0.1402      0.114     -1.232      0.218      -0.363       0.083
    tx             0.0565      0.125      0.453      0.650      -0.188       0.301
    n_clusters    -0.3783      0.256     -1.478      0.139      -0.880       0.123
    const         13.0321      1.201     10.847      0.000      10.677      15.387
    age_local      0.0056      0.011      0.512      0.608      -0.016       0.027
    age_male      -0.0224      0.011     -2.116      0.034      -0.043      -0.002
    ==============================================================================
    

As shown above, the significant features were gender (male), the day of the year, the interaction term between age and gender (male), and a constant. The coefficients above capture the impact of each feature on **log odds**. To see the impact on **probabilities**, we use the marginal effects table below:


```python
# Print marginal effects
display(logreg_fitted.get_margeff().summary())
```


<table class="simpletable">
<caption>Logit Marginal Effects</caption>
<tr>
  <th>Dep. Variable:</th> <td>Status</td> 
</tr>
<tr>
  <th>Method:</th>         <td>dydx</td>  
</tr>
<tr>
  <th>At:</th>            <td>overall</td>
</tr>
</table>
<table class="simpletable">
<tr>
       <th></th>         <th>dy/dx</th>    <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Age</th>        <td>    0.0010</td> <td>    0.001</td> <td>    0.978</td> <td> 0.328</td> <td>   -0.001</td> <td>    0.003</td>
</tr>
<tr>
  <th>male</th>       <td>    0.1062</td> <td>    0.051</td> <td>    2.067</td> <td> 0.039</td> <td>    0.005</td> <td>    0.207</td>
</tr>
<tr>
  <th>local</th>      <td>    0.0015</td> <td>    0.056</td> <td>    0.027</td> <td> 0.979</td> <td>   -0.109</td> <td>    0.112</td>
</tr>
<tr>
  <th>n_sources</th>  <td>   -0.0006</td> <td>    0.021</td> <td>   -0.026</td> <td> 0.979</td> <td>   -0.043</td> <td>    0.041</td>
</tr>
<tr>
  <th>day_year</th>   <td>   -0.0199</td> <td>    0.001</td> <td>  -18.321</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.018</td>
</tr>
<tr>
  <th>rx</th>         <td>   -0.0152</td> <td>    0.012</td> <td>   -1.234</td> <td> 0.217</td> <td>   -0.039</td> <td>    0.009</td>
</tr>
<tr>
  <th>tx</th>         <td>    0.0061</td> <td>    0.014</td> <td>    0.453</td> <td> 0.650</td> <td>   -0.020</td> <td>    0.033</td>
</tr>
<tr>
  <th>n_clusters</th> <td>   -0.0410</td> <td>    0.028</td> <td>   -1.477</td> <td> 0.140</td> <td>   -0.095</td> <td>    0.013</td>
</tr>
<tr>
  <th>age_local</th>  <td>    0.0006</td> <td>    0.001</td> <td>    0.513</td> <td> 0.608</td> <td>   -0.002</td> <td>    0.003</td>
</tr>
<tr>
  <th>age_male</th>   <td>   -0.0024</td> <td>    0.001</td> <td>   -2.128</td> <td> 0.033</td> <td>   -0.005</td> <td>   -0.000</td>
</tr>
</table>


To loosely interpret the marginal effects, on average (and ceteris paribus)...

* Males appear to have a 10% *higher* probability of getting discharged
* For males, the relationship between age and the probability of getting discharged is more negative than that of females, by 0.2%
* Individuals who were identified earlier seem to have a 2% *lower* probability of getting discharged

## Model-Agnostic Interpretation

Next, we dive into some (explainable) machine learning (ML) by computing Shapley values to understand the broad impact of the features on the probability of getting discharged at the local (each sample) and global (across all samples) levels. The Shapley value is the average of all marginal contributions of a feature to all possible coalitions (of feature values). See [Google's Explainable AI white paper](https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf) for a more detailed explanation.

We re-estimate the logistic regression model with the `sklearn` module. We leave all features in, even the statistically insignificant ones, since we care more about predictive power for the rest of this section.


```python
# Prepare data
X_dat = pd.get_dummies(df[['Age', 'male', 'local', 'n_sources', 'day_year', 'rx', 'tx', 'n_clusters']])
X_dat['age_local'] = X_dat.Age * X_dat.local
X_dat['age_male'] = X_dat.Age * X_dat.male
y_dat = (df.Status == 'Discharged').astype(int)

# Initialise model
logreg = LogisticRegression(max_iter=1000, solver='saga', n_jobs=4)

# Fit and predict
logreg.fit(X_dat, y_dat)
preds = logreg.predict(X_dat)

# Plot Shapley values
shap_log = shap.LinearExplainer(logreg, X_dat).shap_values(X_dat)
shap.summary_plot(shap_log, X_dat, plot_size=(15, 10))
```


![](../graphics/2020-04-09-singapore-covid19-case-data/output_44_0.png)


In the diagram above, each value (dots) of each feature (rows) is plotted. High values of each feature are in red, while the low values are in blue. As an example, let's look at Age.
* High values (in red) lie on the right side of the scale, implying a positive impact on the probability of getting discharged
* Meanwhile, low values (in blue) lie on the left side of the scale, implying a negative impact on the probability of getting discharged
* Together, this means a positive correlation between age and getting discharged, which matches our findings above
* ***In English:** the older you are, the higher the chance of getting discharged*

Another example would be the number of prior discharged cases:
* High values (in red) lie on the left, implying a negative impact on the probability of getting discharged
* Low values (in blue) lie on the right, implying a positive impact on the probability of getting discharged
* Together, this means a negative relationship between the number of prior discharged cases and getting discharged, which matches our findings above
* ***In English:** the earlier you contracted COVID-19, the lower the probability of getting discharged*

## Inspection of Predictions
Another important step in model evaluation is inspecting predictions to see if our predictions make sense. If this were an ML project, I would have done this first, in addition to computing standard classification metrics like precision, recall, and Area Under Curve (AUC). After all, ML cares more about whether you shoot accurately, regardless of technique, while statistics cares more about whether you held the gun and pulled the trigger using the proper technique.


```python
# Copy data
high_prob_df = df[['Case_ID', 'Age', 'Gender', 'Nationalit', 'Status']].copy()

# Insert predicted probabilities
high_prob_df['prob'] = logreg.predict_proba(X_dat)[:, 1]

# Inspect predictions
display(high_prob_df.sort_values('prob', ascending=False)[high_prob_df.Status!='Discharged'].head(20))
```


<div style="overflow-x:auto; width: 100%;">
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
      <th>Case_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Nationalit</th>
      <th>Status</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>919</th>
      <td>270</td>
      <td>62</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.944303</td>
    </tr>
    <tr>
      <th>1076</th>
      <td>113</td>
      <td>42</td>
      <td>m</td>
      <td>French (Singapore Work Pass Holder)</td>
      <td>Hospitalised</td>
      <td>0.884593</td>
    </tr>
    <tr>
      <th>1147</th>
      <td>42</td>
      <td>39</td>
      <td>m</td>
      <td>Bangladeshi (Singapore Work Pass Holder)</td>
      <td>Hospitalised</td>
      <td>0.873572</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>41</td>
      <td>71</td>
      <td>m</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.866461</td>
    </tr>
    <tr>
      <th>1154</th>
      <td>35</td>
      <td>64</td>
      <td>m</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.841244</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>90</td>
      <td>75</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Deceased</td>
      <td>0.823856</td>
    </tr>
    <tr>
      <th>996</th>
      <td>193</td>
      <td>26</td>
      <td>m</td>
      <td>Malaysian (Singapore Work Pass Holder)</td>
      <td>Hospitalised</td>
      <td>0.790345</td>
    </tr>
    <tr>
      <th>271</th>
      <td>918</td>
      <td>86</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Deceased</td>
      <td>0.763554</td>
    </tr>
    <tr>
      <th>1047</th>
      <td>142</td>
      <td>26</td>
      <td>m</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.701128</td>
    </tr>
    <tr>
      <th>862</th>
      <td>327</td>
      <td>38</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.684862</td>
    </tr>
    <tr>
      <th>995</th>
      <td>194</td>
      <td>24</td>
      <td>f</td>
      <td>Chinese (Singapore Work Pass Holder)</td>
      <td>Hospitalised</td>
      <td>0.675249</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1121</td>
      <td>66</td>
      <td>m</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.662945</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>129</td>
      <td>68</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.646263</td>
    </tr>
    <tr>
      <th>1062</th>
      <td>127</td>
      <td>64</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.635778</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>128</td>
      <td>70</td>
      <td>m</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.619108</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>130</td>
      <td>66</td>
      <td>m</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.594735</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>120</td>
      <td>62</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.592831</td>
    </tr>
    <tr>
      <th>1028</th>
      <td>161</td>
      <td>73</td>
      <td>m</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.564577</td>
    </tr>
    <tr>
      <th>861</th>
      <td>328</td>
      <td>63</td>
      <td>f</td>
      <td>Singapore Permanent Resident</td>
      <td>Hospitalised</td>
      <td>0.563402</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>182</td>
      <td>76</td>
      <td>f</td>
      <td>Indonesian</td>
      <td>Hospitalised</td>
      <td>0.562031</td>
    </tr>
  </tbody>
</table>
</div>


Above, I list the top 20 cases that we got wrong, ordered by the predicted probability of passing. A big problem: **two of the four decease cases appear here**. In fact, one of them had a predicted discharge probability of 82%!


```python
# Latest discharged cases
display(high_prob_df[high_prob_df.Case_ID.isin([1167, 1076, 1038, 1032, 1020, 1004])])
```


<div style="overflow-x:auto; width: 100%;">
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
      <th>Case_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Nationalit</th>
      <th>Status</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1167</td>
      <td>58</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.121341</td>
    </tr>
    <tr>
      <th>113</th>
      <td>1076</td>
      <td>54</td>
      <td>m</td>
      <td>Singapore Permanent Resident</td>
      <td>Hospitalised</td>
      <td>0.148745</td>
    </tr>
    <tr>
      <th>151</th>
      <td>1038</td>
      <td>52</td>
      <td>f</td>
      <td>Singapore Citizen</td>
      <td>Hospitalised</td>
      <td>0.110903</td>
    </tr>
    <tr>
      <th>157</th>
      <td>1032</td>
      <td>28</td>
      <td>f</td>
      <td>Chinese (Singapore Work Pass holder)</td>
      <td>Hospitalised</td>
      <td>0.071548</td>
    </tr>
    <tr>
      <th>169</th>
      <td>1020</td>
      <td>45</td>
      <td>m</td>
      <td>Bangladeshi (Long Term Pass holder)</td>
      <td>Hospitalised</td>
      <td>0.138252</td>
    </tr>
    <tr>
      <th>185</th>
      <td>1004</td>
      <td>32</td>
      <td>f</td>
      <td>Indian (Singapore Work Pass holder)</td>
      <td>Hospitalised</td>
      <td>0.099359</td>
    </tr>
  </tbody>
</table>
</div>


As a second check, I picked out the latest cases that were discharged (above). All of them were predicted for discharge with a probability of 15% or lower. These results should make you suspicious of everything we've learned from the models above.

## Model Limitations
Overall, the model was simply bad. There are several possible reasons why:

#### There was insufficient personal data
Besides gender, age, and nationality, no other personal data was provided (yay PDPA!). This is a major factor. A person's time to recovery (or even the possibly of recovery) depends on his/her current health, any existing medical conditions, and maybe genes. A epidemiologist could probably tell you more.

#### There will always be insufficient data
Even if we had a lot of data on personal characteristics, we still need a large sample size. However, data will not be generated fast enough. This is because (a) the time from infection to discharge/death is rather long and (b) our population is just too small.

#### Regime change
This refers to the changing relationship between the predictors (i.e. personal data) and the outcome (recovery/death). As our hospitals become better equipped and able to fight COVID-19, we could see, for example, a decrease in the number of days spent in hospital, or fewer deaths. As the government introduces more measures, the local transmission rate could decrease further. These changes could affect a model that predicts outcomes.

# Conclusion
In this post, we explored data on COVID-19 cases in Singapore from the [Gov.sg COVID-19 Dashboard](https://www.gov.sg/article/covid-19-cases-in-singapore) and the [Against COVID-19 Singapore Dashboard](https://www.againstcovid19.com/singapore/dashboard). While we made some general observations about age, gender, source, nationality, days in hospital, and outcome (hospitalised/deceased/discharged), we could not model the outcomes well. This was expected, and probably was so because we didn't (and won't) have enough data.

The most impactful finding for me was the 98% (320 of 326) rate. Our healthcare workers are working themselves to the bone, and the results are clear. To everyone on the frontline in our fight against COVID-19: Stay strong - the nation is behind you! To everyone else: Stay healthy and safe.
