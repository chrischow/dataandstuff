---
type: post  
title: "Extracting and Defining Data Team Roles from Job Listings with Natural Language Processing"
bigimg: /img/jobdesc.jpg
image: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/jobdesc_sq.jpg
share-img: /img/jobdesc.jpg
share-img2: https://raw.githubusercontent.com/chrischow/dataandstuff/gh-pages/img/jobdesc_sq.jpg
tags: [exploratory data analysis, natural language processing]
---  
This week, I took the opportunity to continue my personal development in data science, finishing up both the Data Scientist and Machine Learning Scientist career tracks on [DataCamp](https://www.datacamp.com/tracks/career/). Having also completed the Data Analyst career track before this, I noticed some overlap in the courses for Data Analysts, Data Scientists, and Machine Learning Scientists. I have to admit that even though I've been a data science hobbyist for a few years now, I am not entirely clear how these roles differ from one another. In fact, after some research on how tech companies define these roles, I discovered that there are no standardised definitions in the wider industry. This post generates a set of consensus definitions through exploratory data analysis (EDA) and basic modelling of job listings data.

# TL;DR
* In this post, we examine US job listings from [Indeed.com](https://www.indeed.com/) in 2018 to derive a set of **consensus definitions** through exploratory data analysis (EDA) and a classification model.
* EDA enabled us to identify **popular terms** that were used in job listings.
* The classification model enabled us to identify programming languages/tools and skills that **uniquely defined** the various roles.
* The consensus definitions generated (below) should only be used as a **guide or a template to build on**, since our model did not consider how companies *specifically* employ their Data Analysts, Data Engineers, Data Scientists, and Machine Learning Scientists.
  
  
| Role                       | More of...                                                                             | Less of...                                                  |
|----------------------------|----------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Data Analyst               | Data reporting, descriptive data analysis, and basic data management                   | Machine learning and developing / managing production systems |
| Data Engineer              | In-depth data and data infrastructure management, and distributed computing            | Data analysis and prediction                                |
| Data Scientist             | Data wrangling and analysis, algorithm design, and prediction                          | The roles played by Data Analysts and Data Engineers        |
| Machine Learning Scientist  | Deep learning, developing production systems (possibly using distributed computing), and business reporting | Data management and simpler forms of analysis                                   |
  
  
# The Data
We use [this dataset from Kaggle](https://www.kaggle.com/sl6149/data-scientist-job-market-in-the-us), which compiles information on 7,000 data scientists jobs in the US on [Indeed.com](https://www.indeed.com/) in 2018. The data comprises the following:
* `position`: Position title
* `company`: Company
* `description`: Job description
* `reviews`: No. of reviews
* `location`: US state, and postal code (if available)


```python
# Load data
df = pd.read_csv('alldata.csv')

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
      <th>position</th>
      <th>company</th>
      <th>description</th>
      <th>reviews</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Development Director</td>
      <td>ALS TDI</td>
      <td>Development Director\nALS Therapy Development ...</td>
      <td>NaN</td>
      <td>Atlanta, GA 30301</td>
    </tr>
    <tr>
      <td>An Ostentatiously-Excitable Principal Research...</td>
      <td>The Hexagon Lavish</td>
      <td>Job Description\n\n"The road that leads to acc...</td>
      <td>NaN</td>
      <td>Atlanta, GA</td>
    </tr>
    <tr>
      <td>Data Scientist</td>
      <td>Xpert Staffing</td>
      <td>Growing company located in the Atlanta, GA are...</td>
      <td>NaN</td>
      <td>Atlanta, GA</td>
    </tr>
    <tr>
      <td>Data Analyst</td>
      <td>Operation HOPE</td>
      <td>DEPARTMENT: Program OperationsPOSITION LOCATIO...</td>
      <td>44.0</td>
      <td>Atlanta, GA 30303</td>
    </tr>
    <tr>
      <td>Assistant Professor -TT - Signal Processing &amp; ...</td>
      <td>Emory University</td>
      <td>DESCRIPTION\nThe Emory University Department o...</td>
      <td>550.0</td>
      <td>Atlanta, GA</td>
    </tr>
  </tbody>
</table>
</div>


# Data Cleaning

Next, we perform several data cleaning steps. This involved replacement of string terms (text) to:
* Standardise the **job titles** for the following roles:
    * Data Analyst
    * Data Engineer
    * Data Scientist
    * Machine Learning Scientist
* Standardise terms in the **description**, e.g.:
    * Data visuali**s**ation vs. Data visuali**z**ation
    * Analyz**e** data vs. Analyz**ing** data
* Preparing text for feature extraction:
    * Words with one alphanumeric character like R and C++ (programming languages) would be dropped. Thus, these were changed to "RLanguage" and "CPP" for the feature extraction process.
    * Removal of non-english characters

This process was extremely manual, with the text transformations performed in separate lines of code. Since there were over 100 lines, I will not display the code. The final result was a dataset filtered for the above 5 roles, with an additional column (`pos_label`) containing the standardised job titles.


```python
# Consolidate by job
ds_related_jobs = ['data scientist', 'data engineer', 'data analyst', 'machine learning scientist']
df_ds = df[df.pos_label.isin(ds_related_jobs)].reset_index(drop=True)

# Display data
display(df_ds.head())
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
      <th>position</th>
      <th>company</th>
      <th>description</th>
      <th>reviews</th>
      <th>location</th>
      <th>pos_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data Scientist</td>
      <td>Xpert Staffing</td>
      <td>Growing company located in the Atlanta, GA are...</td>
      <td>NaN</td>
      <td>Atlanta, GA</td>
      <td>data scientist</td>
    </tr>
    <tr>
      <td>Data Analyst</td>
      <td>Operation HOPE</td>
      <td>DEPARTMENT: Program OperationsPOSITION LOCATIO...</td>
      <td>44.0</td>
      <td>Atlanta, GA 30303</td>
      <td>data analyst</td>
    </tr>
    <tr>
      <td>Manager of Data Engineering</td>
      <td>McKinsey &amp; Company</td>
      <td>Qualifications Bachelors degree in Computer Sc...</td>
      <td>385.0</td>
      <td>Atlanta, GA 30318</td>
      <td>data engineer</td>
    </tr>
    <tr>
      <td>Senior Associate - Cognitive Data Scientist Na...</td>
      <td>KPMG</td>
      <td>Known for being a great place to work and buil...</td>
      <td>4494.0</td>
      <td>Atlanta, GA 30338</td>
      <td>data scientist</td>
    </tr>
    <tr>
      <td>Senior Associate, Data Scientist</td>
      <td>KPMG</td>
      <td>Innovate. Collaborate. Shine. Lighthouse  KPMG...</td>
      <td>4494.0</td>
      <td>Atlanta, GA 30338</td>
      <td>data scientist</td>
    </tr>
  </tbody>
</table>
</div>
  
  
# Feature Extraction
Next, we create binary features for every term and bigram (two consecutive terms) in the description column using scikit-learn's `CountVectorizer` function. The binary features tell us whether a given term was present in each job description.

The output from this processing step was a data frame with 240,527 binary columns (one per unique word), and one column for our standardised job title (`pos_label`).


```python
# Initialise count vectoriser
cv_desc = CountVectorizer(binary=True, stop_words='english', ngram_range=(1,2))

# Fit and transform descriptions
vec_desc = cv_desc.fit_transform(df_ds.description)

# Convert to data frame
df_desc = pd.DataFrame(vec_desc.toarray(), columns = cv_desc.get_feature_names())

# Add pos_label
df_desc['pos_label'] = df_ds.pos_label.reset_index(drop=True)

# Standardise terms
df_desc['artificial intelligence'] = df_desc['artificial intelligence'] + df_desc['ai']
df_desc['machine learning'] = df_desc['machine learning'] + df_desc['ml']
df_desc['power bi'] = df_desc['power bi'] + df_desc['powerbi']
df_desc = df_desc.drop(['ai', 'ml', 'powerbi'], axis=1)
df_desc = df_desc.rename(columns={'rlanguage': 'r'})

# Consolidate by job
jobs = ['Data Analyst', 'Data Engineer', 'Data Scientist', 'Machine Learning Scientist']
```

# Exploratory Data Analysis (EDA)
Next, we explore the trends for all four roles (Data Analyst, Data Engineer, Data Scientist, and Machine Learning Scientist) collectively and individually.

## Programming Lanuages / Tools
First, we define several popular tools associated with the chosen roles. These include:
* **General-purpose programming languages** like Python, R, and Scala
* **Computing tools** like Spark, AWS, and Hadoop
* **Database tools** like SQL, MongoDB and Hive
* **Machine learning (ML) libraries** like TensorFlow, Scikit-Learn, and Theano
* **Data visualisation tools** like Tableau, Qlik, and Power BI
* I also threw in Excel and Powerpoint for good measure

This list of tools is not exhaustive. I may expand it in future and update the results here.

Next, we plot the proportions of job listings that specifically mentioned the identified tools. The top five are effectively the most popular tools for general purpose programming (Python/R), databases (SQL) and distributed computing (Spark and Hadoop).


```python
# Tools
tools = [
    'python', 'r', 'cpp', 'csharp', 'scala', 'julia', 'java', 'javascript', 'matlab', 'perl',
    'spark', 'azure', 'aws', 'hadoop',
    'hive',  'sql', 'nosql', 'mongodb',
    'tensorflow', 'torch', 'theano', 'scikit',
    'tableau', 'qlik', 'power bi',
    'excel', 'powerpoint'
]

# Filter data
df_tools = df_desc[tools]

# Plot
df_tools.mean().sort_values(ascending=False).plot.bar(figsize=(12,7))
plt.title('Programming Languages/Tools', fontdict=fontdict)
plt.xticks(rotation=45, ha='right')
plt.show()
```


![](../graphics/2020-04-12-extract-defining-data-team-roles/output_13_0.png)


These results could be skewed by the fact that data scientists made up a majority of the roles (75%). Hence, we plot the top 10 programming languages/tools for each role below, separately. We see substantial overlap across the various roles, possibly because the languages/tools are relevant for tasks performed by all roles. We also note the following differences:

* **Excel** and **Powerpoint** were listed frequently in the job descriptions for Data Analysts
* The hardcore **ML libraries** and **low-level programming languages** were only listed in the job descriptions for ML Scientists


![](../graphics/2020-04-12-extract-defining-data-team-roles/output_15_0.png)


## Skills
Second, we perform the same steps for skills. We create a laundry list of data-related skills and plot the proportions of job listings that specifically mentioned these skills. The top five seemed to include prediction-related terms, with machine learning being extremely popular.


```python
# Prepare list
skill_list = [
    'prepare data', 'data preparation', 'data cleaning', 
    'extract insights', 'descriptive analysis', 'predictive analysis', 'predictive analytics',
    'artificial intelligence', 'machine learning', 'deployment', 'deep learning',
    'statistical analysis', 'eda', 'linear algebra', 'learning theory', 'develop algorithms',
    'design algorithms', 'developing algorithms', 'automation', 'computer vision', 'nlp',
    'simulation', 'optimisation', 'data management', 'data quality', 'database management',
    'data infrastructure', 'data pipeline', 'relational database', 'data architecture',
    'data processing', 'production systems', 'data integration', 'data fusion',
    'database administration', 'kpi', 'dashboard', 'business reporting', 'distributed',
    'image processing', 'modelling', 'model evaluation', 'data analysis', 'data wrangling',
    'etl', 'data visualization', 'data warehousing', 'mapreduce', 'software engineering',
    'system design', 'applied mathematics', 'signal processing', 'data reporting',
    'develop reports'
]

# Filter list
skill_list_filtered = [x for x in skill_list if x in df_desc.columns]
```


```python
# Filter columns
df_skills = df_desc[skill_list_filtered]

# Plot
df_skills.mean().sort_values(ascending=False)[:20].plot.bar(figsize=(12,7))
plt.title('Skills - Overall', fontdict=fontdict)
plt.xticks(rotation=45, ha='right')
plt.show()
```


![](../graphics/2020-04-12-extract-defining-data-team-roles/output_18_0.png)


Next, we plot the top 10 skills for each role. Looking purely at the top three skills, we find greater differentiation across the roles. The **Data Analyst** is required to do more of data analysis and visualisation; the **Data Engineer**, managing extract, transform, load (ETL) processes and distributed computing; the **Data Scientist**, prediction; and the **ML Scientist**, Artificial Intelligence (AI) and deep learning.


![png](../graphics/2020-04-12-extract-defining-data-team-roles/output_20_0.png)


# Predicting Job Titles
## Why Bother Modelling?
From EDA, we could see some differences among the various roles. However, this only summarised the **language that hiring managers/recruiters used** in job listings. It is entirely possible that buzzwords like AI and ML were thrown into the mix for the sake of it, or used to describe the broader efforts in the company. This creates noise in the descriptions that prevent us from understanding what the roles *really* entail. To cut through that, we need to identify the **terms unique to each role**. And to achieve this, we use ML modelling.

## The Model
We develop a simple logistic regression model to predict `pos_label` (the roles with standardised job titles) using the binary features (representing terms and bigrams in the description) we created earlier.

Since this was a multi-class problem (four roles = four classes), we tested both the multinomial and One vs. Rest (OvR) strategies. The OvR strategy estimates one logistic regression model per class (say, class X) to predict whether each sample belongs to class X or not. OvR produced better results, and was therefore chosen for the final model.

**Note:** I also tested a gradient boosting (`lightgbm`) and a Random Forest (`sklearn`) model, but the logistic regression model outperformed them.

Finally, we extract the coefficients for each of the four logistic regression models and plot them in a diagram combining all the results we have thus far.


```python
# Prepare data
X_dat = df_desc[tools + skill_list_filtered]
y_dat = df_desc.pos_label

# Fit logistic regression
lr = LogisticRegression(random_state=123)
lr_cv = cross_val_score(lr, X_dat, y_dat, cv=5)
# print('Logistic Regression: %s' % '{:.2f}%'.format(np.mean(lr_cv)*100))

# Coefficients
lr.fit(X_dat, y_dat)
lr_df = pd.DataFrame({'feat': X_dat.columns, 'coef1': lr.coef_[0],
                     'coef2': lr.coef_[1],
                     'coef3': lr.coef_[2],
                     'coef4': lr.coef_[3]})
lr_df.columns = ['feat'] + lr.classes_.tolist()
```

The bottom two plots for each role display the results from the model:

* **What It Is:** The top 15 programming languages/tools ***or*** skills based on model coefficient
* **What It Isn't:** The bottom 15 programming languages/tools ***or*** skills based on model coefficient


![](../graphics/2020-04-12-extract-defining-data-team-roles/output_25_0.png)


![](../graphics/2020-04-12-extract-defining-data-team-roles/output_25_2.png)


![](../graphics/2020-04-12-extract-defining-data-team-roles/output_25_4.png)


![](../graphics/2020-04-12-extract-defining-data-team-roles/output_25_6.png)

    

To summarise the findings:

| Role                       | More of...                                                                             | Less of...                                                  |
|----------------------------|----------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Data Analyst               | Data reporting, descriptive data analysis, and basic data management                   | Machine learning and developing / managing production systems |
| Data Engineer              | In-depth data and data infrastructure management, and distributed computing            | Data analysis and prediction                                |
| Data Scientist             | Data wrangling and analysis, algorithm design, and prediction                          | The roles played by Data Analysts and Data Engineers        |
| Machine Learning Scientist  | Deep learning, developing production systems (possibly using distributed computing), and business reporting | Data management and simpler forms of analysis                                   |
  
  
## Some Interesting Facts

#### There is a technical hierarchy among the roles.
* The Data Analyst role was related to more simpler forms of analysis.
* Only the ML Scientist role was strongly related to hardcore deep learning tools and skills.
* The Data Scientist role fell somewhere between the Data Analyst and ML Scientist role: not deep learning, but not simple analysis.
* Only the Data Engineer role was strongly related to in-depth database-related skills.
* The job listings for the more technical roles (Data Engineer, Data Scientist, and ML Scientist) seem to be averse to Excel, whereas Excel is one of the top-rated tools of a Data Analyst.
  
  
#### There was more noise for less technical positions.
The top 10 skills by popularity did not always appear in the top 15 tools/skills identified from the logistic regression model, especially for the relatively less technical roles (Data Analyst and Data Scientists). It is possible that more fluff was added to beef up the job descriptions for those roles.

#### Python is not listed as one of the core tools of Data Scientists, while R is.
This doesn't mean R is preferred. **In fact, the opposite is true** because Python was listed in 80% of all Data Scientist positions, while R was listed in only 60% of them. Therefore, in technical terms, the binary feature for the term "Python" did not help in differentiating Data Scientist listings from non-Data Scientist listings. Meanwhile, the term "R" appeared to be *relatively* useful in identifying Data Scientists.

The same explanation can be used for why ML is not listed as a core skill for Data Scientists.

# Conclusion
This study was a **broad, cold meta analysis** of the tools and skills that were most associated with the abovementioned roles, and is arguably useful for generating **consensus definitions**. However, we should note that no company-level descriptions of *specifically how they employ each role* was considered. Each organisation has its own priorities, direction, policy, procedures, and culture, all of which determine the actual functions that Data Analysts, Data Engineers, Data Scientists, and ML Scientists play in that specific organisation. Hence, the definitions of roles outlined in this post should only be used as a guide or a template to build on.