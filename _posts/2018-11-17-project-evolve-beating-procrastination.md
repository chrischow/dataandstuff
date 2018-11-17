
# Project *Evolve* - How I'm Beating Procrastination
Many of you will have realised that I post every weekend. If you think that doing so shows how disciplined I am, you're terribly mistaken. Not too long ago, I faced problems with low motivation, poor self-control, and bad time management. I was unsatisfied with how I lived my life. I noticed my bills getting larger, my waist getting bigger, and my energy dropping lower. One day, I had enough of my own nonsense. I was borrowing money, time, and health from my future self at an unsustainable rate, and I decided this had to stop. That's when I started Project Evolve, a self-driven initiative to build willpower and improve my personal effectiveness. This post is for anyone who wants to break out of a cycle of procrastination and sloth, and evolve into a better and stronger version of himself/herself.  
  
# The Story Behind *Evolve*
  
## Downhill
A few months ago, I was unsatisfied with life and with myself. I didn't feel the sense of fulfilment at work that I once had, and felt that I had lost control over my personal life. I procrastinated on almost everything. At work, I failed to manage my tasks well and only *just* met deadlines. In my personal life, the level of initiative wasn't any higher. I procrastinated on ordering a desktop computer for over 6 months, even though it would have greatly enhanced my gaming and programming experience. I procrastinated on bills, even though I knew that these would result in late penalties.  
  
My personal management was positively poor. For example, I let my expenses run wild as I spent way too much money on transport. Despite living relatively far from my workplace, I took Grab to and from work almost every day, racking up hundreds of dollars in bills per month for transport. I also failed to manage my health. I often ate unhealthy food or overate and yes, I grew fatter. I slowed down in personal development as well. I posted inconsistently on my blog, and procrastinated on picking up Python as my new primary language for data science. I was lagging further behind in my reading goal of 30 books this year. It seemed that the only positive point in all this was that I had perfect awareness on what was happening. Yet, I did not have the willpower and the resolve to correct my bad habits and chase my personal goals.  
  
## The Turnaround
As I tried to keep up on reading, I happened to read *The Power of Habit* by Charles Duhigg, which then inspired me to read *The Now Habit* by Neil Fiore, *Superhuman by Habit* by Tynan, and *The Willpower Instinct* by Kelly McGonigal. These books showed me that my problem was no different from that of alcohol and drug addicts: (1) it stemmed from habits and (2) these habits could be changed. At the same time, I grew increasingly sick of myself: my waistline, my bills, my energy level, the state of completion of tasks, and the level of control over my life. The books came at a timely moment and inspired *Project Evolve*.  
  
## *Evolve*
Using my newfound strategies for building willpower and combating procrastination, I developed a system called *Evolve*. It is an app built on [AppSheet](https://www.appsheet.com/) for tracking finances, activities, and habits. You probably don't understand what tracking myself had to do with willpower or procrastination. And so, let's dive into the science behind *Evolve* first.  
  
# The Science Behind Evolve
  
## Habits
In *The Power of Habit*, Charles Duhigg defines habits as "choices that we deliberately make at some point, and then stop thinking about, but continue doing, often every day". They are made up of cycles comprising a *Cue*, a *Routine*, and a *Reward*. When you see the *Cue*, you execute the *Routine* in order to get the *Reward*. To transform habits, you need to keep the Cue, keep the Reward, change the Routine, and *truly believe* that change is possible.  
  
For example, I had the habit of playing games for many hours at a stretch. I could not limit myself. I identified the cue as boredom and the need for intellectual engagement, the routine as gaming, and the reward as intellectual stimulation. Now, when I notice the cue (boredom), I adopt a different routine. If it is high-intensity intellectual engagement that I need, I dive into programming on a topic of interest (at the moment, finance). If it is low-intensity intellectual engagement that I'm looking for in that moment, I grab my Kindle and read. The end result is still the same: my brain gets some action.
  
## Keystone Habits
Duhigg argues that it is important to develop Keystone Habits and build willpower. Keystone habits are core routines that, when established, help all other routines fall in place. When keystone habits are maintained, we tend to also maintain other good habits. For example, cultivating a habit of exercising helps us to eat healthier. He writes about two Australian researchers, Megan Oaten and Ken Cheng, who created a willpower workout to help children lose weight. After forcing the experiment's participants to undergo a physical exercise programme, they found that the participants became more disciplined overall. They smoked less, drank less, and watched less TV. To eliminate the effect of better health on willpower, they instructed a different set of participants to log their diets instead - a programme that did not necessarily guarantee improvement in health. Yet, they found similar results! To confirm the impact of greater self-discipline in one area of participants' lives on other areas, they instructed yet a different set of participants to log their finances. They observed similar results. The finding? When willpower improved, it improved in many areas of the participants’ lives. Coincidentally, Neil Fiore's system - The Now Habit - recommends exactly that: logging everything you do. The log shows you how much time you spent on productive work, guilt-free leisure, and procrastination.  
  
## Small Wins
Duhigg also writes about achieving small wins to build confidence. This is supported by Kelly McGonigal's findings in *The Willpower Instinct*. She writes about willpower as a muscle that can be trained. The same brain muscle that forces you to break one bad habit is the same muscle that will help you to break others. She confirms Duhigg's idea of keystone habits that establishing new habits in just one area of our lives can spillover to both *other parts* of our lives and *others'* lives. If the brain is a muscle, self-control is exercise, and small wins are endorphins.  
  
The evidence on building keystone habits and achieving small wins was convincing. I figured that I had to try it for myself, and that was how Evolve was born.  
  
# *Evolve*
*Evolve* is simply the application of Duhigg's, Fiore's and McGonigal's ideas: to deliver **small wins** and kickstart **willpower development** for transformative change. It is an app that enables you to track your finances, activities, and habits, and helps you to cultivate self-awareness. By tracking how much money and time you spend, you gain an awareness of how you're spending your precious resources, and can then optimise them. By tracking your keystone habits, you inevitably gain awareness on how you prioritise. In fact, the act of logging is itself a keystone habit.  
  
Below are several screenshots of the app (built in [AppSheet](https://www.appsheet.com/)):

<img src = "Screenshot 1 - Activity Summary.png" width = "400" alt = "Summary of activities">
  
<img src = "Screenshot 2 - Log Activity.png" width = "400" alt = "Logging activities">

I understand that we might be hesitant to partake in any programme that helps us to achieve self-awareness, possibly because we're afraid of what we might find. That's perfectly normal. Until today, I don't dare to tabulate the total amount I spent on Grab. But, at some point in our personal development, we need to give permission to an objective third-party to tell us how we've actually been doing. And this third-party need not be a person - it can be an app like *Evolve*. And there's no better way to demonstrate this than to lead by example.  
  
The rest of this post will be dedicated to exploring activity data that I logged over the past month.


```python
# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
import pandas as pd
import warnings

# Settings
warnings.filterwarnings('ignore')

# Read data
evolve = pd.read_csv('Log - Activities.csv')

# Convert Career Prep
evolve.Category[evolve.Category == 'Career Prep'] = 'Personal Development'

# Convert dates and times
evolve['Time Start'] = pd.to_datetime(evolve['Time Start'], format='%d/%m/%Y %H:%M:%S')
evolve['Time End'] = pd.to_datetime(evolve['Time End'], format='%d/%m/%Y %H:%M:%S')

# Get date
evolve['Day'] = evolve['Time Start'].dt.date
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

# Overview of Time Spent
I developed 11 broad categories to classify my time:  
  
1. **Rest:** Sleep and resting
2. **Sustenance:** Eating and body maintenance
3. **Work:** Self-explanatory
4. **Family:** All activities spent with family and maintaining the household
5. **Personal Development:** Reading, meditating, programming, and blogging
6. **Relationships:** Time spent with friends, which include meet-ups, chatting and texting
7. **Idle:** My favourite category - includes gaming, listening to music, stoning, and watching videos or TV
8. **Health:** Exercise and researching on health matters
9. **Productivity:** Planning, organising, and developing tools for productivity  
  
The donut chart below shows how I've spent my time over the past 30 days. Apart from sleeping, I invested most of my time in work and personal development. I actually spent more time idling (mostly gaming, really) than on household matters, and this is something I hope to change. Let's dive deeper.


```python
# Calculate hours
evolve['hours'] = evolve.Duration / 60

# Configure labels
labs = evolve.groupby('Category').hours.sum().index + ' - ' + \
    (evolve.groupby('Category').hours.sum() / evolve.hours.sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(evolve.groupby('Category').hours.sum(), labels = labs, colors = sns.color_palette("Set2", 11))
plt.title('Breakdown of Overall Time Spent', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
```


![png](output_5_0.png)


# Expectations vs. Reality
Before I started this analysis, I forced myself to guess how much time I spent on sleeping, working, personal development, and idling.  
  
* **Sleep:** I sleep late on weekdays and even later on weekends. My wife and I both felt that I did not sleep enough. Hence, I estimated that I slept *7 hours* a day on average.
* **Work:** Based on gut feel, it felt as if I worked *8 hours* a day on average.
* **Personal Development:** Based on my estimate of 1 hour of reading (on the train) and about 1 hour on programming everyday, I estimated that I spent an average of *2 hours* a day on personal development.
* **Idling:** Once again, based on gut feel, I estimated that I spent *3 hours* on unproductive activities everyday, on average.
  


```python
# Daily data
evolve_daily = pd.DataFrame(evolve.groupby(['Category', 'Day'])['Duration'].sum()).reset_index(drop = False)
evolve_daily_sub = pd.DataFrame(evolve.groupby(['Category', 'Sub-Category', 'Day'])['Duration'].sum()).reset_index(drop = False)

# Targets
tgt = pd.DataFrame([
    {'Category': 'Rest', 'Duration': 7},
    {'Category': 'Work', 'Duration': 8},
    {'Category': 'Personal Development', 'Duration': 2},
    {'Category': 'Idle', 'Duration': 3}
]).set_index('Category')

# Actual data
actual = (evolve_daily.groupby('Category').Duration.mean().sort_values(ascending = False)/60).loc[['Rest', 'Work', 'Personal Development', 'Idle']]
```


```python
# Create bars
bar_actual = actual.values
bar_tgt = tgt.Duration.values
ht_actual = np.arange(len(bar_actual))
ht_tgt = [x + 0.4 for x in ht_actual]

# Plot
plt.figure(figsize = (10, 8))
plt.bar(ht_actual, bar_actual, color='#133056', width=0.4, edgecolor='white', label='Actual')
plt.bar(ht_tgt, bar_tgt, color='#6fceb0', width=0.4, edgecolor='white', label='Estimate')
plt.xticks([x +0.2 for x in ht_actual], ['Rest', 'Work', 'Personal Development', 'Idle'])
plt.title('Estimated vs. Actual Time Spent on Activities', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.ylabel('Hours')
for i in range(len(ht_actual)):
    plt.text(x = ht_actual[i]-0.05, y = bar_actual[i]+0.2, s = bar_actual[i].round(2))
    plt.text(x = ht_tgt[i], y = bar_tgt[i] + 0.2, s = bar_tgt[i].round(2))
plt.legend()
plt.show()
```


![png](output_8_0.png)


The data suggests that my estimates were rather inaccurate! On average, I slept approximately **9 hours**, worked **9.3 hours** (after adjusting for weekends), spent **3.4 hours** on personal development, and wasted only **2 hours** a day. This means that:  
  
1. Neither my wife nor I can complain about me lacking sleep.
2. Assuming a typical work day of 8 am to 6 pm (10 hours), I spend 93% of my work time productively!
3. I spend more time on personal development than I think - I should keep it up and not rest on my laurels.
4. I spend an awful lot of time idling. I need to know why!
  

# Exploring My Idle Time
To reduce my idle time, I need to know specifically how I've been idling. As predicted, more than half of my idle time was spent on gaming (CSGO, to be specific). In case you're wondering, I chose to put gaming under the "Idle" category because my long-term aim is to replace gaming with programming/blogging as my primary source of leisure. However, Neil Fiore recommends that we **plan** for leisure and down time so that we can enjoy these without guilt. Perhaps I'll implement this in the next iteration of *Evolve*.


```python
# Extract overall idle time
idle_full = evolve[evolve.Category == 'Idle']

# Configure labels
labs = idle_full.groupby('Sub-Category').hours.sum().index + ' - ' + \
    (idle_full.groupby('Sub-Category').hours.sum() / idle_full.hours.sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(idle_full.groupby('Sub-Category').hours.sum(), labels = labs, colors = sns.color_palette("Set2", 6))
plt.title('Breakdown of Time Spent Idling', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
```


![png](output_11_0.png)


The bar plot below shows that Tuesdays, Thursdays, Fridays, and Saturdays are the days that I spend the most time idling, on average.


```python
# Weekday
idle_daily = evolve_daily[evolve_daily.Category == 'Idle'].copy()
idle_daily['Weekday'] = pd.to_datetime(idle_daily.Day).dt.weekday

# Plot
plt.figure(figsize = (10, 8))
plt.title('Average Time Spent Idling', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.bar(np.arange(len(idle_daily.Weekday.unique())), idle_daily.groupby('Weekday').mean().values / 60, color = '#b1ceeb')
plt.xticks(np.arange(len(idle_daily.Weekday.unique())), ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
plt.ylim(0, 3.5)
plt.ylabel('Hours')
plt.xlabel('Weekday')
plt.show()
```


![png](output_13_0.png)


Upon closer examination of my idle time on these days, I discovered that I tended to watch more videos on Tuesdays, game more on Thursdays, use my phone more on Fridays, and watch a lot more TV (probably football) on Saturdays than usual. I will have to watch these habits closely if I want to control my time spent on unproductive uses. Alternatively, I could re-define what "unproductive" means. After all, leisure is meant to be guilt-free. What's important is tracking the time I spend **procrastinating**.


```python
# Add weekday
idle_daily_sub = pd.DataFrame(evolve.groupby(['Category', 'Sub-Category', 'Day'])['Duration'].sum()).reset_index(drop = False).copy()
idle_daily_sub = idle_daily_sub[idle_daily_sub.Category == 'Idle']
idle_daily_sub['Weekday'] = pd.to_datetime(idle_daily_sub['Day']).dt.weekday
idle_daily_sub['hours'] = idle_daily_sub.Duration / 60

# Check idle activities on Tuesdays
idle_tue = idle_daily_sub[idle_daily_sub.Weekday == 1].copy()

# Configure labels
labs = idle_tue.groupby('Sub-Category').hours.sum().index + ' - ' + \
    (idle_tue.groupby('Sub-Category').hours.mean() / idle_tue.groupby('Sub-Category').hours.mean().sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(idle_tue.groupby('Sub-Category').hours.mean(), labels = labs, colors = sns.color_palette("Set2", 4))
plt.title('Breakdown of Time Spent Idling on Tuesdays', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# Check idle activities on Thursdays
idle_thu = idle_daily_sub[idle_daily_sub.Weekday == 3].copy()

# Configure labels
labs = idle_thu.groupby('Sub-Category').hours.sum().index + ' - ' + \
    (idle_thu.groupby('Sub-Category').hours.mean() / idle_thu.groupby('Sub-Category').hours.mean().sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(idle_thu.groupby('Sub-Category').hours.mean(), labels = labs, colors = sns.color_palette("Set2", 4))
plt.title('Breakdown of Time Spent Idling on Thursdays', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# Check idle activities on Fridays
idle_fri = idle_daily_sub[idle_daily_sub.Weekday == 4].copy()

# Configure labels
labs = idle_fri.groupby('Sub-Category').hours.sum().index + ' - ' + \
    (idle_fri.groupby('Sub-Category').hours.mean() / idle_fri.groupby('Sub-Category').hours.mean().sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(idle_fri.groupby('Sub-Category').hours.sum(), labels = labs, colors = sns.color_palette("Set2", 4))
plt.title('Breakdown of Time Spent Idling on Fridays', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# Check idle activities on Saturdays
idle_sat = idle_daily_sub[idle_daily_sub.Weekday == 5].copy()

# Configure labels
labs = idle_sat.groupby('Sub-Category').hours.mean().index + ' - ' + \
    (idle_sat.groupby('Sub-Category').hours.mean() / idle_sat.groupby('Sub-Category').hours.mean().sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(idle_sat.groupby('Sub-Category').hours.sum(), labels = labs, colors = sns.color_palette("Set2", 4))
plt.title('Breakdown of Time Spent Idling on Saturdays', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
```


![png](output_15_0.png)



![png](output_15_1.png)



![png](output_15_2.png)



![png](output_15_3.png)


# Exploring My Personal Development Time
Overall, my three key activities for personal development have been Python programming, reading, and blogging. The graph below shows the breakdown of time spent on each:


```python
# Extract overall idle time
pd_full = evolve[evolve.Category == 'Personal Development']

# Configure labels
labs = pd_full.groupby('Sub-Category').hours.sum().index + ' - ' + \
    (pd_full.groupby('Sub-Category').hours.sum() / pd_full.hours.sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(pd_full.groupby('Sub-Category').hours.sum(), labels = labs, colors = sns.color_palette("Set2", 7))
plt.title('Breakdown of Time Spent on Personal Development', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
```


![png](output_17_0.png)


I've spent approximately 6 hours on personal development on the weekends, and 1-3 hours on weekdays. There is a noticeable drop in time spent on personal development on Fridays, possibly because I use Fridays to unwind, and end up slacking off a little.


```python
# Weekday
pd_daily = evolve_daily[evolve_daily.Category == 'Personal Development'].copy()
pd_daily['Weekday'] = pd.to_datetime(pd_daily.Day).dt.weekday

# Plot
plt.figure(figsize = (10, 8))
plt.title('Average Time Spent on Personal Development', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.bar(np.arange(len(pd_daily.Weekday.unique())), pd_daily.groupby('Weekday').mean().values / 60, color = '#b1ceeb')
plt.xticks(np.arange(len(pd_daily.Weekday.unique())), ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
plt.ylim(0, 7)
plt.ylabel('Hours')
plt.xlabel('Weekday')
plt.show()
```


![png](output_19_0.png)


Examining my weekend personal development time further, I noticed that my weekly rhythm involves cleaning up my weekly post on Saturdays, and researching for the following week's post on Sundays. I also spent a lot less time reading on weekends, which implies inconsistent reading. I could improve by catering more time for reading on the weekends, and push some programming work to weekday nights.


```python
# Add weekday
pd_daily_sub = pd.DataFrame(evolve.groupby(['Category', 'Sub-Category', 'Day'])['Duration'].sum()).reset_index(drop = False).copy()
pd_daily_sub = pd_daily_sub[pd_daily_sub.Category == 'Personal Development']
pd_daily_sub['Weekday'] = pd.to_datetime(pd_daily_sub['Day']).dt.weekday
pd_daily_sub['hours'] = pd_daily_sub.Duration / 60

# Check idle activities on Tuesdays
pd_sat = pd_daily_sub[pd_daily_sub.Weekday == 5].copy()

# Configure labels
labs = pd_sat.groupby('Sub-Category').hours.sum().index + ' - ' + \
    (pd_sat.groupby('Sub-Category').hours.mean() / pd_sat.groupby('Sub-Category').hours.mean().sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(pd_sat.groupby('Sub-Category').hours.mean(), labels = labs, colors = sns.color_palette("Set2", 4))
plt.title('Breakdown of Time Spent on Personal Devt on Saturdays', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# Check idle activities on Thursdays
pd_sun = pd_daily_sub[pd_daily_sub.Weekday == 6].copy()

# Configure labels
labs = pd_sun.groupby('Sub-Category').hours.sum().index + ' - ' + \
    (pd_sun.groupby('Sub-Category').hours.mean() / pd_sun.groupby('Sub-Category').hours.mean().sum() * 100).round(2).astype(str) + '%'

# Plot
plt.figure(figsize = (10,10))
plt.pie(pd_sun.groupby('Sub-Category').hours.mean(), labels = labs, colors = sns.color_palette("Set2", 6))
plt.title('Breakdown of Time Spent on Personal Devt on Sundays', fontdict = {'fontweight': 'bold', 'fontsize': 20})
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
```


![png](output_21_0.png)



![png](output_21_1.png)


# Exploring Health
Next, we examine the time I've spent looking after my body. First, we see that my meal timings have not been very consistent. I usually have my lunch within a 3-hour block from 11 am to 2 pm, and dinner within a 2-hour block from 6 pm to 8 pm. The data also shows that I don't eat breakfast consistently.


```python
# Extract eating time
bf_data = evolve[(evolve['Sub-Category'] == 'Eating') & evolve.Remarks.str.contains('Breakfast')].copy()
lunch_data = evolve[(evolve['Sub-Category'] == 'Eating') & evolve.Remarks.str.contains('Lunch')].copy()
dinner_data = evolve[(evolve['Sub-Category'] == 'Eating') & evolve.Remarks.str.contains('Dinner')].copy()

# Compute time
bf_data['time'] = bf_data['Time Start'].dt.time
lunch_data['time'] = lunch_data['Time Start'].dt.time
dinner_data['time'] = dinner_data['Time Start'].dt.time

# Remove outlier
lunch_data.drop(420, axis = 0, inplace = True)

# Get mean breakfast time
bf_times = pd.to_datetime(bf_data['Time Start'])
# pd.to_timedelta(int((bf_times.dt.hour*3600+bf_times.dt.minute*60+bf_times.dt.second).mean()),unit='s')

# Get mean lunch time
lunch_times = pd.to_datetime(lunch_data['Time Start'])
# pd.to_timedelta(int((lunch_times.dt.hour*3600+lunch_times.dt.minute*60+lunch_times.dt.second).mean()),unit='s')

# Get mean dinner time
dinner_times = pd.to_datetime(dinner_data['Time Start'])
# pd.to_timedelta(int((dinner_times.dt.hour*3600+dinner_times.dt.minute*60+dinner_times.dt.second).mean()),unit='s')
```


```python
# Plot mealtimes
plt.figure(figsize = (10, 8))
plt.plot(bf_data.Day, bf_data.time, label = 'Breakfast', color = '#ff9966')
plt.plot(bf_data.Day, bf_data.time, 'rD', color = '#ff9966', markersize = 4)
plt.plot(lunch_data.Day, lunch_data.time, label = 'Lunch', color = '#133056')
plt.plot(lunch_data.Day, lunch_data.time, 'rD', color = '#133056', markersize = 4)
plt.plot(dinner_data.Day, dinner_data.time, label = 'Dinner', color = '#b1ceeb')
plt.plot(dinner_data.Day, dinner_data.time, 'rD', color = '#b1ceeb', markersize = 4)
plt.hlines(y = '08:31:32', xmin = lunch_data.Day.min(), xmax = lunch_data.Day.max(), color = '#133056', linewidth = 1.5, linestyle = 'dashed')
plt.hlines(y = '12:33:24', xmin = lunch_data.Day.min(), xmax = lunch_data.Day.max(), color = '#6fceb0', linewidth = 1.5, linestyle = 'dashed')
plt.hlines(y = '19:03:48', xmin = dinner_data.Day.min(), xmax = dinner_data.Day.max(), color = '#f85b74', linewidth = 1.5, linestyle = 'dashed')
plt.legend()
plt.title('Chow Times', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.ylabel('Time of Day')
plt.xlabel('Date')
plt.show()
```


![png](output_24_0.png)


Next, we look at exercise. Prior to starting the *Evolve* regime, I didn't spend any time on exercising. As I tracked my time, I noticed the lack of any effort in investing in health (and also, my body decided that tight-fit clothing was in fashion). Hence, I resolved to spend some time every day exercising. It could be 10 minutes of push ups or a 20-minute run - it didn't matter how long, as long as I was active for a while. As you can see, I'm still working on that.


```python
# Extract eating time
ex_data = evolve_daily[evolve_daily['Category'] == 'Health'].copy()

plt.figure(figsize = (10, 8))
plt.plot(pd.to_datetime(ex_data.Day), ex_data.Duration, color = '#133056')
plt.xlabel('Date')
plt.ylabel('Minutes')
plt.xticks(rotation = '45')
plt.title('Time Spent on Exercise', fontdict = {'fontweight': 'bold', 'fontsize': 20})
plt.show()
```


![png](output_26_0.png)


# A Final Note on the Data
The data contained all my activities from 9 Oct 18 to 11 Nov 18, which was about a month's worth of data. That means that the recommendations from the simple analysis in this post may not be well-grounded because there was insufficient data. However, it shows the potential of activity and finance tracking to help you optimise your life and gain awareness of how you're spending your time and money. Personally, I feel motivated from both achieving things I want (more exercise, personal development) and knowing where I can improve (exercise, idle time).

# Moving Forward
I plan to stay on this regime for 1 year. Hopefully, I will have good things to review next December. I also plan to release the *Evolve* app for free to enable anyone who wants a go at tackling procrastination in their lives to take the first step toward *evolution*. Stay tuned for more updates.
