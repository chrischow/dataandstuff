# 4/5th Place Solution for the National Data Science Challenge
## Joining the National Data Science Challenge
It's been a crazy month. 9 months ago, I committed to having a baby, and Baby Ethan was born on 11 Mar 19! He has brought me so much joy, even through crying and pooping. Between the nappy changes and feedings, I committed to another baby: the National Data Science Challenge (NDSC) 2019, which was held from 23 Feb 19 to 22 Mar 19. I felt that participating in the NDSC was essential for me to develop my skills in data science and benchmark myself against other data scientists in Singapore. Hence, I teamed up with 3 ex-seniors from the University of Warwick to take a swing at the NDSC.  
  
## The Task
  
### Problem Statement
The NDSC was run by Shopee, who provided us with product listing titles and images on beauty, fashion, and mobile products. The task at hand was to predict attributes of these products (see the table below for the full list). Given that the target attributes were categorical and text data and images were the predictors, this was clearly a **classification problem that required Natural Language Processing (NLP) and image processing techniques**.  
  
| Beauty Product Attributes | Fashion Product Attributes | Mobile Product Attributes |
|---------------------------|----------------------------|---------------------------|
| Benefits                  | Pattern                    | Operating System          |
| Brand                     | Collar Type                | Features                  |
| Colour Group              | Fashion Trend              | Network Connections       |
| Product Texture           | Clothing Material          | Memory RAM                |
| Skin Type                 | Sleeves                    | Brand                     |
|                           |                            | Warranty Period           |
|                           |                            | Storage Capacity          |
|                           |                            | Colour Family             |
|                           |                            | Phone Model               |
|                           |                            | Camera                    |
|                           |                            | Phone Screen Size         |
  
Furthermore, each product attribute required dedicated treatment. Hence, the task required us to optimise **21 different models** - one per product attribute. This implied a lot of work. When we were first briefed on the task, we were uninspired by the lack of social impact and the sheer amount of time and resources it would take to build 21 models. We actually considered dropping out from the NDSC at our first (and only) team meeting. Fortunately, one of my teammates, Shen Ting, was pretty enthusiastic and ran a Random Forest (RF) model on *default settings* on a random mobile product attribute. It scored over 90% in accuracy. I don't know about the other teammates, but this inspired me to give the NDSC a go.  
  
### The Competition
The competition was run on [Kaggle](http://www.kaggle.com), a platform for hosting data science competitions. Submissions to the NDSC on Kaggle required us to predict the product attributes on a mix of approximately 1,000,000 beauty, fashion, and mobile products. Our leaderboard score was based on the number of correct predictions, with a twist. We were allowed two predictions per product, and were given a score of `1.0` if the first prediction was correct, regardless of the second prediction, and a score of `0.5` if the first prediction was wrong, but the second was correct. Therefore, we had to use a machine learning (ML) algorithm that provided us with an **accurate measure of the most probable classes**.  
  
You can think of the competition as an exam. The training set is equivalent to a 10-year Series booklet of past exam questions. The test set is equivalent to a milestone exam like the PSLE or O Levels. In building a model, you hope to learn the underlying principles of how to solve exam questions (using the 10-year Series) so your skills are applicable to the exam. If you instead memorise solutions to the 10-year Series questions, you might not do well on the exam because minor variations in questions will throw you off.
  
## The Outcome
Jumping forward in time, here's the result: **we placed 4th/5th on the Kaggle leaderboard and won the Merit Prize!** I was slightly disappointed we didn't make a podium finish, but I was glad we were close. 




## The Solution
Our solution was extremely simple. We used basic data cleaning techniques, a primitive method for feature extraction, and a primitive ML algorithm for classification. Let's begin with data cleaning.  
  
### Data Cleaning
I'd like to point out first that we completely ignored the images. This was because they were non-standardised and would have contributed to a lot of noise in the models. In addition, it would have required a lot of time to train neural networks (NNs) on the images. Hence, we focused only on text.  
  
For each model, we used only the product listing titles (hereafter referred to as titles). We did just two things: (1) dropped missing values and (2) translated *some* Bahasa Indonesian words to English. We translated only words that were related to the product attributes we were predicting. For example, colours were often written in Bahasa Indonesian in the titles, while the labels for the Colour Family product attribute were in English. Simple stuff.  
  
### 


> "I have not failed 10,000 times. I have successfully found 10,000 ways that will not work." - Thomas Edison

## The Numerous Failures
The solution was straightforward. However, the bulk of the work was in testing other methods for feature extraction and other ML algorithms and techniques.






















