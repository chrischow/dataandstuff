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
The competition was run on [Kaggle](http://www.kaggle.com), a platform for hosting data science competitions. Submissions to the NDSC on Kaggle required us to predict the product attributes on a mix of approximately 1,000,000 beauty, fashion, and mobile products. Our leaderboard score was based on the number of correct predictions, with a twist. The evaluation metric was called Mean Average Precision @ 2 (MAP@2). We were allowed two predictions per product, and were given a score of `1.0` if the first prediction was correct, regardless of the second prediction, and a score of `0.5` if the first prediction was wrong, but the second was correct. Therefore, we had to use a machine learning (ML) algorithm that provided us with an **accurate measure of the most probable classes**.  
  
You can think of the competition as an exam. The training set is equivalent to a 10-year Series booklet of past exam questions. The test set is equivalent to a milestone exam like the PSLE or O Levels. In building a model, you hope to learn the underlying principles of how to solve exam questions (using the 10-year Series) so your skills are applicable to the exam. If you instead memorise solutions to the 10-year Series questions, you might not do well on the exam because minor variations in questions will throw you off.
  
## The Outcome
Jumping forward in time, here's the result: **we placed 4th/5th on the Kaggle leaderboard and won the Merit Prize!** I was slightly disappointed we didn't make a podium finish, but I was glad we were close. 




## The Solution
Given the outcome, you would expect a pretty complex solution, but ours was extremely simple. We used basic data cleaning techniques, a primitive method for feature extraction, and a primitive ML algorithm for classification. All we needed was good intuition in the models. Let's begin with data cleaning.  
  
### Data Cleaning
For each model, we used only the product listing titles (hereafter referred to as titles), ignoring the images completely. This was because they were non-standardised and would have contributed to a lot of noise in the models. In addition, it would have required a lot of time to train neural networks on the images. Hence, we focused only on text. We did just two things: (1) dropped missing values and (2) translated *some* Bahasa Indonesian words to English. We translated only words that were related to the product attributes we were predicting. For example, colours were often written in Bahasa Indonesian in the titles, while the labels for the Colour Family product attribute were in English. Simple stuff.  
  
It is worth noting that I missed out on the translations until the very end of the competition. On the final day, we were in 9th place.  However, after performing translations (1) on the beauty and fashion datasets (which we completely missed out) and (2) on the test set titles, we parachuted into 4th place. The score jumped from 46.5% to 45.8%, which is quite a lot when you're near the top of the laderboard. 
  
### Feature Extraction
We converted titles (from both the training and test set) and product attribute labels (hereafter referred to as labels) into binary features a.k.a binary term frequency (BTF) using Scikit-Learn's `TfidfVectorizer` function. Each binary feature indicated whether a specific N-gram (word or phrase) was present in that title. Therefore, the label BTFs indicated **matches between the titles and the correct label**, while the title BTFs served as a **bank of vocabulary**. The former was used to capture more straightforward relationships between the titles and their true labels, while the latter was used to capture more complex relationships between the text in the titles and their true labels.  
  
### ML Algorithm
We chose Logistic Regression as our classification algorithm. Specifically, we used a One vs. Rest (OVR) approach for developing models. That is, assuming the target product attribute had *k* classes, we split the problem into *k* binary sub-problems. Each sub-problem predicted whether each entry belonged to class *K* or not. Scikit-Learn's `LogisticRegression` class was able to handle this with a single parameter. We performed basic fine tuning of the `C` parameter for regularisation to ensure that the models did not overfit to the training data.  
  
### Results
We fit all of the above into a ML pipeline using Scikit-Learn's `Pipeline` class. This ensured that there was no leakage, and facilitated cross validation (CV). We used 5-fold CV to evaluate our pipeline, and obtained MAP@2 scores between 85% and 99%, depending on the target product attribute. On the test set, we obtained a score of only 46%. I suspect this was because:  
  
1. There was a lot of missing data in the test set (likewise for the training set), resulting in many ungraded (`0.0`) scores.
2. We did not use a sufficiently large bank of vocabulary. Consequently, new words in the titles would not have been picked up.
3. The distribution of product attribute classes was different in the test set. For product attributes with a large number of classes e.g. phone model for the mobile dataset and brand for the beauty dataset, certain classes may not have appeared in the training set, but were present in the test set.
4. Many samples were incorrectly labelled. We found cases like iPhones labelled with the Android operating system. This would have thrown the models off, but we chose not to correct it for fear of violating the "no hand labelling" rule in the competition.  
  
### Further Improvements
First, there was nothing much we could do about missing data in the test set. We chose not to include a "missing" class in training, because providing no prediction is alwasy bad when there is a correct answer. Any answer would stand a better chance than no answer.  
  
Second, we expanded the bank of vocabulary by using (1) **all** titles in the training set, even for samples that had missing values in the target product attribute, and (2) titles in the test set. This did not introduce leakage because the test set was not labelled. If we had more time, we would have scraped titles from their online site ourselves.  
  
Third, we chose not to pursue zero shot learning (classes appearing in the test set and not in the training set) because we did not have the time and resources to do so. However, we explored certain techniques that could have addressed this issue, except that I employed the techniques wrongly. More on this in the next section.  
  
Fourth, we chose not to correct this because errors were likely to be present in the test set as well. We did not know how the models would fare on the test set if we trained them on perfect data. Perhaps there is a better technique for handling this issue, but I have yet to find any solution at the time of writing.  
  
## Things That Did Not Work
On hindsight, the solution was straightforward: it was a textbook text classification approach. However, the bulk of the work was in testing other methods for feature extraction and other ML algorithms and techniques. All in all, we tested the following feature extraction techniques and ML algorithms:  
  
| Features Extracted                                                     | Algorithm                    |
|------------------------------------------------------------------------|------------------------------|
| TF-IDF of Titles                                                       | Logistic Regression          |
| BTF of Titles                                                          | Random Forest (Scikit-Learn) |
| TF-IDF of Labels                                                       | Random Forest (LightGBM)     |
| BTF of Labels                                                          | XGBoost                      |
| Doc2Vec of Titles (50 features)                                        | Gradient Boosting (LightGBM) |
| Doc2Vec of Titles (100 features)                                       | Support Vector Machines      |
| Doc2Vec of Titles (200 features)                                       | Multinomial Naive Bayes      |
| Doc2Vec of Titles (500 features)                                       | K-Nearest Neighbours         |
| Cosine Similarity to all classes for Doc2Vec 50, 100, 200, and 500     |                              |
| K-Means Clustering for Cosine Similarity and Doc2Vec 50, 100, 200, 500 |                              |
  
We also attempted stacking and blending (ensemble techniques), but these failed to beat the good old Logistic Regression. It was the simplest, quickest, and best-performing model.  
  
### Key Lessons Learnt
  
#### Doc2Vec
I have to admit that I employed Doc2Vec wrongly. I trained our Doc2Vec models on the training set titles instead of using a pre-trained Doc2Vec model. Models pre-trained on large corpuses (e.g. Google News) are available online, but I did not know how to implement them at the time. These models are able to recognise the difference between, say, Samsung and Apple, or iPhone X and Galaxy S9. If I had implemented Doc2Vec correctly, we could have possibly created better Doc2Vec features. I'm happy that I learned how not to mess up Doc2Vec for future NLP tasks.  
  
#### Deep Learning
Frustrated with our low ranking toward the end of the competition, I researched on modern text classification techniques. I discovered that the latest technique for text classification is to feed word embeddings (from models like Word2Vec, GloVe, or fastText) into a neural network (NN). With no prior knowledge in training NNs, I avoided the approach entirely. However, I am now convinced that to progress in data science, I must venture into deep learning. My ignorance in this area may have cost us a better ranking.  
  
#### Try, Try, and Try Again
My biggest challenge during this period was to manage my frustration. I came up with idea after idea, and tested them only to see them fail in improving our score. 
  
> "I have not failed 10,000 times. I have successfully found 10,000 ways that will not work." - Thomas Edison  
  


## Other Notes on the Competition
  
### Leakage
During the course of the competition, there were **two instances** of leakage - parts of the fashion dataset "exam questions" were accidentally released. Twice. This required Shopee to change the test set twice, thereby wasting the efforts of the participants who had developed models for the fashion products. The first leakage was detected just a few days into the competition. The entire set of answers to the fashion dataset "exam questions" were uploaded along with the training sets. The second leakage was only detected three days before the end of the competition. Participants who were banking on a high score in the fashion product attributes would have seen a big change in their ranking. I think that this was extremely unprofessional on Shopee's part. It shows how serious they really are about managing their data and levelling up their data science capability.  
  
### 




















