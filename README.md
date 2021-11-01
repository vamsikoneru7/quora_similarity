![GitHub last commit](https://img.shields.io/github/last-commit/vamsikoneru7/quora_similarity)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vamsikoneru7/quora_similarity/HEAD)

# Quora Question Pair Similarilty
## Buisiness Problem
### Description

Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

#credits: Kaggle

#### Problem Statement

* Identify which questions asked on Quora are duplicates of questions that have already been asked.
* This could be useful to instantly provide answers to questions that have already been answered.
* We are tasked with predicting whether a pair of questions are duplicates or not.

### Sources/Useful Links
Source : https://www.kaggle.com/c/quora-question-pairs

Useful Links
* Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
* Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
* Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
* Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

### Real world/Business Objectives and Constraints
* The cost of a mis-classification can be very high.
* You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
* No strict latency concerns.
* Interpretability is partially important.
## Machine Learning Probelm
### Data Overview
- Data will be in a file Train.csv
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290
### Example Data point
|"id"|"qid1"|"qid2"|"question1"|"question2"|"is_duplicate"|
|----|------|------|-----------|-----------|--------------|
|0|1|2|What is the step by step guide to invest in share market in india?|What is the step by step guide to invest in share market?|0|
|1|3|4|What is the story of Kohinoor (Koh-i-Noor) Diamond?|What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?|0|
|7|15|16|How can I be a good geologist?|What should I do to be a great geologist?|1|
|11|23|24|How do I read and find my YouTube comments?|How can I see all my Youtube comments?|1|

### Mapping the real world problem to an ML problem
#### Type of Machine Leaning Problem
It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.

#### Performance Metric
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

#### Metric(s):
- log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
- Binary Confusion Matrix
#### Train and Test Construction
We build train and test by randomly splitting in the ratio of **70:30** or **80:20** whatever we choose as we have sufficient points to work with.

## EDA
### Importing libraries
```python
>>> import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc
import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
```
### Reading data and basic stats

```python
df = pd.read_csv("train.csv")
print("Number of data points:",df.shape[0])
```
Number of data points: 404290
```python
  df.head(4)
```
| |"id"|"qid1"|"qid2"|"question1"|"question2"|"is_duplicate"|
|-|----|------|------|-----------|-----------|--------------|
|0|0|1|2|What is the step by step guide to invest in share market in india?|What is the step by step guide to invest in share market?|0|
|1|1|3|4|What is the story of Kohinoor (Koh-i-Noor) Diamond?|What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?|0|
|2|7|15|16|How can I be a good geologist?|What should I do to be a great geologist?|1|
|3|11|23|24|How do I read and find my YouTube comments?|How can I see all my Youtube comments?|1|

df.head() helps us to see the first 5 rows in DataFrame
```python
>>> df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 404290 entries, 0 to 404289
Data columns (total 6 columns):
 #   Column        Non-Null Count   Dtype 
---  ------        --------------   ----- 
 0   id            404290 non-null  int64 
 1   qid1          404290 non-null  int64 
 2   qid2          404290 non-null  int64 
 3   question1     404289 non-null  object
 4   question2     404288 non-null  object
 5   is_duplicate  404290 non-null  int64 
dtypes: int64(4), object(2)
memory usage: 18.5+ MB
```
As we can see that there are no null values in the whole DataFrame
We are given a minimal number of data fields here, consisting of:

- id: Looks like a simple rowID
- qid{1, 2}: The unique ID of each question in the pair
- question{1, 2}: The actual textual contents of the questions.
- is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other.

#### Distribution of data points among output classes
- Number of duplicate(smilar) and non-duplicate(non similar) questions
```python
>>> df.groupby("is_duplicate")['id'].count().plot.bar()
<AxesSubplot:xlabel='is_duplicate'>
```
![image](https://user-images.githubusercontent.com/78054621/139748209-8296e80b-fab1-4713-8f84-99a326368376.png)
```python
>>> print('~> Total number of question pairs for training:\n   {}'.format(len(df)))
~> Total number of question pairs for training:
   404290
```

```python
>>> print('~> Question pairs are not Similar (is_duplicate = 0):\n   {}%'.format(100 - round(df['is_duplicate'].mean()*100, 2)))
>>> print('\n~> Question pairs are Similar (is_duplicate = 1):\n   {}%'.format(round(df['is_duplicate'].mean()*100, 2)))
~> Question pairs are not Similar (is_duplicate = 0):
   63.08%

~> Question pairs are Similar (is_duplicate = 1):
   36.92%    
```

#### 3.2.2 Number of unique questions

```python
>>>
```

```python
>>> qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
>>> unique_qs = len(np.unique(qids))
>>> qs_morethan_onetime = np.sum(qids.value_counts() > 1)
>>> print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))
#print len(np.unique(qids))
>>> print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))
>>> print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 
>>> q_vals=qids.value_counts()
>>> q_vals=q_vals.values
Total number of  Unique Questions are: 537933

Number of unique questions that appear more than one time: 111780 (20.77953945937505%)

Max number of times a single question is repeated: 157

```



