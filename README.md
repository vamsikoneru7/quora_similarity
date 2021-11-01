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
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
from fuzzywuzzy import fuzz
from sklearn.manifold import TSNE
from os import path
from PIL import Image
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
![image](https://user-images.githubusercontent.com/78054621/139749926-7326fab8-f024-46aa-b333-e6943a466ca1.png)
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

```python
>>> x = ["unique_questions" , "Repeated Questions"]
>>> y =  [unique_qs , qs_morethan_onetime]
>>> plt.figure(figsize=(10, 6))
>>> plt.title ("Plot representing unique and repeated questions  ")
>>> sns.barplot(x,y)
>>> plt.show()
```![image](https://user-images.githubusercontent.com/78054621/139749955-ab3d6760-abb7-4b20-8f24-fc5d397fe148.png)
#### Checking for Duplicates

```python
#checking whether there are any repeated pair of questions
>>> pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()
>>> print ("Number of duplicate questions",(pair_duplicates).shape[0] - df.shape[0])
Number of duplicate questions 0
```
#### Number of occurrences of each question
```python
>>> plt.figure(figsize=(20, 10))
>>> plt.hist(qids.value_counts(), bins=160)
>>> plt.yscale('log', nonpositive='clip')
>>> plt.title('Log-Histogram of question appearance counts')
>>> plt.xlabel('Number of occurences of question')
>>> plt.ylabel('Number of questions')
>>> print ('Maximum number of times a single question is repeated: {}\n'.format(max(qids.value_counts())))
Maximum number of times a single question is repeated: 157
```
![image](https://user-images.githubusercontent.com/78054621/139749983-960fb143-e974-481c-8bd5-286cc494f069.png)
#### Checking for NULL Values

```python
#Checking whether there are any rows with null values
>>> nan_rows = df[df.isnull().any(1)]
>>> print (nan_rows)
            id    qid1    qid2                         question1  \
105780  105780  174363  174364    How can I develop android app?   
201841  201841  303951  174364  How can I create an Android app?   
363362  363362  493340  493341                               NaN   

                                                question2  is_duplicate  
105780                                                NaN             0  
201841                                                NaN             0  
363362  My Chinese name is Haichao Yu. What English na...             0  
```
- There are two rows with null values in question2
```python
# Filling the null values with ' '
>>> df = df.fillna('')
>>> nan_rows = df[df.isnull().any(1)]
>>> print (nan_rows)
Empty DataFrame
Columns: [id, qid1, qid2, question1, question2, is_duplicate]
Index: []
```
### Basic Feature Extraction (before cleaning)
Let us now construct a few features like:

- freq_qid1 = Frequency of qid1's
- freq_qid2 = Frequency of qid2's
- q1len = Length of q1
- q2len = Length of q2
- q1_n_words = Number of words in Question 1
- q2_n_words = Number of words in Question 2
- word_Common = (Number of common unique words in Question 1 and Question 2)
- word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
- word_share = (word_common)/(word_Total)
- freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
- freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2
```python
>>>if os.path.isfile('df_fe_without_preprocessing_train.csv'):
    df = pd.read_csv("df_fe_without_preprocessing_train.csv",encoding='latin-1')
  else:
    df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 
    df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
    df['q1len'] = df['question1'].str.len() 
    df['q2len'] = df['question2'].str.len()
    df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

    def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)
    df['word_Common'] = df.apply(normalized_word_Common, axis=1)

    def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * (len(w1) + len(w2))
    df['word_Total'] = df.apply(normalized_word_Total, axis=1)

    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    df['word_share'] = df.apply(normalized_word_share, axis=1)

    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])

    df.to_csv("df_fe_without_preprocessing_train.csv", index=False)
```
#### Analysis of some of the extracted features
- Here are some questions have only one single words.

```python
>>> print ("Minimum length of the questions in question1 : " , min(df['q1_n_words']))
>>> print ("Minimum length of the questions in question2 : " , min(df['q2_n_words']))
>>> print ("Number of Questions with minimum length [question1] :", df[df['q1_n_words']== 1].shape[0])
>>> print ("Number of Questions with minimum length [question2] :", df[df['q2_n_words']== 1].shape[0])

Minimum length of the questions in question1 :  1
Minimum length of the questions in question2 :  1
Number of Questions with minimum length [question1] : 67
Number of Questions with minimum length [question2] : 24
```
##### Feature: word_share
```python
>>> plt.figure(figsize=(12, 8))
>>> plt.subplot(1,2,1)
>>> sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:])
>>> plt.subplot(1,2,2)
>>> sns.histplot(df[df['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red',kde= True)
>>> sns.histplot(df[df['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue',kde= True )
>>> plt.show()
```
![image](https://user-images.githubusercontent.com/78054621/139750789-49c92d6d-fd9c-43a1-88d2-87611ba5520c.png)
- The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity
- The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)
##### Feature: word_Common
```python
>>> plt.figure(figsize=(12, 8))
>>> plt.subplot(1,2,1)
>>> sns.violinplot(x = 'is_duplicate', y = 'word_Common', data = df[0:])
>>> plt.subplot(1,2,2)
>>> sns.histplot(df[df['is_duplicate'] == 1.0]['word_Common'][0:] , label = "1", color = 'red',kde= True)
>>> sns.histplot(df[df['is_duplicate'] == 0.0]['word_Common'][0:] , label = "0" , color = 'blue',kde =True)
>>> plt.show()
```
![image](https://user-images.githubusercontent.com/78054621/139750914-ed12e04b-2030-47de-98ab-0ce71ccb783c.png)
- The distributions of the word_Common feature in similar and non-similar questions are highly overlapping
```python
>>>if os.path.isfile('df_fe_without_preprocessing_train.csv'):
    df = pd.read_csv("df_fe_without_preprocessing_train.csv",encoding='latin-1')
    df = df.fillna('')
    df.head()
  else:
    print("get df_fe_without_preprocessing_train.csv from drive or run the previous notebook")
```
```python
>>>df.head(2)
```
| id| qid1| qid2| question1| question2| is_duplicate| freq_qid1| freq_qid2| q1len| q2len| q1_n_words| q2_n_words| word_Common| word_Total| word_share| freq_q1+q2| freq_q1-q2|
|---|-----|-----|----------|----------|-------------|----------|----------|------|------|-----------|-----------|------------|-----------|-----------|-----------|-----------|
| 0| 0|	1| 2|	What is the step by step guide to invest in sh...	|What is the step by step guide to invest in sh...	|0|	1|	1|	66|	57|	14|	12|	10.0|	23.0|	0.434783|	2|	0|
| 1| 1|	3| 4|	What is the story of Kohinoor (Koh-i-Noor) Dia...	|What would happen if the Indian government sto...	|0|	4|	1|	51|	88|	8	|13	|4.0	|20.0|	0.200000|	5	| 3|
### Preprocessing of Text
- Preprocessing:
  - Removing html tags
  - Removing Punctuations
  - Performing stemming
  - Removing Stopwords
  - Expanding contractions etc.
```python
# To get the results in 4 decemal points
>>> SAFE_DIV = 0.0001 
>>> STOP_WORDS = stopwords.words("english")
>>>def preprocess(x):
     x = str(x).lower()
     x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                            .replace("€", " euro ").replace("'ll", " will")
     x = re.sub(r"([0-9]+)000000", r"\1m", x)
     x = re.sub(r"([0-9]+)000", r"\1k", x)
     
     porter = PorterStemmer()
     pattern = re.compile('\W')
     
     if type(x) == type(''):
         x = re.sub(pattern, ' ', x)
    
     if type(x) == type(''):
         x = porter.stem(x)
         example1 = BeautifulSoup(x)
         x = example1.get_text()
     
     return x
```
- Function to Compute and get the features : With 2 parameters of Question 1 and Question 2
#### Advanced Feature Extraction (NLP and Fuzzy Features)
##### Definition:
- **Token**: You get a token by splitting sentence a space
- **Stop_Word** : stop words as per NLTK.
- **Word** : A token that is not a stop_word
##### Features:

- **cwc_min** : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
- **cwc_min** = common_word_count / (min(len(q1_words), len(q2_words))

- **cwc_max** : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
- **cwc_max** = common_word_count / (max(len(q1_words), len(q2_words))

- **csc_min** : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
- **csc_min** = common_stop_count / (min(len(q1_stops), len(q2_stops))

- **csc_max** : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
- **csc_max** = common_stop_count / (max(len(q1_stops), len(q2_stops))

- **ctc_min** : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
- **ctc_min** = common_token_count / (min(len(q1_tokens), len(q2_tokens))


- **ctc_max** : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
- **ctc_max** = common_token_count / (max(len(q1_tokens), len(q2_tokens))


- **last_word_eq** : Check if First word of both questions is equal or not
- **last_word_eq** = int(q1_tokens[-1] == q2_tokens[-1])


- **first_word_eq** : Check if First word of both questions is equal or not
- **first_word_eq** = int(q1_tokens[0] == q2_tokens[0])


- **abs_len_diff** : Abs. length difference
- **abs_len_diff**= abs(len(q1_tokens) - len(q2_tokens))


- **mean_len** : Average Token Length of both Questions
- **mean_len** = (len(q1_tokens) + len(q2_tokens))/2


- **fuzz_ratio** : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


- **fuzz_partial_ratio** : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


- **token_sort_ratio** : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

- **token_set_ratio** : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

- longest_substr_ratio** : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2
  - **longest_substr_ratio** = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
```python
>>> def get_token_features(q1, q2):
       token_features = [0.0]*10
    
       # Converting the Sentence into Tokens: 
       q1_tokens = q1.split()
       q2_tokens = q2.split()

       if len(q1_tokens) == 0 or len(q2_tokens) == 0:
           return token_features
       # Get the non-stopwords in Questions
       q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
       q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
       
       #Get the stopwords in Questions
       q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
       q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
       # Get the common non-stopwords from Question pair
       common_word_count = len(q1_words.intersection(q2_words))
    
       # Get the common stopwords from Question pair
       common_stop_count = len(q1_stops.intersection(q2_stops))
    
       # Get the common Tokens from Question pair
       common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
       token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
       token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
       token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
       token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
       token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
       token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
       # Last word of both question is same or not
       token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
       # First word of both question is same or not
       token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
       token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
       #Average Token Length of both Questions
       token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
       return token_features

# get the Longest Common sub string

>>> def get_longest_substr_ratio(a, b):
       strs = list(distance.lcsubstrings(a, b))
       if len(strs) == 0:
           return 0
       else:
           return len(strs[0]) / (min(len(a), len(b)) + 1)

>>> def extract_features(df):
       # preprocessing each question
       df["question1"] = df["question1"].fillna("").apply(preprocess)
       df["question2"] = df["question2"].fillna("").apply(preprocess)

       print("token features...")
    
       # Merging Features with dataset
    
       token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    
       df["cwc_min"]       = list(map(lambda x: x[0], token_features))
       df["cwc_max"]       = list(map(lambda x: x[1], token_features))
       df["csc_min"]       = list(map(lambda x: x[2], token_features))
       df["csc_max"]       = list(map(lambda x: x[3], token_features))
       df["ctc_min"]       = list(map(lambda x: x[4], token_features))
       df["ctc_max"]       = list(map(lambda x: x[5], token_features))
       df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
       df["first_word_eq"] = list(map(lambda x: x[7], token_features))
       df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
       df["mean_len"]      = list(map(lambda x: x[9], token_features))
   
       #Computing Fuzzy Features and Merging with Dataset
    
       # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
       # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
       # https://github.com/seatgeek/fuzzywuzzy
       print("fuzzy features..")

       df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
       # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
       # then joining them back into a string We then compare the transformed strings with a simple ratio().
       df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
       df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
       df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
       df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
       return df
```
```python
>>>if os.path.isfile('nlp_features_train.csv'):
      df = pd.read_csv("nlp_features_train.csv",encoding='latin-1')
      df.fillna('')
   else:
      print("Extracting features for train:")
      df = pd.read_csv("train.csv")
      df = extract_features(df)
      df.to_csv("nlp_features_train.csv", index=False)
   df.head(2)
```
| |id|	qid1|	qid2|	question1|	question2|	is_duplicate|	cwc_min|	cwc_max|	csc_min|	csc_max|	...|	ctc_max	|last_word_eq|	first_word_eq|	abs_len_diff| mean_len| token_set_ratio|	token_sort_ratio|	fuzz_ratio|	fuzz_partial_ratio|	longest_substr_ratio|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|0|	0|	1|	2|	what is the step by step guide to invest in sh...|	what is the step by step guide to invest in sh...|	0|	0.999980|	0.833319|	0.999983|	0.999983|	...|	0.785709|	0.0|	1.0|	2.0|	13.0|	100|	93|	93|	100|	0.982759|
|1|	1|	3|	4|	what is the story of kohinoor koh i noor dia...|	what would happen if the indian government sto...|	0|	0.799984|	0.399996|	0.749981|	0.599988|	...	|0.466664| 0.0|	1.0|	5.0|	12.5|	86|	63|	66|	75|	0.596154|
```python
>>>
```
```python
>>>
```
