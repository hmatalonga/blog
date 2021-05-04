---
title: 🚢 Predicting the Outcome of the Titanic Passengers
date: 2021-04-23
published: false
tags: ['Machine Learning', 'Classification']
description: "The Kaggle's Titanic Competition is a well-known beginner's Machine Learning problem. The goal is simple: Using Machine Learning create a model to predict which passengers survived the shipwreck."
---

The Kaggle's Titanic Competition is a well-known beginner's Machine Learning problem. The goal is simple: Using Machine Learning create a model to predict which passengers survived the shipwreck.

This is a good starting point to build a complete first project. And have the opportunity to explore different techniques along the way.

In this post, I would like to share some of the ideas I have applied to tackle this project and guide you through the different steps of the work done.

I will begin doing some data exploration and visualization. To get a better understanding of the dataset. The next step is to handle any missing values and perform feature engineering to extract some new insights from the data. Finally, I will try to predict the outcome using different models and compare their performances to find the most suitable one for this project. In the end, I will attempt to improve the results by optimizing the final model parameters.

## Installing Dependencies and Importing Libraries

Before diving into the project, make sure you have these requirements installed:
- Python 3 (at the time of writing, I am using Python 3.7.10)
- SciPy (which includes NumPy, SciPy, Pandas, IPython, and matplotlib)
- scikit-learn
- Seaborn

Then, we will import the necessary packages:

```python
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (classification_report, confusion_matrix)
from sklearn.model_selection import (GridSearchCV, learning_curve,
                                     cross_val_score, train_test_split)
```

## Exploring the Data

We will start by loading the train and test data and taking a peek at the first rows and data summary:

```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.info()
```

<img class="mx-auto" src="/images/titanic/train_data_info.png" lazy>

We can observe that the training data has 891 examples and 11 features plus the target class (survived). Seven of the features are numerical, while 4 are categorical.

```python
train_data.head()
```

<img class="mx-auto" src="/images/titanic/train_data_head.png" lazy>

Next, we need to know what each column of our dataset represents and what values they can have. Here is a description, taken from the Kaggle competition:

| Variable | Definition                                 | Key                                            |
|----------|--------------------------------------------|------------------------------------------------|
| survival | Survival                                   | 0 = No, 1 = Yes                                |
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex      | Sex                                        |                                                |
| Age      | Age in years                               |                                                |
| sibsp    | # of siblings / spouses aboard the Titanic |                                                |
| parch    | # of parents / children aboard the Titanic |                                                |
| ticket   | Ticket number                              |                                                |
| fare     | Passenger fare                             |                                                |
| cabin    | Cabin number                               |                                                |
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |


We can look further at some features. Inspecting their data distributions and asking some questions.

```python
# List of features to view descriptive statistics
features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
train_data[features].describe(include=[np.number])
```

<img class="mx-auto" src="/images/titanic/train_data_describe_numeric.png" lazy>

```python
train_data[features].describe(exclude=[np.number])
```

<img class="mx-auto" src="/images/titanic/train_data_describe_categorical.png" lazy>

**Is our training data balanced? Do we have a reasonable distribution of passengers who survived and those who did not?**</br>
The training data is reasonably balanced. Around 61% of the passengers died, and approximately 38% survived.

```python
train_data['Survived'].value_counts(normalize=True)
# 0    0.616162
# 1    0.383838
# Name: Survived, dtype: float64
```

**Do we have missing values in our training data and/or testing data?**<br/>
We have some missing values in both training and testing sets. We will come back to it later on.

```python
train_data.isna().sum()
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2
# dtype: int64
```

```python
test_data.isna().sum()
# PassengerId      0
# Pclass           0
# Name             0
# Sex              0
# Age             86
# SibSp            0
# Parch            0
# Ticket           0
# Fare             1
# Cabin          327
# Embarked         0
# dtype: int64
```

### Data Visualization

Creating data visualizations can make it easier to spot data patterns. We will ask a few more questions about the data, looking at different plots. This can give us insights into how the features are related to each other. And which ones could have a higher impact on the survival rate of the passengers.

**Did one gender survive more than the other?**<br/>
We can confirm that a significantly higher percentage of female passengers survived compared to the male passengers.

```python
sns.catplot(x='Sex', y='Survived', kind='bar', data=train_data)
```

<img class="mx-auto" src="/images/titanic/plot_gender_survival.png" lazy>

**What is the distribution of passengers ages, which Age groups had higher survival odds, comparing gender?**<br/>
The passengers' ages seem to follow a normal distribution.

```python
sns.displot(x='Age', hue='Survived', col='Sex', kde=True, data=train_data)
```

<img class="mx-auto" src="/images/titanic/plot_age_gender_survival.png" lazy>


**What is the distribution of passengers parents/children aboard?**

```python
sns.countplot(data=train_data, x='Parch')
```

<img class="mx-auto" src="/images/titanic/plot_parch_dist.png" lazy>

**What is the distribution of passengers siblings/spouses aboard?**

```python
sns.countplot(data=train_data, x='SibSp')
```

<img class="mx-auto" src="/images/titanic/plot_sibsp_dist.png" lazy>

**What is the distribution of passengers per ticket class?**<br/>
Most passengers were from the 3rd class.

```python
sns.countplot(data=train_data, x='Pclass')
```

<img class="mx-auto" src="/images/titanic/plot_pclass_dist.png" lazy>

**Which classes of passengers survived the most? Comparing gender.**<br/>
It seems that passengers with higher classes had a better survival rate overall. When comparing gender, the difference is more pronounced with the female passengers of the upper classes, having a much higher survival rate than the male passengers. Also, female passengers of the upper classes survived more than the female passengers of the 3rd ticket class.

```python
sns.catplot(x='Sex', y='Survived', hue='Pclass', kind='point', data=train_data)
```

<img class="mx-auto" src="/images/titanic/plot_gender_pclass_survival.png" lazy>

**Which places of embarking have the most survivors? Comparing gender?**

```python
sns.catplot(x='Sex', y='Survived', hue='Embarked', kind='bar', data=train_data)
```

<img class="mx-auto" src="/images/titanic/plot_gender_embarked_survival.png" lazy>


### Handling Missing Values

As we saw before, there are a total of 4 features containing missing values: *Cabin*, *Age*, *Embarked* and *Fare*.

#### Age

For the feature *Age*, I have seen people approaching it in a few ways: Some prefer to group the *Age* in a certain number of bins. There are some examples where they train a regressor model to predict the missing values. Another way, which is the one I am adopting, is to create a normal distribution based on the existing values to fill in the missing ones.

## Applying Feature Engineering



Thank you for taking the time to read. You can follow me on Twitter, where I share my learning journey. I talk about Data Science, Machine Learning, among other topics I am interested in.