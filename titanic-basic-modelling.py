#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


#!pip install pandas==0.25.3
#!pip install scikit-learn==0.22
#!pip install numpy==1.18.0
#!pip install ppscore


# In[3]:


import numpy as np 
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
import seaborn as sns
import ppscore as pps
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import f1_score
from sklearn.externals import joblib

seed = 2020


# In[4]:


print('numpy version: '+ np.__version__)
print('pandas version: '+ pd.__version__)
print('sklearn version: '+ sklearn.__version__)


# In[5]:


pwd


# In[6]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('gender_submission.csv')


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


train.describe()


# In[10]:


train.info()


# In[11]:


# checking for missing values 
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(train.isnull(), cbar=False)
plt.show()


# ### There are missing values in age and Cabin features 

# # Short Exploratory Data Analysis  

# In[12]:


sns.countplot(train.Survived)


# ### Imbalance dataset... resampling techniques will be useful so as to have better performance evaluation

# In[13]:


sns.barplot(x='Sex', y='Survived', data=train)
plt.ylabel("Rate of Surviving")
plt.title("Plot of Survival as function of Sex", fontsize=16)
plt.show()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### There are much more survivals in female than male

# In[14]:


sns.barplot(x='Pclass', y='Survived', data=train)
plt.ylabel("Survival Rate")
plt.title("Plot of Survival as function of Pclass", fontsize=16)
plt.show()
train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### Passengers in Pclass 1 and 2 have higher chances of surviving 

# In[15]:


plt.figure(figsize=(10,5))
sns.heatmap(train.corr(),annot=True)


# Some features are highly correlated with one another, creating new features will be helpful. e.g Pclass and SibSp are highly correlated, a new feature will be created to reduce the Collinearity

# In[16]:


plt.figure(figsize=(10,5))
ax = train.corr()['Survived'].plot(kind='bar',title='correlation of target variable to features')
ax.set_ylabel('correlation')


# The passengerId feature will be dropped as this is unique across all samples. Pclass and Fare have the highest absolute correletion with the target variable

# In[17]:


train_copy = train.copy()
train_copy.dropna(inplace = True)
sns.distplot(train_copy.Age)


# Looks like the distribution of ages is slightly skewed right. Because of this, we can fill in the null values with the median for the most accuracy.

# In[18]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_copy)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# There were more female survivals than male in all the 3 Pclass categories

# In[19]:


sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train_copy)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# From the graphs above,Female survival rate in the all the Pclasses is higher than male survival rate respectively

# In[20]:


train_null = train.isnull().sum()
test_null = test.isnull().sum()
print(train_null[train_null !=0])
print('-'*40)
print(test_null[test_null !=0])


# # Handling missing values 

# In[21]:


from sklearn.impute import SimpleImputer
age_imp = SimpleImputer(strategy= 'median')
age_imp.fit(np.array(train.Age).reshape(-1,1))

train.Age = age_imp.transform(np.array(train.Age).reshape(-1,1))
test.Age = age_imp.transform(np.array(test.Age).reshape(-1,1))
train.head()


# In[22]:


#save age imputer 
with open('age_imputer.joblib', 'wb') as f:
  joblib.dump(age_imp,f)


# In[23]:


emb_imp = SimpleImputer(strategy= 'most_frequent' )
emb_imp.fit(np.array(train.Embarked).reshape(-1,1))

train.Embarked = emb_imp.transform(np.array(train.Embarked).reshape(-1,1))
test.Embarked = emb_imp.transform(np.array(test.Embarked).reshape(-1,1))
train.head()


# In[24]:


#save embark imputer 
with open('embark_imputer.joblib', 'wb') as f:
  joblib.dump(emb_imp,f)


# In[25]:


train.isnull().sum() 
print('-'*40)
test.isnull().sum()


# In[26]:


drop_cols = ['PassengerId','Ticket','Cabin','Name']
train.drop(columns=drop_cols,axis=1,inplace = True)
test_passenger_id = test.PassengerId
test.drop(columns=drop_cols,axis=1,inplace = True)


# In[27]:


test.fillna(value = test.mean(),inplace=True)


# In[28]:


train.isnull().sum().any() , test.isnull().sum().any()


# In[29]:


train['Number_of_relatives'] = train.Parch + train.SibSp
test['Number_of_relatives'] = test.Parch + test.SibSp

train.drop(columns=['Parch','SibSp'],axis=1,inplace=True)
test.drop(columns=['Parch','SibSp'],axis=1,inplace=True)


# In[30]:


train.head()


# In[31]:


gender_dic = {'male':1,'female':0}
train.Sex = train.Sex.map(gender_dic)
test.Sex = test.Sex.map(gender_dic)
train.head()


# In[32]:


cat_col = ['Embarked', 'Pclass']
One_hot_enc = OneHotEncoder(sparse=False,drop='first',dtype=np.int)


# In[33]:


encoded_train = pd.DataFrame(data=One_hot_enc.fit_transform(train[cat_col]), columns=['emb_2','emb_3','Pclass_2','Pclass_3'])
encoded_test = pd.DataFrame(data=One_hot_enc.transform(test[cat_col]),columns=['emb_2','emb_3','Pclass_2','Pclass_3'])


# In[34]:


#save One_hot_enc 
with open('One_hot_enc.joblib', 'wb') as f:
  joblib.dump(One_hot_enc,f)


# In[35]:


train.drop(columns=cat_col,axis=1,inplace=True)
test.drop(columns=cat_col,axis=1,inplace=True)

train = pd.concat([train,encoded_train],axis=1)
test = pd.concat([test,encoded_test],axis=1)
train.head()


# In[36]:


features = test.columns
X = train[features]
y = train.Survived


# In[37]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)


# In[38]:


#save scaler 
with open('scaler.joblib', 'wb') as f:
  joblib.dump(scaler,f)


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)


# In[40]:


logistic_model  = LogisticRegression()
logistic_model.fit(X_train,y_train)


# In[41]:


print('f1_score on training set: {}'.format(f1_score(logistic_model.predict(X_train),y_train)))
print('f1_score on test set: {}'.format(f1_score(logistic_model.predict(X_test),y_test)))


# In[42]:


logistic_model.fit(X,y)
#save model 
with open('model-v1.joblib', 'wb') as f:
  joblib.dump(logistic_model,f)

