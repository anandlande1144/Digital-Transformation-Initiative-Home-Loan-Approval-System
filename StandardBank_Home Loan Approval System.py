#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


df = pd.read_csv(r"D:\Anand\Data Analysis and science\FORAGE\standard bank\train.csv")


# In[25]:


# showing first five rows of the data
df.head()


# In[ ]:


df.columns.str.replace()


# In[ ]:





# In[26]:


df.head()


# In[27]:


#checking the number of columns and rows in data
df.shape


# In[28]:


#statisticalmeasures
df.describe().T


# In[29]:


#finding missing vakue from is column
df.isnull().sum()


# In[30]:


df.shape


# In[31]:


#filling the missing values using different techniuques
df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace = True)


# In[32]:


# dropping the null values from the data
df = df.dropna()


# In[33]:


# checking is there any null values
df.isnull().sum()


# In[34]:


df.shape


# In[12]:


#label encoding
from sklearn.preprocessing import LabelEncoder


# In[13]:


df.replace({'Loan_Status':{'N':0, 'Y':1}}, inplace = True)


# In[14]:


df.Loan_Status.value_counts()


# In[15]:


# labeling on "Dependents" columns
df['Dependents'].value_counts()


# In[16]:


#replacing the value of 3+ with 4
df['Dependents'] = df["Dependents"].replace(to_replace = '3+', value = 4)


# # Visualization

# In[17]:


sns.set_theme()


# In[18]:


# education and Loan Status
sns.countplot( x='Education', hue='Loan_Status', data=df)


# In[19]:


# Marries Status vs Loan Status
sns.countplot(data = df ,x="Married", hue = 'Loan_Status')


# Cheking all the categorial data

# In[20]:


# plotting histogram
df['loanAmount_log'] = np.log(df['LoanAmount'])

print(df['loanAmount_log'].hist(bins=20))
print()


# In[21]:


print("Number Of appllied for Loan Approved By Genders")
print(df.Gender.value_counts())
sns.countplot(x="Gender", data = df, palette = 'Set2')


# In[22]:


print("Number Of appllied for Loan Approved By Married Stutus")
print(df.Married.value_counts())
sns.countplot(x="Married", data = df, palette = 'Set3')


# In[23]:


print("Number Of appllied for Loan Approved By  Credit History")
print(df.Credit_History.value_counts())
sns.countplot(x="Credit_History", data = df, palette = 'Set3')


# In[24]:


print("Number Of appllied for Loan Approved By Self_Emoployed")
print(df.Self_Employed.value_counts())
sns.countplot(x="Self_Employed", data = df, palette = 'Set1')


# In[ ]:





# In[25]:


df.Married.value_counts()


# In[26]:


df.Gender.value_counts()


# In[27]:


df.Self_Employed.value_counts()


# In[28]:


df.Property_Area.value_counts()


# In[29]:


df.Education.value_counts()


# In[30]:


# converting categorical values into numerical values
df.replace({'Married':{'No':0, 'Yes':1},
            'Gender':{'Male':1, 'Female':0},
            'Self_Employed':{'No':0, "Yes":1},
           'Property_Area':{"Rural":0,'Semiurban':1, 'Urban':2},
           "Education":{'Graduate':1,"Not Graduate":0}}, inplace = True)


# In[31]:


#separating data and label
X=df.drop(columns = ['Loan_ID',"Loan_Status"], axis = 1)
Y = df['Loan_Status']


# In[32]:


print(X)
print(Y)


# In[33]:


#splitting the data in train and test
from sklearn.model_selection import train_test_split
X_train, X_test ,Y_train, Y_test = train_test_split(X,Y, test_size = 0.2,stratify = Y, random_state = 2 )


# In[34]:


print(X.shape, X_train.shape, X_test.shape)


# Training the model through Sopport vector model
# 

# In[35]:


from sklearn import svm
from sklearn.metrics import accuracy_score


# In[36]:


svm_cls = svm.SVC(kernel = 'linear')


# In[37]:


#training the vector machine model


# In[38]:


svm_cls.fit(X_train, Y_train)


# Model Evaluation

# In[ ]:





# In[ ]:





# In[51]:


print(test_acc*100,"%")


# In[52]:


#Accuracy Score on testing data
X_test_prediction = svm_cls.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)


# In[ ]:





# In[53]:


print(test_acc*100,"%")


# In[79]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, Y_train)


# In[80]:


#Accuracy Score on training data
X_test_prediction = rf_clf.predict(X_test)


# In[81]:


x_test_accu = accuracy_score(X_test_prediction,Y_test)
print(y_test_accu*100,"%")


# In[77]:


from sklearn.naive_bayes import GaussianNB 
nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)


# In[78]:


#Accuracy Score on training data
X_test_prediction = nb_clf.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print(test_acc*100,"%")


# In[82]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[85]:


print(confusion_matrix(Y_test, X_test_prediction))


# In[ ]:





# In[58]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)


# In[59]:


#Accuracy Score on training data
X_test_prediction = dt_clf.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print(test_acc*100,"%")


# In[60]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, Y_train)


# In[61]:


#Accuracy Score on training data
X_test_prediction = kn_clf.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print(test_acc*100,"%")


# In[62]:





# In[ ]:




