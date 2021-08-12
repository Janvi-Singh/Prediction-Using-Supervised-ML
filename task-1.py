#!/usr/bin/env python
# coding: utf-8

# # Janvi Singh
#  
# 
# 

# 
# <h1> DATA SCIENCE AND BUSINESS ANALYTICS INTERN Under Graduate Rotational Internship Program (Grip August 2021) by The Sparks Foundation</h1>
# <H2>Task-1 Prediction Using Supervised ML</H2>
# 

# <b> The aim of the task is to be predict the percentage of a student based on the no of study hours . It is a simple linear regression task as it involves just two variables . I have used Python Language in this project . Further, different python language [Numpy , Pandas , Seaborn and Matplotlib] are imported for performing different data analytic techniques.</b>

# In[1]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[2]:


import jovian


# In[3]:


# Execute this to save new versions of the notebook
jovian.commit(project="task-1")


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# <b> The Dataset for the project is downloaded from <a>"http://bit.ly/w-data"</a></b>

# In[5]:


dataset=pd.read_csv('http://bit.ly/w-data')


# <h2>Data Prepration and Data Cleaning</h2>

# In[6]:


print(dataset.shape)


# In[7]:


dataset.head(15)


# In[8]:


dataset.info()


# In[9]:


dataset.describe()


# In[10]:


sns.set_style("darkgrid")


# In[11]:


dataset.plot( x= 'Hours' , y='Scores' , style='or')
plt.title("Complete Data")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage scored")
plt.figure(figsize=(12, 6))
plt.show()


# <p><b>By the help of different functions we analyze , prepare and clean the data . Here, </b><ul>* dataset.shape shows that there are 25 rows and 2 columns.</ul><ul>* dataset.describe() describes the data . </ul><ul>* dataset.info() gives information about the data . In this case there are 25 non-null values which depicts that the dataset does not have any null values   and we don't need to take care for that .</ul><ul>* plot() shows the linear relation between Hours and scores which implies that there will be an increase in scores for the higher hours of study .</ul></p> 

# <h2> Training Of Dataset<br></h2>

# In[12]:


X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test ,y_train, y_test= train_test_split(X ,y ,test_size=0.2 , random_state=0)


# In[14]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train , y_train)
print("Successfully trained the model")


# <b>Here, the dataset is first split into training and test series using scikit learn's bulletin method of train_test_split(). Further, The algorithm is trained using Linear Regression . </b>

# In[15]:


plt.figure(figsize=(12, 6))
line=reg.coef_*X+reg.intercept_
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[16]:


print(X_test)
y_pred=reg.predict(X_test)


# <b> Now the Original dataset is compared to the predicted data to check that the model is fitted properly or not.</b>

# In[17]:


res_df=pd.DataFrame({ 'Original Data': y_test , 'Predicted Data':y_pred})
res_df


# In[24]:


plt.figure(figsize=(12, 6))
plt.title("Data Comparision")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage scored")
plt.legend(['Original Data', 'Predicted Data'])
plt.bar(y_test,y_pred)
plt.bar(y_test,y_pred, bottom =y_pred);


# <b>On comparing the datasets we can say that the predicted data does not provide exact values but a close enough approach . Hence, we can say that model is fitted properly and can be used </b>  

# <h2> TO Do Question</h2>
# <br> <b>What will be predicted score if a student studies for 9.25 hrs/day? </b>

# In[19]:


hours=[[9.25]]
mod_pred=reg.predict(hours)
print(" For {} hours Predicted Score is {}" .format(hours , mod_pred[0]))


# <h2>Conclusion</h2>
# <br>
# <p> Since, the model predicts the percentage of student based on the no of study hours <b> Successfully </b> . Moreover, the To-Do question in the task predicts the value close enough to what would have been predicted by original value . Hence, we can conclude that our model predicts the value effectively .  
