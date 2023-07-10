#!/usr/bin/env python
# coding: utf-8

# In[34]:


print('Capstone Project')
print('ALY6140 Analytics Systems Technology Fall 2022')
print('Jenil Desai & Khishan Bhingradiya')

# import package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.io import output_notebook
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import sklearn
import sklearn.metrics
from sklearn import ensemble
from sklearn import linear_model
import warnings

csvdata_url = 'C:/Users/Dell/OneDrive/Desktop/Data_set/kc_house_data.csv'
csvdata = pd.read_csv(csvdata_url, header=0)
csvdata


# In[ ]:





# Data Cleanup
# **bold text*** Select the columns we need.
# * Detect the Null value.
# * Find the outlier and delete them
# * Create the new column year of built.
# * Describe the dataset which we clean up.

# In[3]:


print(csvdata.columns)


# In[4]:


#columns rename
csvdata.rename(columns = {'bedrooms':'Rooms'}, inplace = True)
csvdata.rename(columns = {'yr_built':'BuildYear'}, inplace = True)
csvdata.rename(columns = {'zipcode':'Postcode'}, inplace = True)
csvdata
print(csvdata.columns)


# In[5]:


#data cleaning
print(csvdata.isnull())
print(csvdata.isnull().sum())
print(csvdata.head())


# In[6]:


#remove unwanted data
remove = ['date', 'id']
csvdata.drop(remove, inplace =True, axis = 1)


# In[7]:


#data normalization

from sklearn import preprocessing
import numpy as np

scaler = preprocessing.MinMaxScaler()
names = csvdata.columns
d = scaler.fit_transform(csvdata)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()


# In[ ]:





# See the description of the new dataset. We can see that the data is organized and simple to understand. Additionally, it is neither empty nor outlier. We can use this info to analyze.

# In[8]:


#data description,shape and count
print('\ndata_selected Dataset Describe :\n')
print(csvdata.describe())
print('\nShape Dataset:\n')
print(csvdata.shape)
print('\nNull value count :\n')
print(csvdata.isnull().sum())


# # Create New Colums
# In this section, I add a new column called year to the dataset. The year symbolizes the year of the house. if renovations are made to the home. The first year is the year that was renovated. If not, the built year is the first year. So, to determine the year of the house, I will use 2022 minus the first year.

# In[9]:



csvdata['years']=2022-csvdata['BuildYear']
for x,y in csvdata['yr_renovated'].items():
    if y != 0:
        csvdata.loc[x,'years'] = 2022-y
csvdata = csvdata.drop('BuildYear',axis=1)
csvdata = csvdata.drop('yr_renovated',axis=1)
print(csvdata.head())


# Data Visualization
# 

# In[10]:


# Data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#seaborn
import seaborn as sns
plt.scatter(csvdata.index,csvdata['sqft_living'])
plt.show()


# In[11]:


sns.scatterplot(x = csvdata.index , y = csvdata['price'],hue = csvdata['sqft_living'])


# In[12]:


print(csvdata['price'].describe())
plt.hist(csvdata['price'],bins=9)
plt.title('House prices')
plt.xlabel('Count')
plt.ylabel('Price')


# In[15]:


b_plot = sns.boxplot(csvdata['sqft_living'])
b_plot.set(title = 'sqft for Boxplot', xlabel = 'area')


# In[16]:


plt.subplots(figsize=(15,10))
plt.subplot(2,2,1)
plt.hist(csvdata['sqft_living'],bins=10)
plt.title('House Square Footage of Home')
plt.xlabel('Thousand Square Footage')
plt.ylabel('Count')
plt.subplot(2,2,2)
plt.hist(csvdata['sqft_lot'],bins=10)
plt.title('House Square Footage of Lot')
plt.xlabel('Thousand Square Footage')
plt.ylabel('Count')
plt.subplot(2,2,3)
plt.hist(csvdata['sqft_above'],bins=10)
plt.title('House Square Footage of House Apart From Basement')
plt.xlabel('Thousand Square Footage')
plt.ylabel('Count')
plt.subplot(2,2,4)
plt.hist(csvdata['sqft_basement'],bins=10)
plt.title('House Square Footage of Basement')
plt.xlabel('Thousand Square Footage')
plt.ylabel('Count')


# In[17]:


plt.subplots(figsize=(15,10))
plt.subplot(2,2,1)
b_plt1 = sns.countplot(csvdata['Rooms'])
b_plt1.set(title = 'Rooms Distribution', xlabel = 'Room')
plt.subplot(2,2,2)
b_plt2 = sns.countplot(csvdata['bathrooms'])
b_plt2.set(title = 'bathrooms Distribution', xlabel = 'bathrooms')
plt.subplot(2,2,3)
b_plt3 = sns.countplot(csvdata['floors'])
b_plt3.set(title = 'floors Distribution', xlabel = 'floors')
plt.subplot(2,2,4)
b_plt4 = sns.countplot(csvdata['waterfront'])
b_plt4.set(title = 'waterfront Distribution', xlabel = 'waterfront')


# In[18]:


plt.subplots(figsize=(15,10))
plt.subplot(2,2,1)
b_plt5 = sns.countplot(csvdata['years'])
b_plt5.set(title = 'Year Distribution', xlabel = 'Years')
plt.subplot(2,2,2)
b_plt6 = sns.countplot(csvdata['condition'])
b_plt6.set(title = 'house condition Distribution', xlabel = 'condition')
plt.subplot(2,2,3)
b_plt7 = sns.countplot(csvdata['grade'])
b_plt7.set(title = 'grade', xlabel = 'grade')


# In[30]:


df1 = csvdata.groupby('Rooms').sum()
df2 = df1.sort_values('price',ascending = False)
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.pie(x= df2['price'].head(10),labels = df2.head(10).index)
plt.title("how much Rooms for the house")
plt.show()


# In[31]:


correlation = csvdata.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="Greens")


# In[33]:


fig, ax_1 = plt.subplots(figsize=(15,10)) 
fig.suptitle('Price vs House', fontsize=16)

plt.subplot(2,3,1)
sns.regplot(x='Rooms', y="price", data=csvdata) 
plt.xlabel('Rooms') 

plt.subplot(2,3,2)
sns.regplot(x='waterfront', y="price", data=csvdata) 
plt.xlabel('front viwe') 

plt.subplot(2,3,3)
sns.regplot(x='view', y="price", data=csvdata) 
plt.xlabel('view') 

plt.subplot(2,3,4)
sns.regplot(x='bathrooms', y="price", data=csvdata) 
plt.xlabel('bathrooms') 

plt.subplot(2,3,5)
sns.regplot(x='sqft_basement', y="price", data=csvdata) 
plt.xlabel('sqft_basement') 

plt.subplot(2,3,6)
sns.regplot(x='Postcode', y="price", data=csvdata) 
plt.xlabel('Postcode') 


# # Regrassion Model
# 
# 1.Linear Regrassion

# In[23]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(csvdata ,train_size =0.25, random_state = 0)


# In[24]:


#Regression
from sklearn import linear_model
model = linear_model.LinearRegression()
y_train, X_train = dmatrices('price ~ Rooms + years + condition + bathrooms + sqft_basement + view', data=train_data)
model.fit(X_train, y_train)
yhat_train = model.predict(X_train)


# In[25]:


y_test,X_test = dmatrices('price ~ Rooms + years + condition + bathrooms + sqft_basement + view', data=test_data)
fig, ax = plt.subplots(figsize=(15,10)) 
yhat_test = model.predict(X_test)
plt.subplot(1,2,1)
plt.title("Linear Regression")
plt.plot(y_test, 'r.')
plt.plot(yhat_test, 'g.')
plt.legend(['Actual Price', 'Predicted Price'])
plt.subplot(1,2,2)
sns.distplot(y_test - yhat_test)
plt.title("Linear Dist Plot")
plt.show()


# In[26]:


from sklearn import metrics
print('Simple Model')
mean_squared_error = metrics.mean_squared_error(y_test, yhat_test)
print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))
print('R-squared (training) ', round(model.score(X_train, y_train), 3))
print('R-squared (testing) ', round(model.score(X_test, y_test), 3))
print('Intercept: ', model.intercept_)
print('Coefficient:', model.coef_)


# In[27]:


# Random Forest Regression
from sklearn import ensemble
model2 = sklearn.ensemble.RandomForestRegressor()
model2.fit(X_train,y_train)
yhat_test = model2.predict(X_test)

# Plot

fig, axis = plt.subplots(figsize=(13,10)) 
plt.subplot(1,2,1)
plt.title("Random Forest Regression")
plt.plot(y_test, 'r.')
plt.plot(yhat_test, 'g.')
plt.legend(['Actual Price', 'Predicted Price'])
plt.subplot(1,2,2)
sns.distplot(y_test.ravel() - yhat_test)
plt.title("Random Forest Dist Plot")
plt.show()

