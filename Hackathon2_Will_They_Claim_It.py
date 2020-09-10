#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score,make_scorer,classification_report
from sklearn.ensemble import ExtraTreesClassifier

from jupyterthemes import jtplot
jtplot.style()


# In[2]:


# Unzip the file
#!unzip file.zip


# Read the CSV 

# In[2]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df = df_train


# # EDA Visualization

# In[3]:


Data_per_country = df.groupby(["Destination"])["Claim"].sum().reset_index().sort_values("Claim",ascending=False).reset_index(drop=True)
top10_claim_countries = Data_per_country.iloc[:20]
top10_claim_countries

plt.figure(figsize=[10,8])
sns.barplot(top10_claim_countries['Claim'],top10_claim_countries['Destination'],saturation=.85,errcolor='.045',ci=None)
plt.title('Destination vs Claim',fontsize=15,loc='center')
plt.xticks(rotation = 90)


# In[4]:


import matplotlib.pyplot as plt

# Data to plot
labels = 'Online', 'Offline'
s1 = (df['Distribution Channel']=='Online').sum()
# s1.value_counts()
s2 = (df['Distribution Channel'] =='Offline').sum()
s2
data1 = [s1, s2]
colors = ['b', 'r']
#explode = (0.1, 0,)  # explode 1st slice
fig = plt.figure(figsize =(10, 7)) 
plt.pie(data1, labels = labels, colors= colors,autopct='%18.2f%%') 
plt.title('% of Distribution Channel')
plt.show() 


# In[5]:


plt.figure(figsize=(10,7))
df['Claim'].value_counts().plot.pie(figsize=(6,4),autopct='%.1f')
plt.title('Total % of claim')


# In[6]:


plt.figure(figsize=(9,7))
sns.heatmap(df.corr(), annot=True, fmt = '.2g')


# In[7]:


sns.scatterplot(df['Age'], df['Claim'])


# In[8]:


d1 = df['Age']
d2 = df['Net Sales']
d3 = df['Commision (in value)']
d4 = df['Duration']
fig = plt.figure(figsize =(8,6)) 
# Creating plot 
sns.boxplot(d1,hue=df['Claim'], color='y')
plt.show() 


# In[9]:


sns.boxplot(d2,hue=df['Claim'],color='y')
plt.show() 


# In[10]:


sns.boxplot(d3,hue=df['Claim'],color='g')
plt.show() 


# In[11]:


sns.boxplot(d4,hue=df['Claim'], color='y')
plt.show() 


# # EDA

# In[12]:


df_train.shape


# In[13]:


df_test.shape


# In[14]:


df_train.head(5)


# In[15]:


df_test.head(5)


# In[16]:


df_train.info()


# * check the null values 

# In[17]:


df_train.isna().sum()


# In[18]:


df_test.isna().sum()


# * Description of Dataset

# In[19]:


df_train.describe()


# In[20]:


df_test.describe()


# * Check the counts of agencies

# In[21]:


df_train['Agency'].groupby(df['Agency']).count()


# * Check the counts of product_names

# In[22]:


df_train['Product Name'].groupby(df['Product Name']).count()


# In[23]:


df['Age'].mean()


# In[24]:


# To show full records
pd.set_option("display.max_rows", None, "display.max_columns", None)
df_train['Distribution Channel'].value_counts()


# ## Feature Engg. 

# In[25]:


def improve_data(data, age_mean, duration_mean):
  data['Duration'] = abs(data['Duration'])
  data['Net Sales'] = abs(data['Net Sales'])
  data['Age'] = data.apply(lambda x: age_mean if x['Age'] > 100 else x['Age'], axis=1)
  data['Duration'] = data.apply(lambda x: duration_mean if x['Duration'] > 500 
                                else x['Duration'], axis=1)
  data['Agency'] = data.apply(lambda x: 'OTH' if x['Agency'] not in ['C2B','EPX','CWT','JZI']
                                else x['Agency'], axis=1)
  data_Product = data['Product Name'].value_counts().rename_axis('Product Name').reset_index(name='counts')
  data_Product = data_Product[data_Product['counts'] > 1000]
  #data_Product_LS = data_Product[data_Product['counts'] <= 1000]
  data['Product Name'] = data.apply(lambda x: 'High Sell' if x['Product Name'] in 
                                        list(data_Product['Product Name'])                                         
                                else 'Low Sell', axis=1)
  data_dest = data['Destination'].value_counts().rename_axis('Destination').reset_index(name='counts')
  data_dest = data_dest[data_dest['counts'] > 1000]
  data['Destination'] = data.apply(lambda x: 'Frequent' if x['Destination'] in 
                                        list(data_dest['Destination'])                                         
                                else 'Less Frequent', axis=1)


# In[26]:


def get_merge_data(data, cat_columns, num_coulmns):
  #data = data.round({'Net Sales': 2, 'Commision (in value)': 2})
  df_category = data[cat_columns]
  x_enc = pd.get_dummies(df_category, columns=cat_columns)
  print(x_enc.shape)
  df_numeric = data[num_columns]
  merged_data = pd.concat([df_numeric,x_enc],axis=1)
  print(merged_data.shape)
  return merged_data

def get_features(data, y, count):
  model_features_importance=ExtraTreesClassifier()
  model_features_importance.fit(data,y)
  print(model_features_importance.feature_importances_)
  ranked_features=pd.Series(model_features_importance.feature_importances_,index=data.columns)
  ranked_features.nlargest(count).plot(kind='barh')
  return ranked_features.nlargest(count)


# In[27]:


df_s = df[['Agency','Agency Type','Product Name','Destination','Net Sales','Commision (in value)']]
f_df = pd.get_dummies(df_s,columns=['Agency', 'Agency Type','Product Name','Destination'],drop_first=True)
x1 = f_df#.drop(['Claim','ID'],axis=1)
y1 = df['Claim']
model_features_importance = ExtraTreesClassifier()
model_features_importance.fit(x1,y1)
print(model_features_importance.feature_importances_)
ranked_features = pd.Series(model_features_importance.feature_importances_,index=x1.columns)
ranked_features.nlargest(15)#.plot(kind='barh')
top15 = ['Net Sales', 'Commision (in value)', 'Agency_C2B',
       'Destination_SINGAPORE', 'Agency Type_Travel Agency',
       'Product Name_Annual Silver Plan', 'Agency_EPX',
       'Product Name_Bronze Plan', 'Product Name_Cancellation Plan',
       'Product Name_Silver Plan', 'Product Name_2 way Comprehensive Plan',
       'Destination_CHINA', 'Destination_UNITED STATES', 'Agency_LWC',
       'Agency_JZI']
n_df1 = f_df[top15]
df_corr = pd.concat([n_df1, df['Claim']],axis=1)

plt.figure(figsize=(14,11))
sns.heatmap(df_corr.corr(), annot=True,
            linewidths=.25,
            square = True,
            vmin=-1, vmax=1,
            center= 0,
            fmt='.1g')


# In[ ]:





# In[28]:


df_train_du = df_train
df_test_du = df_test
df_train_du.columns


# In[29]:


improve_data(df_train_du, df_train_du['Age'].mean(), df_train_du['Duration'].mean())
df_train_du.describe()


# In[30]:


#df_train_du['Destination'] == 'Less Frequent'
df_train_du['Destination'].value_counts()


# In[31]:


df_train_du['Product Name'].value_counts()


# * Features Selection for model training

# In[32]:


cat_columns = ['Agency', 'Agency Type', 'Product Name', 'Destination']#,'Agency Type','Destination'
num_columns = ['Net Sales','Commision (in value)'] #'Commision (in value)'
y = df_train_du['Claim']
merged_data = get_merge_data(df_train_du, cat_columns, num_columns)
ranked_features = get_features(merged_data, y, 100)
merged_data_fe = merged_data[ranked_features.index]


# In[33]:


merged_data.shape


# * Split the data

# In[34]:


X_train,X_test,y_train,y_test = train_test_split(merged_data,y,test_size=0.1,random_state=43)


# Checking for Class imbalance and tackling it.
# 
# * For Claim 1, output label=1, for any other claim output label=0

# In[35]:


y_train=((y_train==1).astype(int))  
y_test=((y_test==1).astype(int))
y_train.value_counts()


# In[36]:


y_test.value_counts()


# #  Model Training 

# In[37]:


# Train the model
from sklearn.ensemble import RandomForestClassifier

#reg=RandomForestClassifier(n_estimators=300, max_depth=50, class_weight={0:1,1:1}, n_jobs=-1).fit(X_train,y_train)
#reg=RandomForestClassifier(n_estimators=300, max_depth=50, class_weight={0:830,1:1}).fit(X_train,y_train)
reg=RandomForestClassifier(n_estimators=60, max_depth=40, class_weight={0:30,1:1}).fit(X_train,y_train)
y_pred=reg.predict(X_train)
predict_score=precision_score(y_train,y_pred)
print('Precision Score for train is :',predict_score)
con_matrix = confusion_matrix(y_train,y_pred)
print('Confusion Matrix for train is :',con_matrix)
accuracy = accuracy_score(y_train,y_pred)
print('Accuracy is :',accuracy)

y_predict=reg.predict(X_test)
predict_score=precision_score(y_test,y_predict)


# # Model Results

# In[39]:


print('Precision Score for test is :',predict_score)
accuracy = accuracy_score(y_test,y_predict)
print('Accuracy is :',accuracy)
con_matrix = confusion_matrix(y_test,y_predict)
print('Confusion Matrix is :',con_matrix)
classification_report_ = classification_report(y_test,y_predict)
print(classification_report_)


# In[ ]:





# # Prediction on test data

# In[40]:


# Predict the Test
improve_data(df_test_du, df_test_du['Age'].mean(), df_test_du['Duration'].mean())
#df_test_du.describe()
#create_new_feature(df_test_1)
merged_test_data = get_merge_data(df_test_du, cat_columns, num_columns)
#merged_test_data_fe = merged_test_data[ranked_features.index]
y_test_pred = reg.predict(merged_test_data)
y_test_pred


# * Creating dataframe of output and ID 

# In[41]:


output = pd.DataFrame(zip(df_test['ID'], y_test_pred), columns=['ID','Claim'])

output.to_csv('AMRS_results.csv',index=False)
