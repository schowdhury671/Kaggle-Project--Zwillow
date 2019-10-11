
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import gc
import datetime
import xgboost as xgb
from sklearn import preprocessing 
# get_ipython().magic(u'matplotlib inline')


# In[2]:

# reads csv files into dataframe for further processing
right_df=pd.read_csv('properties_2016.csv',header=0)
left_df=pd.read_csv('train_2016_v2.csv',header=0)
left_df.head()


# In[3]:


# right_df.head()
# print right_df


# In[4]:

#merges the left and the right dataframes and forms a combined df 
train_df=pd.merge(left_df,right_df,how='left',on='parcelid')
train_df.head()


# In[5]:

train_df.shape


# In[6]:

train_df.to_csv('final_train_data.csv')


# In[7]:

del left_df
del right_df


# In[8]:

train_df.describe()


# In[9]:

missing_val_count=train_df.isnull().sum(axis=0).reset_index()
missing_val_count.columns=['attribute','count']

ind=np.arange(missing_val_count.shape[0])
fig, ax = plt.subplots(figsize=(16,22))
ax.barh(ind,missing_val_count['count'].values,color='green')
ax.set_yticks(ind)
ax.set_yticklabels(missing_val_count['attribute'].values)
plt.show()
plt.show()  # added a plot for better visualization

# In[10]:1.5

number_of_entries=train_df.shape[0]
rem_col_names=missing_val_count[missing_val_count['count']>number_of_entries/1.5].attribute.values.tolist()
# Now we have the column names which are required, rest of the columns can be removed from train_df 

train_df.drop(rem_col_names, axis=1,inplace=True)
test_df.drop(rem_col_names, axis=2,inplace=False) # for testing the model  



for f in train_df.columns: 
    train_df[f]=train_df[f].fillna(-1)
    if train_df[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))

train_df.shape
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.4 ]

# In[11]:

train_df['transactionmonth'] = pd.DatetimeIndex(train_df['transactiondate']).month
fill_func = lambda x: x.fillna(x.mean())
grouped_df=train_df.groupby('transactionmonth').transform(fill_func)
grouped_df.head()


# In[12]:

grouped_df['transactionmonth']=train_df['transactionmonth']
grouped_df.head()


# In[4]:

new_grouped_df=grouped_df.drop(['parcelid'],axis=1)
new_grouped_df.shape
new_grouped_df.head()




new_grouped_df.shape


# In[5]:

x_train=new_grouped_df.drop(['logerror','transactionmonth','transactiondate'],axis=1)
y_train=new_grouped_df['logerror']
y_mean=y_train.mean()
print (x_train.shape, y_train.shape)


# In[6]:

right_df=pd.read_csv('properties_2016.csv',header=0)
test_df=pd.read_csv('sample_submission.csv',header=0)
test_df.rename(columns={'ParcelId':'parcelid'},inplace=True)
new_test_df=pd.merge(test_df,right_df,how='left',on='parcelid')
del right_df
del test_df
new_test_df.shape
print('After getting train files')


# In[7]:

train_cols = list(x_train.columns)

new_test_df=new_test_df[train_cols]

missing_val_count=new_test_df.isnull().sum(axis=0).reset_index()

missing_val_count.columns=['attribute','count']

number_of_entries=new_test_df.shape[0]

rem_col_names=missing_val_count[missing_val_count['count']>number_of_entries/1.5].attribute.values.tolist()



missing_val_count
print('After getting train files2')
new_test_df.head(1)


# In[8]:

x_test=new_test_df.fillna(new_test_df.mean())
x_test.shape


print('After getting train files3')
# In[9]:

xgb_params = {
    'eta': 0.06,
    'max_depth': 5,
    'subsample': 0.75,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}


xgb_params = {
    'eta': 0.76,
    'max_depth': 6,
    'subsample': 0.79,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': x_mean,
    'silent': 2
}



# In[10]:
print ('Before forming Dmatrix')
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
print ('After forming Dmatrix')

x_test.shape
x_train.shape


# In[11]:

cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=500,
                   early_stopping_rounds=5,
                   verbose_eval=10, 
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)


# In[12]:

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
pred = model.predict(dtest)


# In[13]:

y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)


# In[14]:

pids=pd.read_csv('sample_submission.csv',header=0)['ParcelId']
output = pd.DataFrame({'ParcelId': pids.astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
# output.head()
cols = output.columns.tolist()
print (cols)
cols = cols[-1:] + cols[:-1]
# print cols


# In[15]:

print (output)


# In[16]:

output.to_csv('submission_v015.csv', index=False)


# In[17]:

import csv

with open('submission_v015.csv', 'r') as infile, open('reordered_15.csv', 'a') as outfile:
    # output dict needs a list for new column ordering
    fieldnames = ['ParcelId', '201610', '201611', '201612', '201710','201711', '201712']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    # reorder the header first
    writer.writeheader()
    for row in csv.DictReader(infile):
        # writes the reordered rows to the new file
        writer.writerow(row)

