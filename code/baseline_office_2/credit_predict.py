#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 数据探索
import pickle
import pandas as pd

# 加载之前保存好的 train1_data, train2_data, test_data (在Version2中已经进行了保存)
with open('./train2.pkl', 'rb') as file:
    train2 = pickle.load(file)
with open('./test.pkl', 'rb') as file:
    test = pickle.load(file)
# 去掉policy_code字段
train2.drop(['policy_code'], axis=1,inplace=True)
test.drop(['policy_code'], axis=1,inplace=True)
test


# In[2]:


# earlies_credit_mon:借款人最早报告的信用额度开立的月份
X_train2 = train2.drop(['isDefault','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train2 = train2['isDefault']
X_test = test[X_train2.columns]


# In[7]:


## 构造新特征
import catboost as cb
cat_model = cb.CatBoostClassifier(iterations=3000, 
                              depth=7, 
                              learning_rate=0.001, 
                              loss_function='Logloss',
                              eval_metric='AUC',
                              logging_level='Verbose', 
                              metric_period=50)
cat_model.fit(X_train2, y_train2, eval_set=(X_train2, y_train2))


# In[8]:


cat_pred = cat_model.predict_proba(X_test)[:, 1]
cat_pred


# In[9]:


# submission
submission = pd.DataFrame({'id':test['loan_id'], 'isDefault':cat_pred})
submission.to_csv('baseline_cat1.csv', index = None)
submission

