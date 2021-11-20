#!/usr/bin/env python
# coding: utf-8

# ## 数据预处理
# * 还原earlies_credit_mon （train2, test）
# * 计算 earlies_credit_mon_diff, issue_date_diff
# * 计算 issue_earlies_diff

# In[1]:


# 数据预处理
# 先将 train_public.csv 另存为 train_public2.csv，并对earlies_credit_mon改成短日期格式 ！！
# 将短日期格式的 2021/12/1 => 2001-12-01 （这里2021应该是系统自动添加上的，实际为 12/1，即月/年）
import pandas as pd
temp = pd.read_csv('./train_public2.csv')
temp['earlies_credit_mon'] = pd.to_datetime(temp['earlies_credit_mon'])
def f(x):
    #print(x)
    if x>= pd.to_datetime('2021-01-01'):
        t = '20' + str(x)[8:10] + '-' + str(x)[5:7] + '-01'
        #print('t=', t)
        return pd.to_datetime(t)
    return x
temp['earlies_credit_mon'] = temp['earlies_credit_mon'].apply(f)
temp.to_pickle('./train_public.pkl')


# In[3]:


# 先将 test_public.csv 另存为 test_public2.csv，并对earlies_credit_mon改成短日期格式 ！！
temp = pd.read_csv('./test_public2.csv')
temp['earlies_credit_mon'] = pd.to_datetime(temp['earlies_credit_mon'])
def f(x):
    #print(x)
    if x>= pd.to_datetime('2021-01-01'):
        t = '20' + str(x)[8:10] + '-' + str(x)[5:7] + '-01'
        #print('t=', t)
        return pd.to_datetime(t)
    return x
temp['earlies_credit_mon'] = temp['earlies_credit_mon'].apply(f)
temp.to_pickle('./test_public.pkl')


# In[9]:


import warnings
import pickle
warnings.filterwarnings('ignore')

# 数据加载
train1 = pd.read_csv('./train_internet.csv')
with open('./train_public.pkl', 'rb') as file:
    train2 = pickle.load(file)
with open('./test_public.pkl', 'rb') as file:
    test = pickle.load(file)
train2


# In[10]:


import datetime

# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train2['issue_date'] = pd.to_datetime(train2['issue_date'])
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train2['issue_date_diff'] = train2['issue_date'].apply(lambda x: x-base_time).dt.days


# In[11]:


train2['earlies_credit_mon'] = pd.to_datetime(train2['earlies_credit_mon'])
# 最小日期为 1952-06-01, 最大日期为 2015-05-01
# 设置初始的时间
base_time = datetime.datetime.strptime('1952-06-01', '%Y-%m-%d')
# 转换为天为单位
train2['earlies_credit_mon_diff'] = train2['earlies_credit_mon'].apply(lambda x: x-base_time).dt.days
train2['issue_earlies_diff'] = (train2['issue_date'] - train2['earlies_credit_mon']).dt.days
train2[['earlies_credit_mon', 'earlies_credit_mon_diff', 'issue_earlies_diff']]


# In[12]:


test['issue_date'] = pd.to_datetime(test['issue_date'])
test['earlies_credit_mon'] = pd.to_datetime(test['earlies_credit_mon'])
# 最小日期为 1959-06-01, 最大日期为 2014-11-01
# 设置初始的时间
base_time = datetime.datetime.strptime('1952-06-01', '%Y-%m-%d')
# 转换为天为单位
test['earlies_credit_mon_diff'] = test['earlies_credit_mon'].apply(lambda x: x-base_time).dt.days
test['issue_earlies_diff'] = (test['issue_date'] - test['earlies_credit_mon']).dt.days
test[['earlies_credit_mon', 'earlies_credit_mon_diff', 'issue_earlies_diff']]


# In[13]:


train2.drop('issue_date', axis = 1, inplace = True)
train2

employer_type = train1['employer_type'].value_counts().index
industry = train1['industry'].value_counts().index
# 标签编码
emp_type_dict = dict(zip(employer_type, [0,1,2,3,4,5]))
industry_dict = dict(zip(industry, [i for i in range(15)]))

# work_year缺失值较多，用 10+ years填充
train2['work_year'].fillna('10+ years', inplace=True)
# 标签编码
work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
train2['work_year']  = train2['work_year'].map(work_year_map)

train2['class'] = train2['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
train2['employer_type'] = train2['employer_type'].map(emp_type_dict)
train2['industry'] = train2['industry'].map(industry_dict)

# 日期类型：issueDate，earliesCreditLine
#train[cat_features]
# 转换为pandas中的日期类型
test['issue_date'] = pd.to_datetime(test['issue_date'])
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
test['issue_date_diff'] = test['issue_date'].apply(lambda x: x-base_time).dt.days
test.drop('issue_date', axis = 1, inplace = True)
test['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
test['work_year']  = test['work_year'].map(work_year_map)
test['class'] = test['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
test['employer_type'] = test['employer_type'].map(emp_type_dict)
test['industry'] = test['industry'].map(industry_dict)


# In[14]:


#train2.info()
#test.info()
train2.columns


# In[15]:


train1.to_pickle('./train1.pkl') # 对应train_interest
train2.to_pickle('./train2.pkl') # 对应train_public
test.to_pickle('./test.pkl') # 对应test_public

