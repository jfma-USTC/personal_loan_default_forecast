# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path="/Users/majiefeng/Desktop/研究生课程/数据挖掘/homework_jfma/个贷违约预测/"
test_data = pd.read_csv(path+"test_public.csv")
train_public_data = pd.read_csv(path+"train_dataset/train_public.csv")
train_internet_data = pd.read_csv(path+"train_dataset/train_internet.csv")


# %%
test_data.info() # 查看基本信息


# %%
train_public_data.info()


# %%
train_internet_data.info()


# %%
# 找出两项训练集间的异同
common_cols = []
for col in train_internet_data.columns:
    if col in train_public_data.columns:
        common_cols.append(col)
    else: continue

train_public_left = list(set(list(train_public_data.columns)) - set(common_cols))
train_internet_left = list(set(list(train_internet_data.columns)) - set(common_cols))

print(common_cols)
print(train_public_left)
print(train_internet_left)


# %%
# 找出public训练集和public测试集间的异同
common_cols = []
for col in test_data.columns:
    if col in train_public_data.columns:
        common_cols.append(col)
    else: continue

train_public_left = list(set(list(train_public_data.columns)) - set(common_cols))
test_left = list(set(list(test_data.columns)) - set(common_cols))

print(common_cols)
print(train_public_left)
print(test_left)


# %%
import re

def workYearDIc(x):
    if str(x)=='nan':
        return -1
    x = x.replace('< 1','0')
    temp = int(re.search('(\d+)', x).group())
    return temp

def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None: # no match, year-month
        return '01-'+val
    month, year = val.split("-")[0], val.split("-")[1]
    return '01-'+month+"-"+year


class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}
timeMax = pd.to_datetime('1-Dec-21')
train_public_data['work_year'] = train_public_data['work_year'].map(workYearDIc)
test_data['work_year'] = test_data['work_year'].map(workYearDIc)
train_public_data['class'] = train_public_data['class'].map(class_dict)
test_data['class'] = test_data['class'].map(class_dict)

train_public_data['earlies_credit_mon'] = pd.to_datetime(train_public_data['earlies_credit_mon'].map(findDig), dayfirst = True)
test_data['earlies_credit_mon'] = pd.to_datetime(test_data['earlies_credit_mon'].map(findDig), dayfirst = True)
train_public_data.loc[ train_public_data['earlies_credit_mon']>timeMax,'earlies_credit_mon' ] = train_public_data.loc[ train_public_data['earlies_credit_mon']>timeMax,'earlies_credit_mon' ]+  pd.offsets.DateOffset(years=-100)  
test_data.loc[ test_data['earlies_credit_mon']>timeMax,'earlies_credit_mon' ] = test_data.loc[ test_data['earlies_credit_mon']>timeMax,'earlies_credit_mon' ]+ pd.offsets.DateOffset(years=-100)
train_public_data['issue_date'] = pd.to_datetime(train_public_data['issue_date'])
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])



#Internet数据处理
train_internet_data['work_year'] = train_internet_data['work_year'].map(workYearDIc)
train_internet_data['class'] = train_internet_data['class'].map(class_dict)
train_internet_data['earlies_credit_mon'] = pd.to_datetime(train_internet_data['earlies_credit_mon'])
train_internet_data['issue_date'] = pd.to_datetime(train_internet_data['issue_date'])


train_public_data['issue_date_month'] = train_public_data['issue_date'].dt.month
test_data['issue_date_month'] = test_data['issue_date'].dt.month
train_public_data['issue_date_dayofweek'] = train_public_data['issue_date'].dt.dayofweek
test_data['issue_date_dayofweek'] = test_data['issue_date'].dt.dayofweek

train_public_data['earliesCreditMon'] = train_public_data['earlies_credit_mon'].dt.month
test_data['earliesCreditMon'] = test_data['earlies_credit_mon'].dt.month
train_public_data['earliesCreditYear'] = train_public_data['earlies_credit_mon'].dt.year
test_data['earliesCreditYear'] = test_data['earlies_credit_mon'].dt.year


###internet数据

train_internet_data['issue_date_month'] = train_internet_data['issue_date'].dt.month
train_internet_data['issue_date_dayofweek'] = train_internet_data['issue_date'].dt.dayofweek
train_internet_data['earliesCreditMon'] = train_internet_data['earlies_credit_mon'].dt.month
train_internet_data['earliesCreditYear'] = train_internet_data['earlies_credit_mon'].dt.year


# %%
# KDE 分布图
dist_cols = 6
dist_rows = len(test_data.columns.drop(['id', 'score']))
plt.figure(figsize=(4*dist_cols, 4*dist_rows))
i = 1
for col in test_data.columns.drop(['id', 'score', 'label']):
    try:
        ax = plt.subplot(dist_rows, dist_cols, i)
        ax = sns.kdeplot(train_public_data[col], color = 'Red', shade=True)
        ax = sns.kdeplot(test_data[col], color = 'Blue', shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax = ax.legend(['train','test'])
        i += 1
    except:
        print(f'变量{col}不是数值型变量')
plt.tight_layout()
plt.show()


