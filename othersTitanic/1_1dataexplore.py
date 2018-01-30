
# coding: utf-8

from __future__ import division
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from numpy.random import randn
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=18)

plt.rc('figure', figsize=(15,10))
np.set_printoptions(precision=4)
# %matplotlib inline

train = pd.read_csv('train.csv')
# PassengerId => 乘客ID
# Pclass => 乘客等级(1/2/3等舱位)
# Name => 乘客姓名
# Sex => 性别
# Age => 年龄
# SibSp => 堂兄弟/妹个数
# Parch => 父母与小孩个数
# Ticket => 船票信息
# Fare => 票价
# Cabin => 客舱
# Embarked => 登船港口


train.head()

train.info()

train.describe()


# # 乘客各个属性分布

fig = plt.figure()
# %matplotlib inline
fig.set(alpha=0.2)#设置图标透明度
plt.rcParams['font.sans-serif']= ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


plt.subplot2grid((2,3),(0,0))
train.Survived.value_counts().plot(kind='bar')
plt.title(u'获救情况（1为获救）', fontproperties=font)
plt.ylabel(u'人数',fontproperties=font)

plt.subplot2grid((2,3),(0,1))
train.Pclass.value_counts().plot(kind='bar')
plt.title(u'乘客等级分布',fontproperties=font)
plt.ylabel(u'人数',fontproperties=font)

plt.subplot2grid((2,3),(0,2))
plt.scatter(train.Survived, train.Age)
plt.title(u'按照年龄看获救分布（1为获救）',fontproperties=font)
plt.ylabel(u'年龄',fontproperties=font)
plt.grid(b=True, which='major', axis='y')

plt.subplot2grid((2,3),(1,0),colspan=2)
train.Age[train.Pclass == 1].plot(kind='kde')
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.xlabel(u'年龄',fontproperties=font)
plt.ylabel(u'密度',fontproperties=font)
plt.title(u'各等级的乘客年龄分布',fontproperties=font)
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best',prop=font)

plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts().plot(kind='bar')
plt.ylabel(u'人数',fontproperties=font)
plt.title(u'各登船口岸上船人数',fontproperties=font)
plt.savefig('1.png')

# plt.show()



# # 属性与获救结果的关联统计

#看看各个乘客等级的获救情况

Survived0 = train.Pclass[train.Survived == 0].value_counts()
Survived1 = train.Pclass[train.Survived == 1].value_counts()

df = pd.DataFrame({'获救':Survived1, '未获救':Survived0})

df.plot(kind='bar', stacked=True)
plt.title(u'各乘客等级的获救情况', fontproperties = font)
plt.xlabel(u'乘客等级', fontproperties = font)
plt.ylabel(u'人数', fontproperties = font)
plt.legend((u'未获救', u'获救'), loc='best',prop=font)
plt.savefig('2.png')
plt.show()
# print df


# # 看看各性别的获救情况


fig = plt.figure()
fig.set(alpha=0.2)

Survivedm = train.Survived[train.Sex == 'male'].value_counts()
Survivedf = train.Survived[train.Sex == 'female'].value_counts()

df = pd.DataFrame({'女性':Survivedf, '男性':Survivedm})
df.plot(kind='bar', stacked=True)
plt.title(u'不同性别乘客的获救情况', fontproperties = font)
plt.xlabel(u'乘客性别', fontproperties = font)
plt.ylabel(u'人数', fontproperties = font)
plt.legend((u'女性', u'男性'), loc='best',prop=font)
plt.savefig('3.png')
plt.show()


# # 各种船舱级别情况下各个性别的获救情况


fig = plt.figure()
fig.set(alpha=0.65)
plt.title(u'根据舱等级和性别的获救情况')


ax1=fig.add_subplot(141)
train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels(['1','0'], rotation=0,fontproperties=font)
ax1.legend([u"女性/高级舱"], loc='best', prop=font)

ax2=fig.add_subplot(142, sharey=ax1)
train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts().plot(kind='bar', label='female lowclass', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0,fontproperties=font)
plt.legend([u"女性/低级舱"], loc='best', prop=font)


ax3 = fig.add_subplot(143, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts().plot(kind='bar', label='male highclass', color='#EA2222')
ax3.set_xticklabels([u'未获救',u'获救'], rotation=0,fontproperties=font)
ax3.legend([u"男性/高级舱"], loc='best', prop=font)

ax4 = fig.add_subplot(144, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts().plot(kind='bar', label='female lowclass', color='#FAEE04')
ax4.set_xticklabels([u'未获救', u'获救'], rotation=0,fontproperties=font)
ax4.legend([u"男性/低级舱"], loc='best', prop=font)
plt.savefig('4.png')
plt.show()

# # 各个登船港口的乘客的获救情况


fig = plt.figure()
fig.set(alpha=0.4)

Survived0 = train.Survived[train.Embarked =='S'].value_counts()
Survived1 = train.Survived[train.Embarked =='C'].value_counts()
Survived2 = train.Survived[train.Embarked =='Q'].value_counts()

df = pd.DataFrame({'S':Survived0, 'C':Survived1, 'Q':Survived2})
df.plot(kind='bar', stacked=True)
plt.title(u'各个登船港口的乘客的获救情况', fontproperties = font)
plt.xlabel(u'是否获救', fontproperties = font)
plt.ylabel(u'人数', fontproperties = font)
plt.legend(('S', 'C','Q'), loc='best',prop=font)
plt.savefig('5.png')
plt.show()



fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df=pd.DataFrame({ u'未获救':Survived_0,u'获救':Survived_1,})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况",fontproperties=font)
plt.xlabel(u"登录港口",fontproperties=font) 
plt.ylabel(u"人数",fontproperties=font) 
plt.legend((u'未获救',u'获救'),loc='best',prop=font)
plt.savefig('6.png')
plt.show()


# # 堂兄弟/妹，孩子/父母有几人对是否获救是否有影响


g = train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print df



g1 = train.groupby(['Parch','Survived'])
df1 = pd.DataFrame(g1.count()['PassengerId'])
print df1


# In[32]:

#看一下船票编号
len(train[train.Cabin.notnull()])


#cabin只有204个乘客有值，先看看它的分布
train.Cabin.value_counts()


# # 按Cabin有无看获救情况


fig = plt.figure()
fig.set(alpha=0.2)


Survived_ca = train[train.Cabin.notnull()].Survived.value_counts()
Survived_noca = train[train.Cabin.isnull()].Survived.value_counts()
# 与下面等价
# Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
# Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
# 与下面等价
# Survived_ca = train.Survived[train.Cabin.notnull()==True].value_counts()
# Survived_noca = train.Survived[train.Cabin.isnull()==True].value_counts()
# train.Survived[train.Sex == 'female'].value_counts()

df = pd.DataFrame({u'获救':Survived_ca, u'未获救':Survived_noca}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u'按Cabin有无查看获救情况', fontproperties = font)
plt.xlabel(u'是否获救', fontproperties = font)
plt.ylabel(u'人数', fontproperties = font)
plt.legend((u'无' ,u'有'), loc='best',prop=font)
plt.savefig('7.png')
plt.show()



# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# Survived_0 = train.Embarked[train.Survived == 0].value_counts()
# Survived_1 = train.Embarked[train.Survived == 1].value_counts()
# df=pd.DataFrame({ u'未获救':Survived_0,u'获救':Survived_1,})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各登录港口乘客的获救情况",fontproperties=font)
# plt.xlabel(u"登录港口",fontproperties=font) 
# plt.ylabel(u"人数",fontproperties=font) 
# plt.legend((u'未获救',u'获救'),loc='best',prop=font)
# plt.show()


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况", fontproperties=font)
plt.xlabel(u"Cabin有无", fontproperties=font) 
plt.ylabel(u"人数", fontproperties=font)
plt.legend((u'有' ,u'无'), loc='best',prop=font)
plt.savefig('8.png')
plt.show()

