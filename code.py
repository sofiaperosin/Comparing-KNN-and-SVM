#!/usr/bin/env python
# coding: utf-8

# **MODULES**

# In[1]:


import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as acc

import matplotlib.pyplot as plt
import plotly.graph_objects as go


# **1**

# In[2]:


data = load_wine()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']


# **2**

# In[3]:


df1 = df.iloc[:,0:2]
#df1 = df[["color_intensity","flavanoids"]]
#df1 = df[["ash","nonflavanoid_phenols"]]
target=df['target']


# In[4]:


for lab in target.unique():
    plt.scatter(df1.iloc[:,0][target==lab],df1.iloc[:,1][target==lab],label=lab)
plt.legend()
plt.xlabel('alcohol')
plt.ylabel("malic_acid")
plt.savefig("data representations")
plt.show()


# **3**

# In[5]:


x_train, x, y_train, y = train_test_split(df1,target,test_size=0.5,train_size=0.5,stratify=target,random_state=2)
x_validation, x_test, y_validation, y_test = train_test_split(x,y,test_size = 0.6,train_size =0.4,stratify=y,random_state=2)


# In[6]:


#check correctly proportion
prop_train=len(x_train)/len(df1)
prop_val=len(x_validation)/(len(df1))
prop_test=len(x_test)/len(df1)
print(prop_train)
print(prop_val)
print(prop_test)


# In[7]:


#check correctly distribution class 1 2 3 among train, test, validation
def calcola(lista):
    tot=len(lista)
    zero=len(lista[lista==0])
    uno=len(lista[lista==1])
    due=len(lista[lista==2])
    print(f"Percentuale 0: {zero/tot}")
    print(f"Percentuale 1: {uno/tot}")
    print(f"Percentuale 2: {due/tot}")

calcola(target)
print()
calcola(y_train)
print()
calcola(y_test)
print()
calcola(y_validation)


# **4**

# In[8]:


k_list=[1,3,5,7]


# In[9]:


def plot_knn(k,X,y):
    h = .02
    
    clf = KNeighborsClassifier(k)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap="Set3")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Classification (k = %i)"% (k))
    
    plt.savefig(f"KNN {k}")


# In[10]:


for k in k_list:
    plot_knn(k,x_train.to_numpy(),y_train.to_numpy())


# In[11]:


plot_knn(15,x_train.to_numpy(),y_train.to_numpy())


# In[12]:


acc_knn_list=[]
for k in k_list:
    clf = KNeighborsClassifier(k)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_validation)
    acc_score=acc(y_validation,y_pred)
    acc_knn_list.append(acc_score)


# In[13]:


acc_knn_list


# In[2]:


acc_knn_list=[0.8, 0.9428571428571428, 0.8857142857142857, 0.9142857142857143]
k_list=[1,3,5,7]

values = [k_list,acc_knn_list]
fig = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [1,1],
  header = dict(
    values = [['<b>K</b>'],['<b>Accuracy</b>']],
    line_color='black',
    line_width=3,
    fill_color='white',
    align=['left','center'],
    font=dict(color='black', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='black',
    fill=dict(color=[['white','yellow','white','white'], ['white','yellow','white','white']]),
    align=['left', 'center'],
    font_size=12,
    height=30)
    )
])

fig.show()


# **5**

# In[14]:


fig, ax = plt.subplots()
ax.plot(k_list, acc_knn_list)

ax.set(xlabel='k', ylabel='accuracy',title='Accuracy on Validation')
ax.grid()
plt.savefig("accuracy KNN")
plt.show()


# **7**

# In[15]:


max_knn=k_list[np.array(acc_knn_list).argmax()]
max_knn


# In[16]:


clf = KNeighborsClassifier(max_knn)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc_score=acc(y_test,y_pred)
acc_score


# **8**

# In[17]:


c_list=[0.001,0.01,0.1,1,10,100,1000]


# In[18]:


def plot_svm(c,X,y,conto):
    h = .02
    
    clf = SVC(kernel="linear",C=c)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap="Set3")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Classification(c={c})")
    
    plt.savefig(f"{conto}")


# In[19]:


conto=0
for c in c_list:
    plot_svm(c,x_train.to_numpy(),y_train.to_numpy(),conto)
    conto=conto+1


# In[20]:


acc_svm_list=[]
for c in c_list:
    clf = SVC(kernel="linear",C=c)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_validation)
    acc_score=acc(y_validation,y_pred)
    acc_svm_list.append(acc_score)


# **9**

# In[21]:


fig, ax = plt.subplots()
ax.plot(c_list, acc_svm_list)

ax.set(xlabel='C', ylabel='accuracy',title='Accuracy on Validation')
ax.grid()
plt.savefig("Linear SVM acc")
plt.show()


# In[33]:


acc_svm_list


# In[42]:


acc_svm_list=[0.4,0.4857142857142857,0.8285714285714286,0.8857142857142857,0.8285714285714286,0.8285714285714286,0.8571428571428571]
c_list=[0.001,0.01,0.1,1,10,100,1000]

values = [c_list,acc_svm_list]
fig = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [1,1],
  header = dict(
    values = [['<b>C</b>'],['<b>Accuracy</b>']],
    line_color='black',
    line_width=3,
    fill_color='white',
    align=['left','center'],
    font=dict(color='black', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='black',
    fill=dict(color=[['white','white','white','yellow','white','white'], ['white','white','white','yellow','white','white']]),
    align=['left', 'center'],
    font_size=12,
    height=30)
    )
])
plt.savefig("SVM table")
fig.show()


# **11**

# In[22]:


c_list[np.array(acc_svm_list).argmax()]


# In[23]:


clf = SVC(kernel="linear",C=1)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc_score=acc(y_test,y_pred)
acc_score


# **12**

# In[24]:


def plot_rbf_svm(c,X,y,conto):
    h = .02
    
    clf = SVC(kernel="rbf",C=c)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap="Set3")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Classification(c={c})")
    
    plt.savefig(f"RBF {conto}")


# In[25]:


conto=0
for c in c_list:
    plot_rbf_svm(c,x_train.to_numpy(),y_train.to_numpy(),conto)
    conto=conto+1


# In[26]:


acc_rbf_list=[]
for c in c_list:
    clf = SVC(kernel="rbf",C=c)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_validation)
    acc_score=acc(y_validation,y_pred)
    acc_rbf_list.append(acc_score)


# In[27]:


acc_rbf_list


# In[44]:


acc_rbf_list=[0.4,0.4,0.9142857142857143,0.9428571428571428,0.9142857142857143, 0.8857142857142857,0.8571428571428571]
c_list=[0.001,0.01,0.1,1,10,100,1000]

values = [c_list,acc_rbf_list]
fig = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [1,1],
  header = dict(
    values = [['<b>C</b>'],['<b>Accuracy</b>']],
    line_color='black',
    line_width=3,
    fill_color='white',
    align=['left','center'],
    font=dict(color='black', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='black',
    fill=dict(color=[['white','white','white','yellow','white','white'], ['white','white','white','yellow','white','white']]),
    align=['left', 'center'],
    font_size=12,
    height=30)
    )
])
plt.savefig("RBF table")
fig.show()


# In[28]:


fig, ax = plt.subplots()
ax.plot(c_list, acc_rbf_list)

ax.set(xlabel='C', ylabel='accuracy',title='Accuracy on Validation')
ax.grid()
plt.savefig("Accuracy RGB C")
plt.show()


# In[29]:


max_rbf_c=c_list[np.array(acc_rbf_list).argmax()]
max_rbf_c


# In[30]:


clf = SVC(kernel="rbf",C=max_rbf_c)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc_score=acc(y_test,y_pred)
acc_score


# **15**

# In[31]:


c_list=[0.1,1,2,5,10]
gamma_list=np.linspace(0.0000001,2,100)
best_c=0
best_gamma=0
max_acc=0
for c in c_list:
    for gam in gamma_list:
        clf = SVC(kernel="rbf",C=c,gamma=gam)
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_validation)
        acc_score=acc(y_validation,y_pred)
        if acc_score>max_acc:
            best_c=c
            best_gamma=gam
            max_acc=acc_score


# In[32]:


print(max_acc)
print(best_c)
print(best_gamma)


# In[33]:


clf = SVC(kernel="rbf",C=1,gamma=0.74747481010101)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc_score=acc(y_test,y_pred)
acc_score


# In[34]:


def plot_rbf_spec_svm(c,g,X,y):
    h = .02
    
    clf = SVC(kernel="rbf",C=c,gamma=g)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap="Set3")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    g1 = "{:.2f}".format(g)
    plt.title(f"Classification (c={c}, γ={g1})")
    
    plt.savefig("gamma e c")
    plt.show()


# In[35]:


plot_rbf_spec_svm(1,0.74747481010101,x_train.to_numpy(),y_train.to_numpy())


# **16**

# In[36]:


x_train_2=x_train.append(x_validation)
y_train_2=y_train.append(y_validation)


# In[37]:


parameters = {'gamma':np.linspace(0,2,100), 'C':[0.1,1,2,5,10]}
svc = SVC(kernel="rbf")
clf = GridSearchCV(svc, parameters,cv=5,scoring="accuracy")
clf.fit(x_train_2,y_train_2)


# In[38]:


clf.best_estimator_


# In[39]:


clf.best_score_


# In[40]:


clf = SVC(kernel="rbf",C=1,gamma=0.787878787878788)
clf.fit(x_train_2,y_train_2)
y_pred=clf.predict(x_test)
acc_score=acc(y_test,y_pred)
acc_score


# **16.b**

# In[41]:


x_train_2=x_train.append(x_test)
y_train_2=y_train.append(y_test)


# In[43]:


parameters = {'gamma':np.linspace(0,2,100), 'C':[0.1,1,2,5,10]}
svc = SVC(kernel="rbf")
clf = GridSearchCV(svc, parameters,cv=5,scoring="accuracy")
clf.fit(x_train_2,y_train_2)


# In[44]:


print(clf.best_estimator_)
print()
print(clf.best_score_)


# In[45]:


clf = SVC(kernel="rbf",C=10,gamma=0.04040404040404041)
clf.fit(x_train_2,y_train_2)
y_pred=clf.predict(x_validation)
acc_score=acc(y_validation,y_pred)
acc_score


# **COMPARING**

# In[46]:


final_name=["KNN","Linear SVM","RBF SVM C","RBF SVM C AND GAMMA "]
final_value=[0.7222222222222222,0.6666666666666666,0.7407407407407407,0.7222222222222222]
plt.scatter(final_name,final_value)
plt.ylabel("Final accuracy")
plt.title("Alcohol - Malic_aid")
plt.savefig("alcohol -mali")
plt.ylim(0.5,0.95)
plt.savefig("final")


# In[52]:


acc=[0.7222222222222222,0.6666666666666666,0.7407407407407407,0.7222222222222222]
alg=["KNN","Linear SVM","RBF SVM C","RBF SVM C AND GAMMA "]

values = [alg,acc]
fig = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [1,1],
  header = dict(
    values = [['<b>Algorithms</b>'],['<b>Accuracy</b>']],
    line_color='black',
    line_width=3,
    fill_color='white',
    align=['left','center'],
    font=dict(color='black', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='black',
    fill=dict(color=[['white','white','yellow','white'], ['white','white','yellow','white']]),
    align=['left', 'center'],
    font_size=12,
    height=30)
    )
])
fig.show()


# In[47]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[48]:


df2 = df.iloc[:,0:-1]
target=df["target"]


# In[49]:


bestfeatures = SelectKBest(score_func=chi2, k=2)
fit = bestfeatures.fit(df2,target)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(df2.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Attribute','Score'] 
res=featureScores.nlargest(13,'Score')
res

