
# coding: utf-8

# # INF552 Assignment1 KNN Classification
# ### Tong Lyu 1076139647
# 
# ## 1 Introduction
# ### 1.1 Libraries

# In[1]:


import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from sklearn import neighbors
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import classification_report, confusion_matrix  
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Pre-Processing and Exploratory data analysis
# ### 2.1 Load Data and  make scatterplots
# The table of abbreviations and corresponding independent variables is as follows:
# 
# |Independent variables|Abbreviations
# | :------: | :------: |
# |pelvic incidence|PEI
# |pelvic tilt|PET
# |lumbar lordosis angle|LLA
# |sacral slope|SAS
# |pelvic radius|PER
# |grade of spondylolisthesis|GOS

# In[2]:


columns = ["PEI","PET","LLA","SAS","PER","GOS","CLASS"]
data = pd.read_csv("./data/column_2C.dat",sep=" ",header=None, names=columns)
plt.figure(figsize=(6,6))
sns.pairplot(data,hue="CLASS",size=3, diag_kind='kde')
plt.show()


# ### 2.2 Make boxplots for each of the independent variables

# In[3]:


attributes = ["PEI","PET","LLA","SAS","PER","GOS"]
plt.figure(figsize=(12,9))
plt.subplots_adjust(hspace=0.3, wspace=0.5)
arrange = 230
for attr in attributes:
    arrange += 1
    plt.subplot(arrange)
    sns.boxplot(data=data,x="CLASS",y=attr)


# ### 2.3 Select training dataset and test dataset
# Select the ﬁrst 70 rows of Class 0 and the ﬁrst 140 rows of Class 1 as the training set and the rest of the data as the test set.

# In[4]:


def tranCat2Num(category):
    if (category == "AB"):
        return 1
    else:
        return 0
    
data["CLASS"] = data.apply(lambda row: tranCat2Num(row["CLASS"]), axis=1)
train_AB = data.loc[0:139]
test_AB = data.loc[140:209]
train_NO = data.loc[210:279]
test_NO = data.loc[280:309]
train = pd.concat([train_AB,train_NO])
test = pd.concat([test_AB, test_NO])


# ## 3. Classiﬁcation using KNN
# ### 3.1 Explore effect of neighbors K
# 
# In this session, we built KNN classifier with Euclidean distance and different numbers of neighbors, k ∈ {208, 207, . . . , 3, 2, 1, }.   
# The result showed the optimal neighbors for this set up is 4.

# In[5]:


def knn_k():
    train_err = []
    test_err = []
    nei = np.arange(1,209)

    for k in nei:
        knn_k = neighbors.KNeighborsClassifier(k, metric="euclidean")
        knn_k.fit(train[attributes], train["CLASS"])

        train_err.append(1 - knn_k.score(train[attributes], train["CLASS"]))
        test_err.append(1 - knn_k.score(test[attributes], test["CLASS"]))

    opt_k = nei[test_err.index(min(test_err))]
    nei_list = nei.tolist()
    low_TER = train_err[nei_list.index(opt_k)]
    return nei,train_err,test_err,opt_k, low_TER

nei,train_err, test_err, opt_k, low_TER = knn_k()
low = {}
low["1st Euclidean KNN"] = low_TER

fig = plt.figure(figsize=(8,6))
plt.plot(nei, train_err, label='Train')
plt.plot(nei, test_err, label='Test')
plt.legend(loc='lower right')
plt.xlim(0,208)
plt.xlabel('Neighbors (k) in KNN')
plt.ylabel('Error rate')
plt.title("Error rate against the number of neighbors in KNN")
ax = plt.gca()
ax.invert_xaxis()


# In[6]:


opt = neighbors.KNeighborsClassifier(opt_k, metric="euclidean")
opt.fit(train[attributes], train["CLASS"])
pred = opt.predict(test[attributes])
con_matrix = confusion_matrix(test["CLASS"], pred) 
TP = con_matrix[1][1]
FP = con_matrix[0][1]
TN = con_matrix[0][0]
FN = con_matrix[1][0]

print("The optimal number of neighbors(k) is " + str(opt_k) +".\n")
print("The confusion matrix for test data (k=%d) is:"%opt_k)
print("\t| predicted AB\t| predicted NO")
print("real AB\t| "+str(TP)+"\t\t| "+str(FN))
print("real NO\t| "+str(FP) +"\t\t| "+str(TN))

TP_rate = float(TP)/(TP + FN)
TN_rate = float(TN)/(FP+ TN)
precision = float(TP)/(TP + FP)
recall = float(TP)/(TP + FN)
F1_score = (2 * precision * recall) / (precision + recall)

print("\nTrue positive rate = %.2f" %TP_rate)
print("True negotive rate = %.2f" %TN_rate)
print("Precision = %.2f" %precision)
print("F1 score = %.2f" %F1_score)


# In[7]:


pred = opt.predict(train[attributes])
con_matrix = confusion_matrix(train["CLASS"], pred) 
TP = con_matrix[1][1]
FP = con_matrix[0][1]
TN = con_matrix[0][0]
FN = con_matrix[1][0]

print("The confusion matrix for train data (k=%d) is:"%opt_k)
print("\t| predicted AB\t| predicted NO")
print("real AB\t| "+str(TP)+"\t\t| "+str(FN))
print("real NO\t| "+str(FP) +"\t\t| "+str(TN))

TP_rate = float(TP)/(TP + FN)
TN_rate = float(TN)/(FP+ TN)
precision = float(TP)/(TP + FP)
recall = float(TP)/(TP + FN)
F1_score = (2 * precision * recall) / (precision + recall)

print("\nTrue positive rate = %.2f" %TP_rate)
print("True negotive rate = %.2f" %TN_rate)
print("Precision = %.2f" %precision)
print("F1 score = %.2f" %F1_score)


# ### 3.2 Explore the effect of size of the training dataset

# In[8]:


def knn_N():
    test_err = []
    train_err = []
    nei = np.arange(1,200,5)
    
    N = np.arange(10,220,10)
    for number in N:
        max_acc = 0
        train_score = 0
        n_NO = int(number / 3)
        n_AB = number - int(number / 3)
        neis = [i for i in nei if i <= number]
        for k in neis:
            trainsub = pd.concat([train_AB.iloc[:n_AB],train_NO.iloc[:n_NO]])
            knn_N = neighbors.KNeighborsClassifier(k, algorithm="auto", metric="euclidean")
            knn_N.fit(trainsub[attributes], trainsub["CLASS"])
            score = knn_N.score(test[attributes], test["CLASS"])
            if score > max_acc :
                max_acc = score
                train_score = 1 - knn_N.score(train[attributes], train["CLASS"])
        test_err.append(1 - max_acc)
        train_err.append(train_score)
    return N,test_err, train_err

N, test_err, train_err = knn_N()
low_TER = train_err[test_err.index(min(test_err))]
low["2nd best training dataset KNN"] = low_TER

fig = plt.figure(figsize=(8,6))
plt.plot(N, test_err, label='Test')
plt.xlim(10,210)
plt.xlabel('Size of training dataset')
plt.ylabel('Error rate')
plt.title("Learning Curve (test error rate) against the size of training dataset")


# ## 4. Distance Metrics
# ### 4.1 Mahanttan distance

# In[9]:


def knn_manhattan():
    nei = np.arange(1,200,5)
    max_acc = 0
    train_err = 0
    opt_k = 0
    for k in nei:
        knn_manhattan = neighbors.KNeighborsClassifier(k, algorithm = "auto",metric="manhattan")
        knn_manhattan.fit(train[attributes], train["CLASS"])
        score = knn_manhattan.score(test[attributes], test["CLASS"])
        if score > max_acc :
            max_acc = score
            opt_k = k
            train_err = 1 - knn_manhattan.score(train[attributes], train["CLASS"])
    test_err = 1 - max_acc
    return opt_k,test_err,train_err

opt_k_man, test_err_man, train_err_man = knn_manhattan()
low["3rd Manhattan distance KNN"] = train_err_man
print("The optimal k for KNN Classifier with Manhattan distance is %d" %opt_k_man)
print("The corresponding test error rate (k=%d) is %.2f" %(opt_k_man,test_err_man))


# ### 4.2 Best Log_10(p) in Minkowski distance

# In[10]:


def knn_log():
    power_p = np.arange(0.1,1.1,0.1)
    p_list = list(map(lambda x: pow(10,x),power_p))

    max_acc = 0
    train_err = 0
    opt_p = p_list[0]
    for p_metric in p_list:
        knn_log = neighbors.KNeighborsClassifier(opt_k_man, algorithm="auto",p=p_metric, metric="minkowski")
        knn_log.fit(train[attributes], train["CLASS"])
        score = knn_log.score(test[attributes], test["CLASS"])
        if score > max_acc :
            max_acc = score
            opt_p = p_metric
            train_err = 1 - knn_log.score(train[attributes], train["CLASS"])
    test_err = 1 - max_acc
    return power_p[p_list.index(opt_p)], test_err, train_err

opt_p_log, test_err_log, train_err_log = knn_log()
low["4th Minkowski distance log_10(p) KNN"] = train_err_log

print("The optimal log_10(p) for KNN Classifier with Minkowski distance is %.1f" %opt_p_log)
print("The corresponding test error rate (log_10(p)=%.1f) is %.2f" %(opt_p_log,test_err_log))


# ### 4.3 Chebyshev Distance

# In[11]:


def knn_chebyshev():
    nei = np.arange(1,200,5)
    max_acc = 0
    train_err = 0
    opt_k = 0
    for k in nei:
        knn_chebyshev = neighbors.KNeighborsClassifier(k, algorithm="auto",metric="chebyshev")
        knn_chebyshev.fit(train[attributes], train["CLASS"])
        score = knn_chebyshev.score(test[attributes], test["CLASS"])
        if score > max_acc :
            max_acc = score
            opt_k = k
            train_err = 1 - knn_chebyshev.score(train[attributes], train["CLASS"])
    test_err = 1 - max_acc
    return opt_k,test_err, train_err

opt_k_che, test_err_che, train_err_che = knn_chebyshev()
low["5th Chebyshev distance KNN"] = train_err_che
print("The optimal k for KNN Classifier with Chebyshev distance is %d" %opt_k_che)
print("The corresponding test error rate (k=%d) is %.2f" %(opt_k_che ,test_err_che))


# ### 4.4 Mahalanobis Distance

# In[12]:


def knn_mahalanobis():
    nei = np.arange(1,200,5)
    max_acc = 0
    train_err = 0
    opt_k = 0
    for k in nei:
        knn_mahalanobis = neighbors.KNeighborsClassifier(k, algorithm="brute",metric="mahalanobis",metric_params={'V': train[attributes].cov()})
        knn_mahalanobis.fit(train[attributes], train["CLASS"])
        score = knn_mahalanobis.score(test[attributes], test["CLASS"])
        if score > max_acc :
            max_acc = score
            opt_k = k
            train_err = 1 - knn_mahalanobis.score(train[attributes], train["CLASS"])
    test_err = 1 - max_acc
    return opt_k,test_err, train_err

opt_k_mas, test_err_mas,train_err_mas = knn_mahalanobis()
low["6th Mahalanobis distance KNN"] = train_err_mas
print("The optimal k for KNN Classifier with Mahalanobis distance is %d" %opt_k_mas)
print("The corresponding test error rate (k=%d) is %.2f" %(opt_k_mas,test_err_mas))


# ### Summarized Table
# | Distance method|Metrics|Best test error rate
# | :------: | :------: | :------: |
# |Mahanttan distance| k = 6| 0.11
# |Minkowski distance| log_10(p) = 0.6| 0.06
# |Chebyshev Distance| k = 16 | 0.08
# |Mahalanobis Distance| k = 6|0.16

# ## 5. Weighted model
# ### 5.1 Weighted Euclidean distance

# In[13]:


def knn_w_euc():
    nei = np.arange(1,200,5)
    max_acc = 0
    train_err = 0
    opt_k = 0
    
    for k in nei:
        knn_euc = neighbors.KNeighborsClassifier(k, weights="distance",algorithm="auto",p=2, metric="minkowski")
        knn_euc.fit(train[attributes], train["CLASS"])
        score = knn_euc.score(test[attributes], test["CLASS"])
        if score > max_acc :
            max_acc = score
            opt_k = k
            train_err = 1 - knn_euc.score(train[attributes], train["CLASS"])
    test_err = 1 - max_acc
    return opt_k,test_err,train_err

opt_k_w_euc, test_err_w_euc, train_err_w_euc = knn_w_euc()
low["7th Weighted Euclidean distance KNN"] = train_err_w_euc
print("The optimal k for KNN Classifier with weighted Euclidean distance is %d" %opt_k_w_euc)
print("The corresponding test error rate (k=%d) is %.2f" %(opt_k_w_euc,test_err_w_euc))


# ### 5.2 Weighted Mahanttan distance

# In[14]:


def knn_w_man():
    nei = np.arange(1,200,5)
    max_acc = 0
    train_err = 0
    opt_k = 0
    for k in nei:
        knn_manhattan = neighbors.KNeighborsClassifier(k, weights="distance", algorithm = "auto",metric="manhattan")
        knn_manhattan.fit(train[attributes], train["CLASS"])
        score = knn_manhattan.score(test[attributes], test["CLASS"])
        if score > max_acc :
            max_acc = score
            opt_k = k
            train_err = 1 - knn_manhattan.score(train[attributes], train["CLASS"])
    test_err = 1 - max_acc
    return opt_k,test_err,train_err

opt_k_w_man, test_err_w_man,train_err_w_man = knn_w_man()
low["8th Weighted Manhattan distance"] = train_err_w_man
print("The optimal k for KNN Classifier with weighted Manhattan distance is %d" %opt_k_w_man)
print("The corresponding test error rate (k=%d) is %.2f" %(opt_k_w_man,test_err_w_man))


# ### 5.3 Weighted Chebyshev distance

# In[15]:


def knn_w_che():
    nei = np.arange(1,200,5)
    max_acc = 0
    train_err = 0
    opt_k = 0
    for k in nei:
        knn_chebyshev = neighbors.KNeighborsClassifier(k, weights="distance",algorithm="auto",metric="chebyshev")
        knn_chebyshev.fit(train[attributes], train["CLASS"])
        score = knn_chebyshev.score(test[attributes], test["CLASS"])
        if score > max_acc :
            max_acc = score
            opt_k = k
            train_err = 1 - knn_chebyshev.score(train[attributes], train["CLASS"])
    test_err = 1 - max_acc
    return opt_k,test_err,train_err

opt_k_w_che, test_err_w_che,train_err_w_che = knn_w_che()
low["9th Weighted Chebyshev distance KNN"] = train_err_w_che
print("The optimal k for KNN Classifier with weighted Chebyshev distance is %d" %opt_k_w_che)
print("The corresponding test error rate (k=%d) is %.2f" %(opt_k_w_che ,test_err_w_che))


# ### Summarized Table
# | Distance method|Metrics|Best test error rate
# | :------: | :------: | :------: |
# |Euclidean Distance| k = 6|0.1
# |Mahanttan distance| k = 26| 0.10
# |Chebyshev Distance| k = 16 | 0.11

# ## 6. Lowest training error

# In[16]:


df = pd.DataFrame.from_dict(low,orient='index')
print(df)
print("\nThe lowest training error in this exercise is %.f" %low[min(low,key=low.get)])

