
# coding: utf-8

# # INF552 Assignment3 Time Series Classiﬁcation
# ### Tong Lyu 1076139647
# ## 1 Introduction
# ### 1.1 Libraries

# In[1]:


import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statistics
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.ticker import MultipleLocator
from itertools import cycle
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## 2. Feature Extraction
# ### 2.1 Extract the time-domain features (ci & cii)
#     For common use, min, max, mean, median, std, 1st quantile, 3rd quantile, range, etc. are general.

# In[2]:


# cii
columns = ["time", "avg_rss12", "var_rss12","avg_rss13","var_rss13","avg_rss23","var_rss23"]
data = {}
feature = {}
instance_num = 1
for i in range(1,8):
    data[instance_num] = pd.read_csv("./AReM/bending1/dataset" + str(i) + ".csv", sep=",",header=4, names=columns)
    tmp = []
    data_min = data[instance_num].min()
    data_max = data[instance_num].max()
    data_mean = data[instance_num].mean()
    data_median = data[instance_num].median()
    data_std = data[instance_num].std()
    data_1st = data[instance_num].quantile(0.25)
    data_3rd = data[instance_num].quantile(0.75)
    for j in range(1,7):
        tmp.append(data_min[j])
        tmp.append(data_max[j])
        tmp.append(data_mean[j])
        tmp.append(data_median[j])
        tmp.append(data_std[j])
        tmp.append(data_1st[j])
        tmp.append(data_3rd[j])
    feature[instance_num] = tmp
    instance_num += 1
    
for i in range(1,7):
    if i == 4:
        data[instance_num] = pd.read_csv("./AReM/bending2/dataset" + str(i) + ".csv", sep=" ",header=4, names=columns)
    else:
        data[instance_num] = pd.read_csv("./AReM/bending2/dataset" + str(i) + ".csv", sep=",",header=4, names=columns)
    tmp = []
    data_min = data[instance_num].min()
    data_max = data[instance_num].max()
    data_mean = data[instance_num].mean()
    data_median = data[instance_num].median()
    data_std = data[instance_num].std()
    data_1st = data[instance_num].quantile(0.25)
    data_3rd = data[instance_num].quantile(0.75)
    for j in range(1,7):
        tmp.append(data_min[j])
        tmp.append(data_max[j])
        tmp.append(data_mean[j])
        tmp.append(data_median[j])
        tmp.append(data_std[j])
        tmp.append(data_1st[j])
        tmp.append(data_3rd[j])
    feature[instance_num] = tmp
    instance_num += 1

for folder in ["cycling","lying","sitting","standing","walking"]:
    for i in range(1,16):
        data[instance_num] = pd.read_csv("./AReM/"+ folder +"/dataset" + str(i) + ".csv", sep=",",header=4, names=columns)
        tmp = []
        data_min = data[instance_num].min()
        data_max = data[instance_num].max()
        data_mean = data[instance_num].mean()
        data_median = data[instance_num].median()
        data_std = data[instance_num].std()
        data_1st = data[instance_num].quantile(0.25)
        data_3rd = data[instance_num].quantile(0.75)
        for j in range(1,7):
            tmp.append(data_min[j])
            tmp.append(data_max[j])
            tmp.append(data_mean[j])
            tmp.append(data_median[j])
            tmp.append(data_std[j])
            tmp.append(data_1st[j])
            tmp.append(data_3rd[j])
        feature[instance_num] = tmp
        instance_num += 1
feature_columns = ['min1', 'max1', 'mean1', 'median1','std1','1st_quart1','3rd_quart1',
                  'min2', 'max2', 'mean2', 'median2','std2','1st_quart2','3rd_quart2',
                  'min3', 'max3', 'mean3', 'median3','std3','1st_quart3','3rd_quart3',
                  'min4', 'max4', 'mean4', 'median4','std4','1st_quart4','3rd_quart4',
                  'min5', 'max5', 'mean5', 'median5','std5','1st_quart5','3rd_quart5',
                  'min6', 'max6', 'mean6', 'median6','std6','1st_quart6','3rd_quart6']
feature = pd.DataFrame.from_dict(feature, orient='index', columns=feature_columns)
print(feature)


# ### 2.2 Standard deviation (ciii)

# In[3]:


# ciii Estimate the standard deviation
std_feature = feature.std()
stds = pd.DataFrame(std_feature,columns=["Estimate Std"])
#a 90% bootstrap conﬁdence interval for the standard deviation
for col in feature:
    col_bs = bs.bootstrap(np.array(feature[col]),stat_func = bs_stats.std,alpha=0.05)
    stds.loc[col,"Confidence Interval Std"] = col_bs.value
print(stds)


# ### 2.3 Select the three most important time-domain features (civ)
# In this case, we would like first select min, max and mean
# ## 3 Binary Classiﬁcation Using Logistic Regression
# ### 3.1 Classify bending from other activities (di)

# In[4]:


# di split training data and test data
train = pd.concat([feature.loc[3:7],feature.loc[10:13],feature.loc[17:28],feature.loc[32:43],
                   feature.loc[47:58],feature.loc[62:73],feature.loc[77:88]])
test =  pd.concat([feature.loc[1:2],feature.loc[8:9],feature.loc[14:16],feature.loc[29:31],
                   feature.loc[44:46],feature.loc[59:61],feature.loc[74:76]])

train_blending_col = ['min1', 'max1', 'mean1', 'min2', 'max2', 'mean2', 'min6', 'max6', 'mean6']
train_blending = pd.DataFrame(train[train_blending_col])
train_blending_class = [1 if i < 10 else 0 for i in range(1, 70)]
train_blending["CLASS"] = train_blending_class

plt.figure(figsize=(10,10))
sns.pairplot(train_blending,hue="CLASS",vars=train_blending_col,diag_kind='kde')
plt.show()


# ### 3.2 Break each time series into two (dii)
# 
# Difference:     
# When the time series features are splitted, some features such as min2 and min6 has a more evenly distribution (rather than a const in one instance), which can better represent the relationships with other features. 

# In[5]:


# dii
def feature_2(data,train_list):
    instance_num = 1
    feature = {}
    for i in train_list:    
        tmp = []
        tmp = get_description(data[i].loc[0:240], tmp)
        tmp = get_description(data[i].loc[240:480], tmp)
        feature[instance_num] = tmp
        instance_num += 1

    feature_columns = ['min11', 'max11', 'mean11', 'median11','std11','1st_quart11','3rd_quart11',
                       'min12', 'max12', 'mean12', 'median12','std12','1st_quart12','3rd_quart12',
                       'min21', 'max21', 'mean21', 'median21','std21','1st_quart21','3rd_quart21',
                       'min22', 'max22', 'mean22', 'median22','std22','1st_quart22','3rd_quart22',
                       'min31', 'max31', 'mean31', 'median31','std31','1st_quart31','3rd_quart31',
                       'min32', 'max32', 'mean32', 'median32','std32','1st_quart32','3rd_quart32',
                       'min41', 'max41', 'mean41', 'median41','std41','1st_quart41','3rd_quart41',
                       'min42', 'max42', 'mean42', 'median42','std42','1st_quart42','3rd_quart42',
                       'min51', 'max51', 'mean51', 'median51','std51','1st_quart51','3rd_quart51',
                       'min52', 'max52', 'mean52', 'median52','std52','1st_quart52','3rd_quart52',
                       'min61', 'max61', 'mean61', 'median61','std61','1st_quart61','3rd_quart61',
                      'min62', 'max62', 'mean62', 'median62','std62','1st_quart62','3rd_quart62']

    feature = pd.DataFrame.from_dict(feature, orient='index', columns=feature_columns)
    train_blending_col = ['min11', 'max11', 'mean11', 'min12', 'max12', 'mean12',
                          'min21', 'max21', 'mean21', 'min22', 'max22', 'mean22',
                          'min61', 'max61', 'mean61', 'min62', 'max62', 'mean62']
    train_blending = pd.DataFrame(feature[train_blending_col])
    train_blending_class = [1 if i < 10 else 0 for i in range(1, 70)]
    train_blending["CLASS"] = train_blending_class
    plt.figure(figsize=(12,12))
    sns.pairplot(train_blending,hue="CLASS",vars=train_blending_col, diag_kind='kde')
    plt.show()
    return train_blending

def get_description(data, tmp):
    data_min = data.min()
    data_max = data.max()
    data_mean = data.mean()
    data_median = data.median()
    data_std = data.std()
    data_1st = data.quantile(0.25)
    data_3rd = data.quantile(0.75)
    for j in range(1,7):
        tmp.append(data_min[j])
        tmp.append(data_max[j])
        tmp.append(data_mean[j])
        tmp.append(data_median[j])
        tmp.append(data_std[j])
        tmp.append(data_1st[j])
        tmp.append(data_3rd[j])
    return tmp

test_list = [1,2,8,9,14,15,16,29,30,31,44,45,46,59,60,61,74,75,76]
train_list = [x for x in range(1,89) if x not in test_list]
train_blending_2 = feature_2(data,train_list)


# ### 3.3 Logistic regression binary classiﬁcation (diii)
# * Right way to perform cross-validation
#     Apply cross-validation to find the top k predictors having the largest correlation with the class labels.
#     Apply cross-validation to classifier such as logistic regression, using only these top k predictors.
# * Wrong way to perform cross-validation
#     Only apply cross-validation to classifier , using all predictors.

# In[6]:


# diii
def feature_n(data,train_list,l):
    instance_num = 1
    train_feature = {}
    interval = int(480 / l);
    for i in train_list: 
        values = data[i].values
        min_max_scaler = preprocessing.MinMaxScaler()
        values_scaled = min_max_scaler.fit_transform(values)
        newDF = pd.DataFrame(values_scaled)
        tmp = []
        start = 0;
        end = interval
        for part in range(1,l + 1):
            if (interval > 480):
                tmp = get_description(newDF.loc[start:480], tmp)
            else:
                tmp = get_description(newDF.loc[start:end], tmp)
                start = end
                end = start + interval
        train_feature[instance_num] = tmp
        instance_num += 1

    train_feature = pd.DataFrame.from_dict(train_feature, orient='index')
    return train_feature

def get_description(data, tmp):
    data_min = data.min()
    data_max = data.max()
    data_mean = data.mean()
    data_median = data.median()
    data_std = data.std()
    data_1st = data.quantile(0.25)
    data_3rd = data.quantile(0.75)
    for j in range(1,7):
        tmp.append(data_min[j])
        tmp.append(data_max[j])
        tmp.append(data_mean[j])
        tmp.append(data_median[j])
        tmp.append(data_std[j])
        tmp.append(data_1st[j])
        tmp.append(data_3rd[j])
    return tmp


# In[7]:


l_list = np.arange(1,21)
l_score = []
kf = StratifiedKFold(5)
for l in l_list:
    train_feature = feature_n(data,train_list,l)
    train_blending_class = [1 if i < 10 else 0 for i in range(1, 70)]
    train_feature["CLASS"] = train_blending_class
    scores = []
    for train_index, test_index in kf.split(train_feature.drop(["CLASS"],axis=1),train_feature["CLASS"]):
        X_train,X_test = train_feature.iloc[train_index,:-1],train_feature.iloc[test_index,:-1]
        Y_train,Y_test = train_feature.iloc[train_index,-1],train_feature.iloc[test_index,-1]
        model = LogisticRegression(C=1000)
        selector = RFECV(model, step = 1, cv = 5)
        selector = selector.fit(X_train, Y_train)
        selected_features = [x for x in range(0, len(selector.support_)) if selector.support_[x] == True]
        result = model.fit(X_train.iloc[:,selected_features],Y_train)
        score = result.score(X_test.iloc[:,selected_features],Y_test)
        scores.append(score)
    print("For l=%d, the score of model is %.5f" % (l, statistics.mean(scores)))
    l_score.append(statistics.mean(scores))
print("The best l is %d, the corresponding score is %.5f" %(l_score.index(max(l_score)) + 1, max(l_score)))


# ### 3.4 Test the best classiﬁer on the train set (div)

# In[8]:


#div
def get_formula(columns):
    formula = 'CLASS ~ '
    for col in range(0,6):
        if col == 5:
            formula = formula + str(columns[col])
        else:
            formula = formula + str(columns[col]) + "+"
    return formula

best_train_feature = feature_n(data,train_list,1)
train_blending_class = [1 if i < 10 else 0 for i in range(1, 70)]
best_train_feature["CLASS"] = train_blending_class

best_X_train = best_train_feature.iloc[:,:-1]
best_Y_train = best_train_feature.iloc[:,-1]
clf = LogisticRegression(C=1e9)
selector = RFECV(clf, step = 1, cv = 5)
selector = selector.fit(best_X_train, best_Y_train)
selected_features = [x for x in range(0, len(selector.support_)) if selector.support_[x] == True]
selected_train = best_X_train.iloc[:,selected_features]

col_div=["f"+str(i) for i in range(0,len(selected_features))]
selected_train.columns = col_div
selected_train["CLASS"] = train_blending_class

formula = get_formula(col_div)
model = smf.glm(formula = formula, data=selected_train, family=sm.families.Binomial())
result= model.fit(maxiter=5)

#confusion matrix
best_Y_pred = result.predict(selected_train)
con_matrix = confusion_matrix(best_Y_train, (best_Y_pred>0.5).astype(int))
TP = con_matrix[1][1]
FP = con_matrix[0][1]
TN = con_matrix[0][0]
FN = con_matrix[1][0]
print("The confusion matrix of training dataset for l = 1:")
print("\t\t\t| predicted Blending\t| predicted NO Blending")
print("real Blending\t\t| "+str(TP)+"\t\t\t| "+str(FN))
print("real NO Blending\t| "+str(FP) +"\t\t\t| "+str(TN))

#ROC
fpr, tpr, thresholds = roc_curve(best_Y_train, best_Y_pred)
print("\nThe ROC curve for l = 1 is as follows, and the AUC = %.2f"% auc(fpr,tpr))
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#p-value
print(result.summary())


# ### 3.5 Test the classiﬁer on the test set (dv)

# In[9]:


#dv
test_feature = feature_n(data,test_list,1)
test_blending_class = [1 if i < 5 else 0 for i in range(1, test_feature.shape[0] + 1)]
test_feature["CLASS"] = test_blending_class

kf = StratifiedKFold(5)
test_scores = []
for train_index, test_index in kf.split(test_feature.drop(["CLASS"],axis=1),test_feature["CLASS"]):
    X_train,X_test = test_feature.iloc[train_index,:-1],test_feature.iloc[test_index,:-1]
    Y_train,Y_test = test_feature.iloc[train_index,-1],test_feature.iloc[test_index,-1]
    test_model = LogisticRegression(C=1e9)
    test_selector = RFECV(test_model, step = 1, cv = 5)
    test_selector = test_selector.fit(X_train, Y_train)
    test_selected_features = [x for x in range(0, len(test_selector.support_)) if test_selector.support_[x] == True]
    test_result = test_model.fit(X_train.iloc[:,test_selected_features],Y_train)
    test_score = test_result.score(X_test.iloc[:,test_selected_features],Y_test)
    test_scores.append(test_score)
print(statistics.mean(test_scores))


# For l=4, the accuracy of train set is 0.88571, while the accuracy for test set is 0.65
# ### 3.6 Well-separated(dvi)
# The classes are well-seperated, and some of the p-value of the parameters are higher than expected, which means the model is very instable.
# ### 3.7 Case-control sampling (dvii)
# Yes, one class binary classes has 9 intances, while the other has 60, which is imbalanced.

# In[10]:


#d vii
sm = SMOTE()
X_res, Y_res = sm.fit_sample(best_X_train,best_Y_train)
X_res = pd.DataFrame(X_res)
Y_res = pd.DataFrame(Y_res)
res_model = LogisticRegression(C=1e9)
res_selector = RFECV(res_model, step = 1, cv = 5)
res_selector = res_selector.fit(X_res, Y_res)
res_selected_features = [x for x in range(0, len(res_selector.support_)) if res_selector.support_[x] == True]
res_result = res_model.fit(X_res.iloc[:,res_selected_features],Y_res)
res_pred = res_model.predict(X_res.iloc[:,res_selected_features])

#confusion matrix
con_matrix = confusion_matrix(Y_res, (res_pred>0.5).astype(int))
TP = con_matrix[1][1]
FP = con_matrix[0][1]
TN = con_matrix[0][0]
FN = con_matrix[1][0]
print("The confusion matrix of Case-control sampling model for l = 1:")
print("\t\t\t| predicted Blending\t| predicted NO Blending")
print("real Blending\t\t| "+str(TP)+"\t\t\t| "+str(FN))
print("real NO Blending\t| "+str(FP) +"\t\t\t| "+str(TN))

#ROC
fpr, tpr, thresholds = roc_curve(Y_res, res_pred)
print("\nThe ROC curve for l = 1 is as follows, and the AUC = %.2f"% auc(fpr,tpr))
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# ## 4.Binary Classiﬁcation Using L1 -penalized logistic regression
# ### 4.1 L1 -penalized logistic regression (ei)

# In[11]:


# ei
l1_l_score = []
for l in l_list:
    train_feature = feature_n(data,train_list,l)
    train_blending_class = [1 if i < 10 else 0 for i in range(1, 70)]
    train_feature["CLASS"] = train_blending_class
    scores = []
    for train_index, test_index in kf.split(train_feature.drop(["CLASS"],axis=1),train_feature["CLASS"]):
        X_train,X_test = train_feature.iloc[train_index,:-1],train_feature.iloc[test_index,:-1]
        Y_train,Y_test = train_feature.iloc[train_index,-1],train_feature.iloc[test_index,-1]
        model = LogisticRegressionCV(cv=5,penalty="l1",solver="liblinear")
        result = model.fit(X_train,Y_train)
        score = result.score(X_test,Y_test)
        scores.append(score)
    print("For l=%d, the score of model is %.5f" % (l, statistics.mean(scores)))
    l1_l_score.append(statistics.mean(scores))
print("The best l is %d, the corresponding score is %.5f" %(l1_l_score.index(max(l1_l_score)) + 1, max(l1_l_score)))


# ### 4.2 Comparasion of two logistic regression (eii)
# From the table below, the accuracy of L1-penalized logistic regression is little higher in general. As for implementation, L1-penalized logistic regression is easier to implemented.

# In[12]:


# eii
model_score = pd.concat([pd.Series(l_score),pd.Series(l1_l_score)],axis=1)
model_score.columns=["Origin LR","L1 LR"]
model_score.index = [str(i) for i in range(1,21)]
print(model_score)


# ## 5.Multi-class Classiﬁcation (The Realistic Case)
# ### 5.1 L1 penalized multinomial regression model (fi)

# In[13]:


#fi
multi_l_score = []
multi_class = []
for i in range(1,70):
    if i < 10:
        multi_class.append(1)
    elif i < 22:
        multi_class.append(2)
    elif i < 34:
        multi_class.append(3)
    elif i < 46:
        multi_class.append(4)
    elif i < 58:
        multi_class.append(5)
    else:
        multi_class.append(6)
for l in l_list:
    train_feature = feature_n(data,train_list,l)
    train_feature["CLASS"] = multi_class
    scores = []
    for train_index, test_index in kf.split(train_feature.drop(["CLASS"],axis=1),train_feature["CLASS"]):
        X_train,X_test = train_feature.iloc[train_index,:-1],train_feature.iloc[test_index,:-1]
        Y_train,Y_test = train_feature.iloc[train_index,-1],train_feature.iloc[test_index,-1]
        model = LogisticRegressionCV(cv=5,penalty="l1",solver="saga",multi_class="multinomial")
        result = model.fit(X_train,Y_train)
        score = result.score(X_test,Y_test)
        scores.append(score)
    print("For l=%d, the score of model is %.5f" % (l, statistics.mean(scores)))
    multi_l_score.append(statistics.mean(scores))
print("The best l is %d, the corresponding score is %.5f" %(multi_l_score.index(max(multi_l_score)) + 1, max(multi_l_score)))


# In[14]:


multi_train_feature = feature_n(data,train_list,6)
multi_class = []
for i in range(1,70):
    if i < 10:
        multi_class.append(1)
    elif i < 22:
        multi_class.append(2)
    elif i < 34:
        multi_class.append(3)
    elif i < 46:
        multi_class.append(4)
    elif i < 58:
        multi_class.append(5)
    else:
        multi_class.append(6)

# Binarize the output

X_train, X_test, Y_train, Y_test = train_test_split(multi_train_feature, multi_class, test_size=.2,random_state=0)

multi_model = LogisticRegressionCV(cv=5,penalty="l1",solver="saga",multi_class="multinomial")
multi_result = multi_model.fit(X_train,Y_train)
Y_pred = multi_model.predict(X_test)

#confusion matrix
con_matrix = confusion_matrix(Y_test, Y_pred)
print("The confusion matrix of for l = 6:")
print(con_matrix)

#ROC
Y_pred = label_binarize(Y_pred, classes=[1,2,3,4,5,6])
Y_test = label_binarize(Y_test, classes=[1,2,3,4,5,6])
n_classes = 6
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print("\nThe ROC curve for l = 6 is as follows")
plt.figure()
lw = 2
colors = cycle(['red','blue','green','gold', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (auc = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc=2,bbox_to_anchor=(1.05,1.0))
plt.show()


# ### 5.2 Naive Bayes’ classiﬁer (fii)

# In[20]:


#fii
#Gaussian
gau_l_score = []
for l in l_list:
    train_feature = feature_n(data,train_list,l)
    train_blending_class = [1 if i < 10 else 0 for i in range(1, 70)]
    train_feature["CLASS"] = multi_class
    scores = []
    for train_index, test_index in kf.split(train_feature.drop(["CLASS"],axis=1),train_feature["CLASS"]):
        X_train,X_test = train_feature.iloc[train_index,:-1],train_feature.iloc[test_index,:-1]
        Y_train,Y_test = train_feature.iloc[train_index,-1],train_feature.iloc[test_index,-1]
        model = GaussianNB()
        result = model.fit(X_train,Y_train)
        score = result.score(X_test,Y_test)
        scores.append(score)
    print("For l=%d, the score of model is %.5f" % (l, statistics.mean(scores)))
    gau_l_score.append(statistics.mean(scores))
print("The best l for Gaussian is %d, the corresponding score is %.5f" %(gau_l_score.index(max(gau_l_score)) + 1, max(gau_l_score)))


#Multinomial
mnom_l_score = []
for l in l_list:
    train_feature = feature_n(data,train_list,l)
    train_blending_class = [1 if i < 10 else 0 for i in range(1, 70)]
    train_feature["CLASS"] = multi_class
    scores = []
    for train_index, test_index in kf.split(train_feature.drop(["CLASS"],axis=1),train_feature["CLASS"]):
        X_train,X_test = train_feature.iloc[train_index,:-1],train_feature.iloc[test_index,:-1]
        Y_train,Y_test = train_feature.iloc[train_index,-1],train_feature.iloc[test_index,-1]
        model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        result = model.fit(X_train,Y_train)
        score = result.score(X_test,Y_test)
        scores.append(score)
    print("For l=%d, the score of model is %.5f" % (l, statistics.mean(scores)))
    mnom_l_score.append(statistics.mean(scores))
print("The best l for Multinomial is %d, the corresponding score is %.5f" %(mnom_l_score.index(max(mnom_l_score)) + 1, max(mnom_l_score)))


# ### 5.3 Comparasion(fiii)
# From the above, we can see Multinomial method is the best for Gaussian method classiﬁcation in this problem.
