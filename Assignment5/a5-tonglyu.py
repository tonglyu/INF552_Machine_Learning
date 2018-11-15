
# coding: utf-8

# # INF552 Assignment5 Multi-class and Multi-Label Classiﬁcation
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

from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import hamming_loss
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from scipy.spatial import distance

import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## 2. Multi-class and Multi-Label Classiﬁcation Using Support Vector Machines
# ### 2.1 Evaluation metrics (bi)
# * Accuracy score/ Exact match metric:      
#     This function calculates subset accuracy meaning the predicted set of labels should exactly match with the true set of labels.
# * Hamming Loss:        
#     The fraction of the wrong labels to the total number of labels, which for binary labels becomes:      
#     HammingLoss = $\frac{1}{N}\sum_{i=1}^{N}{\frac{XOR(Y_(i,j),P_(i,j))}{L}}$ 

# ### 2.2 SVM (1(b)ii)

# In[2]:


#1(b)ii SVM
input_data = pd.read_csv("./data/Frogs_MFCCs.csv", sep=",",header=0)
data_X = input_data.drop(["Family","Genus","Species","RecordID"],axis=1)
data_Y = input_data.loc[:,["Family","Genus","Species"]]
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.3)

penalties = np.linspace(0.0001,1,10)
gammas = np.linspace(0.00001,1,10)


# In[3]:


def svm_model(label):
    svm_score = {}
    for lamda in penalties:
        for gamma in gammas:
            svm_gua_model = svm.SVC(kernel="rbf",decision_function_shape='ovr',C=lamda,gamma=gamma)
            svm_gua_score = cross_val_score(svm_gua_model, X_train, y_train[label], cv=10, scoring=make_scorer(hamming_loss))
            svm_score[(lamda,gamma)] = np.mean(svm_gua_score)
    opt_param = min(svm_score,key=svm_score.get)
    return opt_param

def multi_class(label):
    opt_lamda, opt_gamma = svm_model(label)
    svm_gua_model = svm.SVC(kernel="rbf",decision_function_shape='ovr',C=opt_lamda,gamma=opt_gamma)
    svm_gua_model.fit(X_train,y_train[label])
    new_label = label+"_label" 
    df_test[new_label] = svm_gua_model.predict(X_test)
    svm_gua_hamming = hamming_loss(y_test[label], df_test[new_label])
    print("For label %s, the optimal penalty = %.5f, the optimal gamma = %.5f.The corresponding hamming loss = %.5f" %(label, opt_lamda, opt_gamma, svm_gua_hamming))

df_test = pd.DataFrame(y_test)
labels = ["Family","Genus","Species"]

def svm_multi_class():
    for label in labels:
        multi_class(label)
    df_true = df_test[labels].values
    df_pred=df_test.drop(["Family","Genus","Species"],axis=1).values
    loss = np.sum(np.not_equal(df_true,df_pred))/(3* float(df_true.size))
    print("The hamming loss of SVM classification for the whole dataset is %f\n" %loss)

svm_multi_class()


# ### 2.3 L1-penalized SVM (1(b)iii)

# In[6]:


#1(b)ii L1-penalized SVM
def l1_svm_model(label):
    svm_score = {}
    for lamda in penalties:
        svm_gua_model = svm.LinearSVC(penalty='l1', dual=False, tol=0.0001, C=lamda, multi_class='ovr')
        svm_gua_score = cross_val_score(svm_gua_model, X_train, y_train[label], cv=10, scoring=make_scorer(hamming_loss))
        svm_score[lamda] = np.mean(svm_gua_score)
    opt_param = min(svm_score,key=svm_score.get)
    return opt_param

def l1_multi_class(label):
    opt_lamda = l1_svm_model(label)
    svm_gua_model = svm.LinearSVC(penalty='l1', dual=False, tol=0.0001, C=opt_lamda, multi_class='ovr')
    svm_gua_model.fit(X_train,y_train[label])
    new_label = label+"_label" 
    df_test[new_label] = svm_gua_model.predict(X_test)
    svm_gua_hamming = hamming_loss(y_test[label], df_test[new_label])
    print("For label %s, the optimal penalty = %.5f.The corresponding hamming loss = %.5f" %(label, opt_lamda, svm_gua_hamming))

# Since the dataset has been normalized, we do not need to standardize again.
# X_scale = preprocessing.scale(X_train)
def l1_svm_multi_class():
    for label in labels:
        l1_multi_class(label)
    df_true = df_test[labels].values
    df_pred=df_test.drop(["Family","Genus","Species"],axis=1).values
    loss = np.sum(np.not_equal(df_true,df_pred))/(3*float(df_true.size))
    print("The hamming loss of L1-penalized SVM classification for the whole dataset is %f\n" %loss)

l1_svm_multi_class()


# ### 2.4 Using SMOTE to handle imbalance (1(b)iv)

# In[11]:


#1(b)iv Remedy class imbalance
sm = SMOTE()

def oversampling(label):
    X_res, Y_res = sm.fit_sample(X_train,y_train[label])
    X_res = pd.DataFrame(X_res)
    Y_res = pd.DataFrame(Y_res)
    return X_res, Y_res
    
def l1_smote_model(label,X_res, Y_res):
    svm_score = {}
    for lamda in penalties:
        svm_gua_model = svm.LinearSVC(penalty='l1', dual=False, tol=0.0001, C=lamda, multi_class='ovr')
        svm_gua_score = cross_val_score(svm_gua_model, X_res, Y_res, cv=10, scoring=make_scorer(hamming_loss))
        svm_score[lamda] = np.mean(svm_gua_score)
    opt_param = min(svm_score,key=svm_score.get)
    return opt_param

def l1_smote_class(label):
    X_res, Y_res = oversampling(label)
    opt_lamda = l1_smote_model(label, X_res, Y_res)
    svm_gua_model = svm.LinearSVC(penalty='l1', dual=False, tol=0.0001, C=opt_lamda, multi_class='ovr')
    svm_gua_model.fit(X_res, Y_res)
    new_label = label+"_label" 
    df_test[new_label] = svm_gua_model.predict(X_test)
    svm_gua_hamming = hamming_loss(y_test[label], df_test[new_label])
    print("For label %s, the optimal penalty = %.5f.The corresponding hamming loss = %.5f" %(label, opt_lamda, svm_gua_hamming))

def l1_smote_multi_class():
    for label in labels:
        l1_smote_class(label)
    df_true = df_test[labels].values
    df_pred=df_test.drop(["Family","Genus","Species"],axis=1).values
    loss = np.sum(np.not_equal(df_true,df_pred))/(3* float(df_true.size))
    print("The hamming loss L1-penalized SVM classification using SMOTE for the whole dataset is %f\n" %loss)

l1_smote_multi_class()


# ### Summary
# ##### Non-linear SVM model
# 
# |Label|Penalty|Gamma|Hamming Loss
# |:------: | :------: |:------:|:------:
# |Family|1|1|0.01575
# |Genus|1|1|0.01945
# |Species|1|1|0.01621
# |Hamming Loss for whole dataset = 0.005713
# 
# ##### L1-penalized SVM model
# 
# |Label|Penalty|Hamming Loss
# |:------: | :------: |:------:
# |Family|0.55560|0.06160
# |Genus|0.88890|0.06021
# |Species|1|0.04956
# |Hamming Loss for whole dataset = 0.019042
# 
# ##### L1-penalized using SMOTE SVM model
# 
# |Label|Penalty|Hamming Loss
# |:------: | :------: |:------:
# |Family|1|0.08569
# |Genus|1|0.08059
# |Species|0.88890|0.04215
# |Hamming Loss for whole dataset = 0.023210
# 
# From above three models, we can find that non-linear SVM models perform best among three, and L1-penalized models perform worst. As for the SMOTE, the oversampling did not improve the model. That means, the datasets is not linear-separable, which may fit better with a non-linear model.

# ## 3. K-Means Clustering on a Multi-Class and Multi-Label Data Set

# In[8]:


#2a K-Means with CH index
def get_opt_k():
    kmeans_k = np.arange(2,10)
    kmeans_score = {}
    for k in kmeans_k:
        kMeans_model = KMeans(n_clusters=k).fit(data_X)
        k_labels = kMeans_model.labels_
        kmeans_score[k] = metrics.calinski_harabaz_score(data_X,k_labels)
    opt_k = max(kmeans_score,key=kmeans_score.get)
    return opt_k


# In[9]:


#2b Determine labels
def family_label(cluster):
    return df[df['cluster']==cluster]["Family"].value_counts().index[0]

def genus_label(cluster):
    return df[df['cluster']==cluster]["Genus"].value_counts().index[0]

def species_label(cluster):
    return df[df['cluster']==cluster]["Species"].value_counts().index[0]

#2c Calculate Hamming Distance
def one_simulation(number,opt_k):
    print("\nThe result for %dth experiment with optimal k=%d:" %(number, opt_k))
    kMeans_model = KMeans(n_clusters=opt_k).fit(data_X)
    kMeans_labels = kMeans_model.labels_
    df['cluster'] = kMeans_labels

    labels = ["Family","Genus","Species"]
    df['Family_label'] = df['cluster'].apply(family_label)
    df['Genus_label'] = df['cluster'].apply(genus_label)
    df['Species_label'] = df['cluster'].apply(species_label)
    
    ham_dist_once = []
    for cluster_id in range(0, opt_k):
        cluster = df[df['cluster']==cluster_id]
        for label in labels:
            cluster_label = label+"_label"    
            dist=distance.hamming(cluster[label],cluster[cluster_label])
            ham_dist_once.append(dist)
            print("Hamming score of cluster %d for label %s is %f" %(cluster_id,label,dist))
            
    avg_dist = np.mean(ham_dist_once)
    ham_dist_all.append(avg_dist)
    print("The average hamming score for %dth experiment is %f\n\n" %(number, avg_dist))


# In[10]:


# Monte-Carlo Simulation
ham_dist_all = []
df = pd.DataFrame(data_Y)
for i in range(0,50):   
    opt_k = get_opt_k()
    one_simulation(i+1, opt_k)
print("The average haming distance for 50 simulations is %f" %np.mean(ham_dist_all))
print("The standard deviation for 50 simulations is %f" %np.std(ham_dist_all))

