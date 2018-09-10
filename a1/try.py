# coding: utf-8

# # INF552 Assignment1 KNN
# ### Tong Lyu 1076139647
#
# ## 1 Introduction
# ### 1.1 Libraries

# In[124]:


import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from sklearn import neighbors


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

# In[152]:


columns = ["PEI", "PET", "LLA", "SAS", "PER", "GOS", "CLASS"]
data = pd.read_csv ( "./data/column_2C.dat", sep=" ", header=None, names=columns )
plt.figure ( figsize=(6, 6) )
sns.pairplot ( data, hue="CLASS", size=3, diag_kind='kde' )
plt.show ()

# ### 2.2 Make boxplots for each of the independent variables

# In[42]:


attributes = ["PEI", "PET", "LLA", "SAS", "PER", "GOS"]
plt.figure ( figsize=(12, 9) )
plt.subplots_adjust ( hspace=0.3, wspace=0.5 )
arrange = 230
for attr in attributes:
    arrange += 1
    plt.subplot ( arrange )
    sns.boxplot ( data=data, x="CLASS", y=attr )


# ### 2.3 Select training dataset and test dataset
# Select the ﬁrst 70 rows of Class 0 and the ﬁrst 140 rows of Class 1 as the training set and the rest of the data as the test set.

# In[ ]:


def tranCat2Num(category):
    if (category == "AB"):
        return 1
    else:
        return 0


data["CLASS"] = data.apply ( lambda row: tranCat2Num ( row["CLASS"] ), axis=1 )
train_AB = data.loc[0:139]
test_AB = data.loc[140:209]
train_NO = data.loc[210:279]
test_NO = data.loc[280:309]
train = pd.concat ( [train_AB, train_NO] )
test = pd.concat ( [test_AB, test_NO] )


# ## 3. Classiﬁcation using KNN
# ### 3.1 Variance of K

# In[160]:


# calculate accuracy:
def score(prediction, reality):
    correct = 0
    total = prediction.shape[0]
    for i in range ( 0, total ):
        if (prediction[i] == reality[i]):
            correct += 1
    return float ( correct ) / total


knn_k = neighbors.KNeighborsClassifier ( 3 )
knn_k.fit ( train[attributes], train["CLASS"] )
pred_test = knn_k.predict ( test[attributes] )
print ( test["CLASS"] )
test_err = knn_k.score ( pred_test, test["CLASS"] )

