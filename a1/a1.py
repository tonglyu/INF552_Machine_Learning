import numpy;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;

data = pd.read_csv("./data/column_2C.dat",sep=" ",header=None, names=["pel_in","pel_tilt", "lum_angle", "sac_slope", "pel_r", "spo", "class"])

#ax = sns.stripplot(x="pel_tilt", y="class",data=data, jitter=True)
#sns.pairplot(data,hue="class",size=3, diag_kind='kde')

box = sns.boxplot(data=data,x="class",y="spo")

plt.show()