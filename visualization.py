from matplotlib import pyplot as plot
from matplotlib import style
import pandas as pd
import numpy 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
df = pd.read_csv('train.csv', index_col=0)
dataset = df.iloc[:,1:]
print(dataset.shape)
print(dataset.dtypes)
pd.set_option('display.max_columns', None)
print(dataset.describe())
print(dataset.skew())
print(dataset.groupby('Cover_Type').size())
size = 10
data=dataset.iloc[:,:size]
cols=data.columns 
data_corr = data.corr()
threshold = 0.5
corr_list = []
for i in range(0,size):
	for j in range(i+1,size): 
		if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
			corr_list.append([data_corr.iloc[i,j],i,j]) 
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
for v,i,j in s_corr_list:
	print (cols[i]+" "+cols[j]+" "+str(v))
for v,i,j in s_corr_list:
	sns.pairplot(dataset, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
	plt.show()
cols = dataset.columns
size = len(cols)-1
x = cols[size]
y = cols[0:size]
for i in range(0,size):
	sns.violinplot(data=dataset,x=x,y=y[i])
	plt.show()
rem = []
for c in dataset.columns:
	if dataset[c].std() == 0: #standard deviation is zero",
		rem.append(c)
dataset.drop(rem,axis=1,inplace=True)
print(rem)
r, c = dataset.shape
cols = dataset.columns
i_cols = []
for i in range(0,c-1):
	i_cols.append(i)
