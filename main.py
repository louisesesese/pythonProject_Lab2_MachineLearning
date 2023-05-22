import pandas as pd
import numpy as np
import sklearn.decomposition
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('glass.csv')
var_names = list(df.columns)
labels = df.to_numpy('int')[:,-1]
data = df.to_numpy('float')[:,:-1]
data = preprocessing.minmax_scale(data)

fig1, axs = plt.subplots(2,4)

for i in range(data.shape[1]-1):
    axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)],c=labels,cmap='hsv')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i+1])
# plt.show()

#метод главных компонент 85%
pca = PCA(n_components = 4, svd_solver='auto')
pca_data = pca.fit(data).transform(data)
#значение объясненной дисперсии в процентах и собственные числа соответствующие компонентам
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
#диаграмма рассеяния после метода главных компонентpca_data=pca.inverse_transform(pca_data)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv')
#plt.show()


#модификации метода главных компонент

kpca = sklearn.decomposition.KernelPCA(n_components = 2)
kpca_data = kpca.fit(data).transform(data)
plt.scatter(kpca_data[:,0],kpca_data[:,1],c=labels,cmap='hsv')
# plt.show()

spca = sklearn.decomposition.SparsePCA(n_components = 2)
spca_data = spca.fit(data).transform(data)
plt.scatter(spca_data[:,0],spca_data[:,1],c=labels,cmap='hsv')
# plt.show()