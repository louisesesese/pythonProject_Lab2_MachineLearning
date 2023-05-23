import pandas as pd
import numpy as np
import sklearn.decomposition
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

dataset = pd.read_csv('glass.csv')

column_names = list(dataset.columns)
labels = dataset['Type']

data = dataset.iloc[:, :-1]

scaled_data = preprocessing.minmax_scale(data)

fig, axs = plt.subplots(2, 4)

for i, ax in enumerate(axs.flat):
    if i < scaled_data.shape[1]-1:
        ax.scatter(scaled_data[:,i], scaled_data[:,i+1], c=labels, cmap='hsv')
        ax.set_xlabel(column_names[i])
        ax.set_ylabel(column_names[i+1])

# PCA with 85% explained variance
pca = PCA(n_components = 4, svd_solver='auto')
pca_data = pca.fit(scaled_data).transform(scaled_data)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

plt.scatter(pca_data[:,0], pca_data[:,1], c=labels, cmap='hsv')

# KernelPCA
kpca = sklearn.decomposition.KernelPCA(n_components = 2)
kpca_data = kpca.fit(scaled_data).transform(scaled_data)
plt.scatter(kpca_data[:,0], kpca_data[:,1], c=labels, cmap='hsv')

# SparsePCA
spca = sklearn.decomposition.SparsePCA(n_components = 2)
spca_data = spca.fit(scaled_data).transform(scaled_data)
plt.scatter(spca_data[:,0], spca_data[:,1], c=labels, cmap='hsv')