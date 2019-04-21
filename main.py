'''
Klaszterezés és osztályozási feladat az 'Iris Data Set'-en, az UCI Machine Learning Repository-ból.
https://archive.ics.uci.edu/ml/datasets/Iris
Attribútumok:
    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm
    5. class:
    -- Iris Setosa
    -- Iris Versicolour
    -- Iris Virginica
'''

#Adathalmaz beolvasása
import pandas as pd

iris_dataset = pd.read_csv('data/iris_dataset.txt',
                           sep=',',
                           names=['sepal_len',
                                  'sepal_wid',
                                  'petal_len',
                                  'petal_wid',
                                  'class'])
X = iris_dataset.drop('class',axis=1)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
pcaMapping = pca.transform(X)

import matplotlib.pyplot as plt
plt.scatter(pcaMapping[:,0],pcaMapping[:,1])
plt.title('Iris dataset visualization')
plt.show()
plt.clf()

