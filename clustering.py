'''
Klaszterezési feladat az 'Iris Data Set'-en, az UCI Machine Learning Repository-ból.
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
from sklearn.preprocessing import StandardScaler

iris_dataset = pd.read_csv('data/iris_dataset.txt',
                           sep=',',
                           names=['sepal_len',
                                  'sepal_wid',
                                  'petal_len',
                                  'petal_wid',
                                  'class'])
X = iris_dataset.drop('class',axis=1)
X = StandardScaler().fit_transform(X)
Y = iris_dataset['class']

#Dimenzióredukció végrehajtása az adathalmazon
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, iris_dataset[['class']]], axis = 1)

#A dimenzióredukció eredményének vizualizációja
import matplotlib.pyplot as plt
plt.figure(figsize = (6,6))
plt.title('PCA az az Iris Dataset-en')
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
plt.legend(targets)
plt.grid()
plt.show()
plt.savefig('data/pca.png')
plt.clf()

#Keresett klaszterek számosságának beállítása
K = 3
#KMeans algoritmus használata az adathalmazon klaszterezésként
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=K).fit(X)

#KMeans eredmény megjelenítése scatter ploton
plt.figure(figsize = (6,6))
for i in range(0,K):
    plt.scatter(principalComponents[kmeans.labels_ == i][:, 0],
                principalComponents[kmeans.labels_ == i][:, 1])
plt.title('K-Means eredmény')
plt.grid()
plt.show()
plt.savefig('data/kmeans.png')
plt.clf()

#DBSCAN algoritmus használata az adathalmazon
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1,min_samples=1).fit(X)

#DBSCAN eredmény megjelenítése scatter ploton
plt.figure(figsize = (6,6))
for i in range(0,K):
    plt.scatter(principalComponents[dbscan.labels_ == i][:, 0],
                principalComponents[dbscan.labels_ == i][:, 1])
plt.title('DBSCAN eredmény')
plt.grid()
plt.show()
plt.savefig('data/dbscan.png')
plt.clf()
