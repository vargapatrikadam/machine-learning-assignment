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

iris_dataset = pd.read_csv('data/iris_dataset.txt',
                           sep=',',
                           names=['sepal_len',
                                  'sepal_wid',
                                  'petal_len',
                                  'petal_wid',
                                  'class'])
X = iris_dataset.drop('class',axis=1)

#Dimenzióredukció végrehajtása az adathalmazon
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
pcaMapping = pca.transform(X)

#A dimenzióredukció eredményének vizualizációja
import matplotlib.pyplot as plt
plt.scatter(pcaMapping[:,0],pcaMapping[:,1])
plt.title('Iris dataset visualization')
plt.show()
plt.clf()

#Keresett klaszterek számosságának beállítása
K = 3

#KMeans algoritmus használata az adathalmazon klaszterezésként
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=K).fit(X)

#KMeans eredmény megjelenítése scatter ploton
for i in range(0,K):
    plt.scatter(pcaMapping[kmeans.labels_ == i][:, 0],
                pcaMapping[kmeans.labels_ == i][:, 1])
plt.title('K-Means eredmény')
plt.show()
plt.clf()

#DBSCAN algoritmus használata az adathalmazon
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1,min_samples=10).fit(X)

#DBSCAN eredmény megjelenítése scatter ploton
for i in range(0,K):
    plt.scatter(pcaMapping[dbscan.labels_ == i][:, 0],
                pcaMapping[dbscan.labels_ == i][:, 1])
plt.title('DBSCAN eredmény')
plt.show()
plt.clf()

#Klaszterek metszetének kiszámolása
def clusterIntersection(cluster1, cluster2):
    result = 0
    for c1 in cluster1:
        for c2 in cluster2:
            if (c1 - c2).sum() == 0:
                result += 1
                continue
    return result

for kmeansIndex in set(kmeans.labels_):
    for dbscanIndex in set(dbscan.labels_):
        print(kmeansIndex,',',dbscanIndex,',',
        clusterIntersection(
            X[kmeans.labels_ == kmeansIndex].values,
            X[dbscan.labels_ == dbscanIndex].values))

