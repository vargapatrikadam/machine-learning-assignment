'''
Klaszterezési feladat a 'Mushroom' data set-en, az UCI Machine Learning Repository-ból.
https://archive.ics.uci.edu/ml/datasets/Mushroom
Attribútumok:
    1. cap-shape:                 bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
    2. cap-surface:               fibrous=f,grooves=g,scaly=y,smooth=s
    3. cap-color:                 brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
    4. bruises?:                  bruises=t,no=f
    5. odor:                      almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
    6. gill-attachment:           attached=a,descending=d,free=f,notched=n
    7. gill-spacing:              close=c,crowded=w,distant=d
    8. gill-size:                 broad=b,narrow=n
    9. gill-color:                black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d
    23. class:                    edible=e, poisonous=p
'''

#Adathalmaz beolvasása
import pandas as pd

mushroom_dataset = pd.read_csv('data/agaricus-lepiota.txt',
                           sep=',',
                           names=['class',
                                  'cap-shape',
                                  'cap-surface',
                                  'cap-color',
                                  'bruises',
                                  'odor',
                                  'gill-attachment',
                                  'gill-spacing',
                                  'gill-size',
                                  'gill-color',
                                  'stalk-shape',
                                  'stalk-root',
                                  'stalk-surface-above-ring',
                                  'stalk-surface-below-ring',
                                  'stalk-color-above-ring',
                                  'stalk-color-below-ring',
                                  'veil-type',
                                  'veil-color',
                                  'ring-number',
                                  'ring-type',
                                  'spore-print-color',
                                  'population',
                                  'habitat'])
#Hiányzó adatokat tartalmazó rekordok eldobása
mushroom_dataset = mushroom_dataset[mushroom_dataset['stalk-root'] != '?']
#Oszlopok típusainak átkonvertálása
for col in mushroom_dataset:
    mushroom_dataset[col] = mushroom_dataset[col].astype('category')
#Attribútumok leválasztása az osztálytól
X = mushroom_dataset.drop('class',axis=1)
cat_columns = X.select_dtypes(['category']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
Y = mushroom_dataset['class']

#Dimenzióredukció végrehajtása az adathalmazon
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, mushroom_dataset[['class']]], axis = 1)

#A dimenzióredukció eredményének vizualizációja
import matplotlib.pyplot as plt


plt.figure(figsize = (6,6))
plt.title('PCA a Mushroom data set-en')
targets = ['p','e']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
plt.legend(['edible','poisionous'])
plt.grid()
plt.savefig('data/pca.png')
plt.show()
plt.clf()

#Keresett klaszterek számosságának beállítása
K = 2
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
plt.savefig('data/kmeans.png')
plt.show()
plt.clf()

#DBSCAN algoritmus használata az adathalmazon
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=3.471,min_samples=44).fit(X)

#DBSCAN eredmény megjelenítése scatter ploton
plt.figure(figsize = (6,6))
for i in range(-1,K):
    plt.scatter(principalComponents[dbscan.labels_ == i][:, 0],
                principalComponents[dbscan.labels_ == i][:, 1])
plt.title('DBSCAN eredmény')
plt.grid()
plt.savefig('data/dbscan.png')
plt.show()
plt.clf()

#Agglomerációs algoritmus használata az adathalmazon klaszterezésként
from sklearn.cluster import AgglomerativeClustering

agglomerative = AgglomerativeClustering(n_clusters=K).fit(X)

#Agglomerációs klaszterezés eredmény megjelenítése scatter ploton
plt.figure(figsize = (6,6))
for i in range(0,K):
    plt.scatter(principalComponents[agglomerative.labels_ == i][:, 0],
                principalComponents[agglomerative.labels_ == i][:, 1])
plt.title('Agglomerációs klaszterezés eredmény')
plt.grid()
plt.savefig('data/meanshift.png')
plt.show()
plt.clf()