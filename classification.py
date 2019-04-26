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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#Adatok szétválasztása tanító és tesztelő egységekre
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

#GridSearchCV és MLP Classifier beimportálása
from sklearn.neural_network import MLPClassifier

#Decision Tree classifier beimportálása
from sklearn import tree

#Naive Bayes classifier beimportálása
from sklearn.naive_bayes import GaussianNB

#Support Vector Machine classifier beimportálása
from sklearn import svm

#Osztályozóalgoritmusok statisztikájáért felelős algoritmus beimportálása
from sklearn.metrics import classification_report, precision_score

#Idő mérésére osztály beimportálása
from datetime import datetime

#Decision Tree beállítása
decisionTree = tree.DecisionTreeClassifier(max_depth=3)
start=datetime.now()
decisionTree.fit(X_train,Y_train)
print('Decision tree tanítási idő: ', (datetime.now()-start))

'''
mlpParameters = {
    #'hidden_layer_sizes':[(2,4,2),(5,6,8,5),(10,20,10,6)],
    #'hidden_layer_sizes':[(2,4,2),(5,6,8)],
    'hidden_layer_sizes':[(90,),(80,),(50,),(2,4,3),(10,20,10),(4,)],
    'activation':('tanh','relu'),
    'solver':('sgd','adam'),
    'learning_rate':('constant','adaptive')
}
mlpClassifier = GridSearchCV(MLPClassifier(max_iter=10),mlpParameters, cv=10, scoring='accuracy')
'''
mlpClassifier = MLPClassifier(max_iter=10, hidden_layer_sizes=(50,),learning_rate='constant',solver='adam')
start=datetime.now()
mlpClassifier.fit(X_train,Y_train)
print('MLP tanítási idő: ', (datetime.now()-start))

gaussian = GaussianNB()
start=datetime.now()
gaussian.fit(X_train, Y_train)
print('Naive Bayes tanítási idő: ', (datetime.now()-start))

supportVectorMachine = svm.SVC(kernel='linear',decision_function_shape='ovo', gamma='auto')
start=datetime.now()
supportVectorMachine.fit(X_train, Y_train)
print('Support Vector Machine tanítási idő: ', (datetime.now()-start))


print('Decision tree eredménye a test set-en:\n',classification_report(Y_test, decisionTree.predict(X_test)))
scores = cross_val_score(decisionTree, X_test, Y_test, cv=5)
print("Decision Tree pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print('Naive Bayes eredménye a test set-en:\n',classification_report(Y_test, gaussian.predict(X_test)))
scores = cross_val_score(gaussian, X_test, Y_test, cv=5)
print("Naive Bayes pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print('MLP Classifier eredménye a test set-en:\n',classification_report(Y_test, mlpClassifier.predict(X_test)))
scores = cross_val_score(mlpClassifier, X_test, Y_test, cv=5)
print("MLP pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print('Support Vector Machine eredménye a test set-en:\n',classification_report(Y_test, supportVectorMachine.predict(X_test)))
scores = cross_val_score(supportVectorMachine, X_test, Y_test, cv=5)
print("Support Vector Machine pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('Osztályozók tesztelése')
print('MLP Classifier eredménye: %f' % (precision_score(Y_test, mlpClassifier.predict(X_test), average='weighted')))
print('Decision Tree eredménye: %f' % (precision_score(Y_test, decisionTree.predict(X_test), average='weighted')))
print('Naive Bayes eredménye: %f' % (precision_score(Y_test, gaussian.predict(X_test), average='weighted')))
print('Support Vector Machine eredménye: %f' % (precision_score(Y_test, supportVectorMachine.predict(X_test),average='weighted')))