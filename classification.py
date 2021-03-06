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

import matplotlib.pyplot as plt

import numpy as np

#Adatok szétválasztása tanító és tesztelő egységekre
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

#GridSearchCV és MLP Classifier beimportálása
from sklearn.neural_network import MLPClassifier

#Decision Tree classifier beimportálása
from sklearn import tree

#Naive Bayes classifier beimportálása
from sklearn.naive_bayes import ComplementNB

#Support Vector Machine classifier beimportálása
from sklearn import svm

#Osztályozóalgoritmusok statisztikájáért felelős algoritmus beimportálása
from sklearn.metrics import classification_report, confusion_matrix

#Idő mérésére osztály beimportálása
from datetime import datetime


#Decision Tree beállítása
decisionTree = GridSearchCV(
    tree.DecisionTreeClassifier(max_depth=3),
    {},
    cv=5)
start=datetime.now()
decisionTree.fit(X_train,Y_train)
print('DT tanítási idő: ', (datetime.now()-start))


mlpParameters = {
    'hidden_layer_sizes':[(90,),(80,),(50,),(2,4,3),(10,20,10),(4,)],
    'activation':('tanh','relu'), 
    'solver':('sgd','adam'),
    'learning_rate':('constant','adaptive')
}
mlpClassifier = GridSearchCV(MLPClassifier(max_iter=10),mlpParameters, cv=10, scoring='accuracy')

mlpClassifier = GridSearchCV(
    MLPClassifier(max_iter=10, activation='tanh', hidden_layer_sizes=(18,),learning_rate='constant',solver='adam'),
    {},
    cv=5)
start=datetime.now()
mlpClassifier.fit(X_train,Y_train)
print('MLP tanítási idő: ', (datetime.now()-start))

complimentNaive = GridSearchCV(
    ComplementNB(),
    {},
    cv=5)
start=datetime.now()
complimentNaive.fit(X_train, Y_train)
print('CNB tanítási idő: ', (datetime.now()-start))
'''
svmParameters ={
    'kernel':['linear','rbf','poly','sigmoid'],
    'decision_function_shape':['ovr','ovo'],
    'gamma':['auto','scale']
}
'''
supportVectorMachine = GridSearchCV(
    svm.SVC(kernel='linear',decision_function_shape='ovr', gamma='scale'),
    {},
    cv=5)
start=datetime.now()
supportVectorMachine.fit(X_train, Y_train)
print('SVM tanítási idő: ', (datetime.now()-start))


meanTrainScore = decisionTree.cv_results_['mean_test_score']
stdTrainScore = decisionTree.cv_results_['std_test_score']
print("DT cv score: %0.2f (+/- %0.2f)" % (meanTrainScore, stdTrainScore * 2))

meanTrainScore = mlpClassifier.cv_results_['mean_test_score']
stdTrainScore = mlpClassifier.cv_results_['std_test_score']
print("MLP cv score: %0.2f (+/- %0.2f)" % (meanTrainScore, stdTrainScore * 2))

meanTrainScore = complimentNaive.cv_results_['mean_test_score']
stdTrainScore = complimentNaive.cv_results_['std_test_score']
print("CNB cv score: %0.2f (+/- %0.2f)" % (meanTrainScore, stdTrainScore * 2))

meanTrainScore = supportVectorMachine.cv_results_['mean_test_score']
stdTrainScore = supportVectorMachine.cv_results_['std_test_score']
print("SVM cv score: %0.2f (+/- %0.2f)" % (meanTrainScore, stdTrainScore * 2))

dtPredict = decisionTree.predict(X_test)
mlpPredict = mlpClassifier.predict(X_test)
cnbPredict = complimentNaive.predict(X_test)
svcPredict = supportVectorMachine.predict(X_test)


print('DT eredménye a test set-en:\n',classification_report(Y_test, dtPredict, labels=['e','p'],
                                                            target_names=['edible','poisonous']))


print('CNB eredménye a test set-en:\n',classification_report(Y_test, cnbPredict,
                                                             labels=['e', 'p'],
                                                             target_names=['edible', 'poisonous']
                                                             ))


print('MLP eredménye a test set-en:\n',classification_report(Y_test, mlpPredict,
                                                             labels=['e', 'p'],
                                                             target_names=['edible', 'poisonous']
                                                             ))


print('SVC eredménye a test set-en:\n',classification_report(Y_test, svcPredict,
                                                             labels=['e', 'p'],
                                                             target_names=['edible', 'poisonous']
                                                             ))


def plot_confusion_matrix(y_true, y_pred, classes,
                          title=None,
                          cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred)
    #classes = list(unique_labels(y_true, y_pred))

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


plot_confusion_matrix(Y_test, dtPredict, classes=['edible','poisonous'],title='DT Confusion Matrix')
plt.savefig('data/dt_conf_matrix.png')
plt.show()
plot_confusion_matrix(Y_test, mlpPredict, classes=['edible','poisonous'],title='MLP Confusion Matrix')
plt.savefig('data/mlp_conf_matrix.png')
plt.show()
plot_confusion_matrix(Y_test, cnbPredict, classes=['edible','poisonous'],title='CNB Confusion Matrix')
plt.savefig('data/cnb_conf_matrix.png')
plt.show()
plot_confusion_matrix(Y_test, svcPredict, classes=['edible','poisonous'],title='SVC Confusion Matrix')
plt.savefig('data/svc_conf_matrix.png')
plt.show()

