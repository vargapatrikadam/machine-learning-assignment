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

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV

import matplotlib.pyplot as plt

import numpy as np

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
from sklearn.metrics import classification_report, precision_score, confusion_matrix

#Idő mérésére osztály beimportálása
from datetime import datetime

from sklearn.utils.multiclass import unique_labels

#Decision Tree beállítása
decisionTree = tree.DecisionTreeClassifier(max_depth=3)
start=datetime.now()
decisionTree.fit(X_train,Y_train)
print('Decision tree tanítási idő: ', (datetime.now()-start))
dt_train_sizes, dt_train_scores, dt_valid_scores = learning_curve(decisionTree, X, Y, cv=10)


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
mlp_train_sizes, mlp_train_scores, mlp_valid_scores = learning_curve(mlpClassifier, X, Y, cv=10)


gaussian = GaussianNB()
start=datetime.now()
gaussian.fit(X_train, Y_train)
print('Naive Bayes tanítási idő: ', (datetime.now()-start))
nb_train_sizes, nb_train_scores, nb_valid_scores = learning_curve(gaussian, X, Y, cv=10)


supportVectorMachine = svm.SVC(kernel='linear',decision_function_shape='ovo', gamma='auto')
start=datetime.now()
supportVectorMachine.fit(X_train, Y_train)
print('Support Vector Machine tanítási idő: ', (datetime.now()-start))
svm_train_sizes, svm_train_scores, svm_valid_scores = learning_curve(supportVectorMachine, X, Y, cv=10, shuffle=True)

print('Decision tree eredménye a test set-en:\n',classification_report(Y_test, decisionTree.predict(X_test)))
scores = cross_val_score(decisionTree, X, Y, cv=5)
print("Decision Tree pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print('Naive Bayes eredménye a test set-en:\n',classification_report(Y_test, gaussian.predict(X_test)))
scores = cross_val_score(gaussian, X, Y, cv=5)
print("Naive Bayes pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print('MLP Classifier eredménye a test set-en:\n',classification_report(Y_test, mlpClassifier.predict(X_test)))
scores = cross_val_score(mlpClassifier, X, Y, cv=5)
print("MLP pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print('Support Vector Machine eredménye a test set-en:\n',classification_report(Y_test, supportVectorMachine.predict(X_test)))
scores = cross_val_score(supportVectorMachine, X, Y, cv=5)
print("Support Vector Machine pontosság: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('Osztályozók tesztelése')
print('MLP Classifier eredménye: %f' % (precision_score(Y, mlpClassifier.predict(X), average='weighted')))
print('Decision Tree eredménye: %f' % (precision_score(Y, decisionTree.predict(X), average='weighted')))
print('Naive Bayes eredménye: %f' % (precision_score(Y, gaussian.predict(X), average='weighted')))
print('Support Vector Machine eredménye: %f' % (precision_score(Y, supportVectorMachine.predict(X),average='weighted')))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = list(unique_labels(y_true, y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(Y_test, decisionTree.predict(X_test), classes=['edible','poisonous'],title='DT Confusion Matrix')
plt.savefig('data/dt_conf_matrix.png')
plt.show()
plot_confusion_matrix(Y_test, mlpClassifier.predict(X_test), classes=['edible','poisonous'],title='MLP Confusion Matrix')
plt.savefig('data/mlp_conf_matrix.png')
plt.show()
plot_confusion_matrix(Y_test, gaussian.predict(X_test), classes=['edible','poisonous'],title='NB Confusion Matrix')
plt.savefig('data/nb_conf_matrix.png')
plt.show()
plot_confusion_matrix(Y_test, supportVectorMachine.predict(X_test), classes=['edible','poisonous'],title='SVC Confusion Matrix')
plt.savefig('data/svc_conf_matrix.png')
plt.show()
