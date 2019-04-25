'''
Osztályozási feladat az 'Iris Data Set'-en, az UCI Machine Learning Repository-ból.
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

#Adathalmazról az 'class' attrúbútum leválasztása Y változóba
X = iris_dataset.drop('class',axis=1)
Y = iris_dataset['class']

#Standard scaler alkalmazása az adatokon
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
scaled = scaler.transform(X)

#Adatok szétválasztása tanító és tesztelő egységekre
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(scaled,Y)

#GridSearchCV és MLP Classifier beimportálása
from sklearn.model_selection import GridSearchCV
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
decisionTreeParams = {}
decisionTree = GridSearchCV(tree.DecisionTreeClassifier(), decisionTreeParams, cv=10, scoring='accuracy')
print('Decision Tree tanítás alatt')
start=datetime.now()
decisionTree.fit(X_train,Y_train)
#decisionTree = tree.DecisionTreeClassifier().fit(X_train,Y_train)
print('Decision tree tanítási idő: ', (datetime.now()-start))

#GridSearchCV beállítása a MLP Classifier-re
mlpParameters = {
    #'hidden_layer_sizes':[(2,4,2),(5,6,8,5),(10,20,10,6)],
    #'hidden_layer_sizes':[(2,4,2),(5,6,8)],
    'hidden_layer_sizes':[(90,),(80,),(2,4,3),(10,20,10),(4,)],
    'activation':('tanh','relu'),
    'solver':('sgd','adam'),
    'learning_rate':('constant','adaptive')
}
clf = GridSearchCV(MLPClassifier(max_iter=10000),mlpParameters, cv=10, scoring='accuracy')
print('MLP tanítás alatt')
start=datetime.now()
clf.fit(X_train,Y_train)
print('MLP tanítási idő: ', (datetime.now()-start))

#GridSearchCV beállítása a Naive Bayes classifier-re
gaussanparams = {}
gaussian = GridSearchCV(GaussianNB(), gaussanparams, cv=10, scoring='accuracy')
print('Naive Bayes tanítás alatt')
start=datetime.now()
gaussian.fit(X_train, Y_train)
print('Naive Bayes tanítási idő: ', (datetime.now()-start))

#GridSearchCV beállítása Support Vector Machine classifier-re
supportVectorMachineParams = {'decision_function_shape':('ovo','ovr'),
                              'gamma':('auto','scale')}
supportVectorMachine = GridSearchCV(svm.SVC(),supportVectorMachineParams, cv=10,scoring='accuracy')
print('Support Vector Machine tanítás alatt')
start=datetime.now()
supportVectorMachine.fit(X_train, Y_train)
print('Support Vector Machine tanítási idő: ', (datetime.now()-start))


#Decision Tree legjobb score
print('Decision Tree legjobb score:\n',decisionTree.best_score_)
#A Decision Tree teszteredményei
print('Decision tree eredménye a test set-en:\n',classification_report(Y_test, decisionTree.predict(X_test)))

#Naive Bayes legjobb score
print('Naive Bayes legjobb score:\n',gaussian.best_score_)
#Naive Bayes teszteredményei
print('Naive Bayes eredménye a test set-en:\n',classification_report(Y_test, gaussian.predict(X_test)))

#Legjobb paraméterű MLP Classifier a megadottak közül a GridSearch szerint
print('MLP Classifier legjobb paraméter:\n', clf.best_params_)
print('MLP CLassifier legjobb score:\n',clf.best_score_)
#A legjobb eredménnyel rendelkező MLP Classifier-nek a teszteredményei
print('MLP Classifier eredménye a test set-en:\n',classification_report(Y_test, clf.predict(X_test)))

#Legjobb paraméterű Support Vector Machine a megadottak közül a GridSearch szerint
print('Support Vector Machine legjobb paraméter:\n', supportVectorMachine.best_params_)
print('Support Vector Machine legjobb score:\n',supportVectorMachine.best_score_)
#A legjobb eredménnyel rendelkező Support Vector Machine-nek a teszteredményei
print('Support Vector Machine eredménye a test set-en:\n',classification_report(Y_test, supportVectorMachine.predict(X_test)))


#Adatok rng-szer splitelése, ezáltal az osztályozók pontossága a teszt adatokkal a precision változókban egy mediánba
rng = 10000
print('Osztályozók tesztelése {} -szer'.format(rng))
mlpPrecision = 0
decisionTreePrecision = 0
gaussianPrecision = 0
supportVectorMachinePrecision = 0
start=datetime.now()
for i in range(rng):
    X_train, X_test, Y_train, Y_test = train_test_split(scaled, Y)
    mlpPrecision += precision_score(Y_test, clf.predict(X_test), average='weighted')
    decisionTreePrecision += precision_score(Y_test, decisionTree.predict(X_test), average='weighted')
    gaussianPrecision += precision_score(Y_test, gaussian.predict(X_test), average='weighted')
    supportVectorMachinePrecision += precision_score(Y_test, supportVectorMachine.predict(X_test),average='weighted')
mlpPrecision = mlpPrecision / rng
decisionTreePrecision = decisionTreePrecision / rng
gaussianPrecision = gaussianPrecision/ rng
supportVectorMachinePrecision = supportVectorMachinePrecision / rng
print('Osztályozók tesztelése ideje: {}'.format(datetime.now()-start))
print('MLP Classifier eredménye %d iteráció után: %f' % (rng, mlpPrecision))
print('Decision Tree eredménye %d iteráció után: %f' % (rng, decisionTreePrecision))
print('Naive Bayes eredménye %d iteráció után: %f' % (rng, gaussianPrecision))
print('Support Vector Machine eredménye %d iteráció után: %f' % (rng, supportVectorMachinePrecision))