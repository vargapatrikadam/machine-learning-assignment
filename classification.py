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

X = iris_dataset.drop('class',axis=1)
Y = iris_dataset['class']

#Standard scaler alkalmazása az adatokon
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
scaled = scaler.transform(X)

#Train és test splitting
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, Y_train, Y_test = train_test_split(scaled,Y)

#GridSearchCV with MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

#Setting up parameters for the GridSearch with the MLP Classifier
mlpParameters = {
    #'hidden_layer_sizes':[(2,4,2),(5,6,8,5),(10,20,10,6)],
    #'hidden_layer_sizes':[(2,4,2),(5,6,8)],
    'hidden_layer_sizes':[(90,),(80,),(2,4,3),(5,6,3),(4,)],
    'activation':('tanh','relu'),
    'solver':('sgd','adam'),
    'learning_rate':('constant','adaptive')
}

#Executing the grid search
clf = GridSearchCV(MLPClassifier(max_iter=10000),mlpParameters, cv=10, scoring='accuracy')
clf.fit(X_train,Y_train)

# Best parameter set
print('MLP Classifier best parameters found:\n', clf.best_params_)
print('MLP CLassifier best score:\n',clf.best_score_)

#Printing the reports on the best set tested with the test set
from sklearn.metrics import classification_report
print('MLP Classifier results on the test set:')
print(classification_report(Y_test, clf.predict(X_test)))

#Setting up the decision tree for the problem
from sklearn import tree

decisionTree = tree.DecisionTreeClassifier().fit(X_train,Y_train)

#Printing out the score of the decision tree
print('Decision tree score:\n',decisionTree.score(X_train,Y_train))

#Printing out the results of the decision tree tested with the test set
print('Decision tree results on the test set:')
print(classification_report(Y_test, decisionTree.predict(X_test)))
