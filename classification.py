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

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

#GridSearchCV
from sklearn.model_selection import GridSearchCV

mlpParameters = {
    'hidden_layer_sizes':[(3,4,5),(10,20,10),(10,5,2),(5,10,20,25,30,15,6)],
    'activation':('tanh','relu','logistic'),
    'solver':('sgd','adam'),
    'learning_rate':('constant','adaptive')
}
mlp = MLPClassifier()
gridSearch = GridSearchCV(mlp,mlpParameters, cv=5)
gridSearch.fit(X_train,Y_train)
gridSearch.score(X_test, Y_test)