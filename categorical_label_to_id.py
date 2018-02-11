import random
import pandas
import numpy as np
from sklearn import metrics, cross_validation

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

random.seed(42)
"""
data = pandas.read_csv('titanic_train.csv')
X = data[["Embarked"]]
y = data["Survived"]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)

embarked_classes = X_train["Embarked"].unique()
n_classes = len(embarked_classes) + 1
print('Embarked has next classes: ', embarked_classes)
"""

X_train = ["s", "a", "s", "d"]
cat_processor = learn.preprocessing.CategoricalProcessor()
X_train = np.array(list(cat_processor.fit_transform(X_train)))
t = X_train[0][0]

result = cat_processor.vocabularies_[0].reverse(t)
