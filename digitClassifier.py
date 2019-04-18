import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets,neighbors, metrics

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)

# set classifier
steps = [
        ('scaler', StandardScaler()),
        ('knn', neighbors.KNeighborsClassifier())
        ]

param_grid = {
        'knn__n_neighbors':np.arange(1,8)
        }

pipeline = Pipeline(steps)

gcv = GridSearchCV(pipeline, param_grid, cv=5)

gcv.fit(X_train,y_train)
print(gcv.best_params_) # Print for which value of n_neighbors we have highest accuracy
print(gcv.best_score_)  #and print the best score

y_pred = gcv.predict(X_test)

print(metrics.confusion_matrix(y_test, y_pred))