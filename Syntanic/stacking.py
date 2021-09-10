import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class StackingEnsemble:
    """
    Stacking ensemble
    ---------
    layers: list of lists of classifiers
    
    It consists of the lists of classifiers used in
    layers for stacking. The last layer must consist of 1 classifier
    otherwise the exception will be thrown"""
    def __init__(self, layers):
        assert len(layers[-1]) == 1
        assert type(layers) is list
        assert all(map(lambda layer: type(layer) is list, layers))
        self.layers = []
        for layer in layers:
            self.layers.append(layer.copy())
        self.n_layers = len(self.layers)
    
    def fit(self, X, y):
        current_X = X.copy()
        current_y = y.copy()
        for i, layer in enumerate(self.layers):
            if i == self.n_layers - 1:
                layer[0].fit(current_X, current_y)
                break
            X_train, X_test, y_train, y_test = train_test_split(current_X, current_y,
                                                                train_size=len(current_X) // (self.n_layers - i))
            for clf in layer:
                clf.fit(X_train, y_train)
            current_X = pd.DataFrame({
                k: clf.predict(X_test)
                for k, clf in enumerate(layer)
            })
            current_y = y_test
    
    def predict(self, X):
        current_X = X.copy()
        for layer in self.layers:
            current_X = pd.DataFrame({
                k: clf.predict(current_X)
                for k, clf in enumerate(layer)
            })
        return current_X        