
from data_preperation import clean_data_strategy_1
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.stats.stats import pearsonr


def rmse(predictions, targets):
    """
    Root mean squared error
    
    Arguments:
        predictions {list} -- prediction values
        targets {list} -- real values
    
    Returns:
        [float] -- RMSE
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_strategy_1(X_train_de, X_val_de, y_train_de, y_val_de):
    """
    Training strategy 1 models

        
    Arguments:
        X_train_de {list} -- X training
        X_val_de {list} -- X validation
        y_train_de {list} -- y training
        y_val_de {list} -- y validation
    """
    print('#'*10)
    print('IN PROGRESS: Training Strategy_1')
    print('#'*10)
    models = [LinearRegression(), BayesianRidge(), MLPRegressor()]


    for model in models:
        print(str(model).split('(')[0])
        model.fit(X_train_de, y_train_de)
        predictions = model.predict(X_val_de)
        pearson = pearsonr(y_val_de, predictions)
        print(f'RMSE: {rmse(predictions,y_val_de)} Pearson {pearson[0]}')
        print('#'*10)


def train_baseline(X_train_de, X_val_de, y_train_de, y_val_de):
    """
    Training the baseline
    
    Arguments:
        X_train_de {list} -- X training
        X_val_de {list} -- X validation
        y_train_de {list} -- y training
        y_val_de {list} -- y validation
    """
    print('#'*10)
    print('IN PROGRESS: Training Baseline')
    print('#'*10)
    for k in ['linear','poly','rbf']:
        clf_t = SVR(kernel=k)
        clf_t.fit(X_train_de, y_train_de)
        print(k+' SVM')
        predictions = clf_t.predict(X_val_de)
        pearson = pearsonr(y_val_de, predictions)
        print(f'RMSE: {rmse(predictions,y_val_de)} Pearson {pearson[0]}')
        print()
        print('#'*10)

    print('RandomForestRegressor')
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 666)
    rf.fit(X_train_de, y_train_de)
    predictions = rf.predict(X_val_de)

    pearson = pearsonr(y_val_de, predictions)
    print('RMSE:', rmse(predictions,y_val_de))
    print(f"Pearson {pearson[0]}")
    print('#'*10)


def main_strategy_1():
    """
    the main starting point for strategy 1
    """
    
    X_train_de, X_val_de, y_train_de, y_val_de = clean_data_strategy_1()
    train_baseline(X_train_de, X_val_de, y_train_de, y_val_de)
    train_strategy_1(X_train_de, X_val_de, y_train_de, y_val_de)
