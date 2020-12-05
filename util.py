import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_scale_xy_with_25p_split():
    dataset = pd.read_csv("data/features_30_sec.csv")
    X = dataset.iloc[:, 1:59].values
    y = dataset.iloc[:, 59].values

    #importing the dataset
    X_train, X_test = train_test_split(X, test_size=0.25, random_state= 0)
    y_train, y_test = train_test_split(y, test_size=0.25, random_state= 0)

    #feature scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_scale_x_encode_y():
    dataset = pd.read_csv("data/features_30_sec.csv")
    X = dataset.iloc[:, 1:59].values
    y = dataset.iloc[:, 59].values

    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return X, y
    
def get_accuracy(cm):
    sum = 0
    for i in range(cm.shape[0]):
        sum = sum + cm[i][i]
    return 100*(sum/np.sum(cm))

def fit_predict_print(fit_predict_function, X_train, y_train, X_test, y_test):
    y_pred = fit_predict_function(X_train, y_train, X_test)
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    print("Accuracy: ", get_accuracy(cm))
    plt.matshow(cm)
    plt.show()

def fit_predict_print_unsupervised(fit_predict_function, X_train, X_test, y_test):
    y_pred = fit_predict_function(X_train, X_test)
    if y_test is not None:
        cm = confusion_matrix(y_test, y_pred)
        #print(cm)
        print("Accuracy: ", get_accuracy(cm))
        plt.matshow(cm)
        plt.show()
    

def search_fit_predict_print(fit_predict_function, X_train, y_train, X_test, y_test):
    ensemble, y_pred = fit_predict_function(X_train, y_train, X_test)
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    print("Accuracy: ", get_accuracy(cm))
    print("Hyperparamters: ", ensemble.best_params_)
    plt.matshow(cm)
    plt.show()
