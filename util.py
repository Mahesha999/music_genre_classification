import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

def load_preprocess_xy(file_path, split_percentage, scale_x, encode_y, dummify_y):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, 1:59].values
    y = dataset.iloc[:, 59].values

    if encode_y:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    if scale_x:
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)

    if dummify_y:
        y = pd.get_dummies(y).values

    if split_percentage > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_percentage, random_state = 0)
        return X_train, X_test, y_train, y_test
    
    return X, y

def get_accuracy(cm):
    sum = 0
    for i in range(cm.shape[0]):
        sum = sum + cm[i][i]
    return 100*(sum/np.sum(cm))


def fit_predict_print(fit_predict_function, X_train, y_train, X_test, y_test):
    y_pred = fit_predict_function(X_train, y_train, X_test)
    if y_pred.dtype != y_test.dtype:
        y_pred = np.argmax(y_pred,axis=1)
        y_test = np.argmax(y_test,axis=1)
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

