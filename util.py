import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

_categories=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_preprocess_xy(file_path, split_percentage, scale_x, encode_y, dummify_y):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, 1:59].values
    y = dataset.iloc[:, 59].values

    if encode_y:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    if scale_x:
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X) #TODO should we scale test and train separately? Yes, but wont have much difference?

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
    print("Accuracy: ", get_accuracy(cm))
    make_confusion_matrix(cm, figsize=(10, 6.5), categories=_categories)


def fit_predict_print_unsupervised(fit_predict_function, X_train, X_test, y_test):
    y_pred = fit_predict_function(X_train, X_test)
    if y_test is not None:
        cm = confusion_matrix(y_test, y_pred)
        print("Accuracy: ", get_accuracy(cm))
        make_confusion_matrix(cm, figsize=(10, 6.5), categories=_categories)
    

def search_fit_predict_print(fit_predict_function, X_train, y_train, X_test, y_test):
    ensemble, y_pred = fit_predict_function(X_train, y_train, X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy: ", get_accuracy(cm))
    print("Hyperparamters: ", ensemble.best_params_)
    make_confusion_matrix(cm, figsize=(10, 6.5), categories=_categories)


# Ref:
# https://github.com/DTrimarchi10/confusion_matrix
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

    plt.show()