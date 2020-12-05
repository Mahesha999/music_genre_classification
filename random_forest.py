from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# def rf_fit_predict(X_train, y_train, X_test):
#   classifier = RandomForestClassifier(n_estimators= 10, criterion="entropy", random_state=0)
#   classifier.fit(X_train,y_train)
#   print(classifier.get_params())
#   return classifier.predict(X_test)
  

def rf_fit_predict(X_train, y_train, X_test):
  classifier = RandomForestClassifier(max_samples=0.95, n_estimators= 3000, bootstrap=True, min_samples_split=2, min_samples_leaf=1, criterion="entropy", random_state=0)
  # classifier = ExtraTreesClassifier(max_samples=0.75, n_estimators= 3000, bootstrap=True, min_samples_split=2, min_samples_leaf=1, criterion="entropy", random_state=0)
  classifier.fit(X_train,y_train)
  # print(classifier.get_params())
  return classifier.predict(X_test)

def randomized_search_fold_size_rf_fit_predict(X_train, y_train, X_test):
    max_samples = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    random_grid = {'max_samples': max_samples}

    classifier = RandomForestClassifier(n_estimators= 5000, bootstrap=True, max_depth=40, min_samples_split=2, min_samples_leaf=1, criterion="entropy", random_state=0)
    rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)
    return rf_random, rf_random.predict(X_test)

def randomized_search_cv_rf_fit_predict(X_train, y_train, X_test):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    max_features = ['auto', 'sqrt']
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    classifier = RandomForestClassifier(criterion="entropy")
    rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(X_train, y_train)

    #print("Hyperparameters: ", rf_random.best_params_)
    return rf_random, rf_random.predict(X_test)