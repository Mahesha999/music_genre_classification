from sklearn import svm

def svm_fit_predict(X_train, y_train, X_test):
  model=svm.SVC(C=2,kernel='rbf') #regularization parameter, radial basis function, 
  model.fit(X_train,y_train)
  return model.predict(X_test)