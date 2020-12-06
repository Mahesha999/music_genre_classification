from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def nn_fit_predict(X_train, y_train, X_test):
    model = Sequential([
        Dense(58, input_shape=(58,), activation='relu'),
        Dense(58, input_shape=(58,), activation='relu'),
        Dense(10,  activation='softmax'),
    ])

    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=10,epochs=100)
    return model.predict(X_test)
  
#   ANN_score=np.zeros(5,dtype=float)
# i=0
# for train_index,test_index in kf.split(X):
#   X_train,X_test,y_train,y_test=X[train_index],X[test_index],Y[train_index],Y[test_index]
#   ANN_score[i]=fn_ANN(X_train,X_test,y_train,y_test)
#   i=i+1
# avg_ANN_score=np.sum(ANN_score)/5.0
# print(avg_ANN_score)