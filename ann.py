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