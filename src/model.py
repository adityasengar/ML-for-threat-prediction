from keras.models import Sequential
from keras.layers import Dense

def create_model(input_dim):
    """Creates the DNN model."""
    model = Sequential()
    num = 128
    model.add(Dense(num, activation='relu', input_dim=input_dim))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
