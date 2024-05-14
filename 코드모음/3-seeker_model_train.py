from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
import numpy as np
import pickle

data = np.load('train_set_seeker_WL120_Q4.npz')
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']

print(f'x_train.shape: {x_train.shape}')
print(f'y_train.shape: {y_train.shape}') 
print(f'x_val.shape: {x_val.shape}')
print(f'y_val.shape: {y_val.shape}') 

######################################################학습###################################################################################
WL = 120
input_shape = (WL,9) # x,y,z pos, vel, obse
num_epochs = 200
early_stop = EarlyStopping(monitor='val_loss', patience=10)


# ##############################################WL100-128-32#######################################
# model = Sequential()
# model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
# model.add(Dropout(0.2))
# model.add(LSTM(32, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(3))
# model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
# model.summary()

# history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_val, y_val), callbacks=[early_stop])
# model.save('seeker.WL100-Q5-100-32')
# with open('history-seeker-WL100-Q5-100-32.pkl', 'wb') as f:
#     pickle.dump(history.history, f)


##############################################WL100-128-64#######################################
model1 = Sequential()
model1.add(LSTM(128, return_sequences=True, input_shape=input_shape))
model1.add(Dropout(0.2))
model1.add(LSTM(64, return_sequences=False))
model1.add(Dropout(0.2))
model1.add(Dense(3))
model1.compile(optimizer=Adam(learning_rate=0.0005), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
model1.summary()

history1 = model1.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_val, y_val), callbacks=[early_stop])
model1.save('seeker.WL120-Q4-128-64')
with open('history-seeker-WL120-Q4-128-64.pkl', 'wb') as f:
    pickle.dump(history1.history, f)


##############################################WL100-128-32#######################################
model2 = Sequential()
model2.add(LSTM(128, return_sequences=True, input_shape=input_shape))
model2.add(Dropout(0.2))
model2.add(LSTM(32, return_sequences=False))
model2.add(Dropout(0.2))
model2.add(Dense(3))
model2.compile(optimizer=Adam(learning_rate=0.0005), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
model2.summary()

history2 = model2.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_val, y_val), callbacks=[early_stop])
model2.save('seeker.WL120-Q4-128-32')
with open('history-seeker-WL120-Q4-128-32.pkl', 'wb') as f:
    pickle.dump(history2.history, f)