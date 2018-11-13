# Vanilla RNN

# This is a vanilla version of rudimentary RNN with random structure 
# which cannot yet be tested (waiting on data from Antoine), most likely 
# will need to be tweaked as I can’t test (assumed single step ahead using
# 10 days of data as ‘features’, assumed output as predicted value i.e. 
# the higher the predicted value the higher the confidence that we predict
# the asset goes up & vice versa). Further details in my journal googledoc.

# Since I have no knowledge of Keras, this follows the tutorial on 
# machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# Full Keras documentation can be found here: https://keras.io/layers/recurrent/

# 1. Import dependancies
import numpy as numpy
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# 2. Functions
def antoineData():
	# Antoine's script will go here, prelim data will be 'assetCode',
	# 'time', 'volume', 'open', 'returnsOpenPrevMktres1', 
    # 'returnsOpenPrevMkres10', 'returnsOpenNextMktres10', 
    # 'sentimentNegative', 'sentimentNeutral'
	return 0

# 3. Import data
x_train, y_train, x_test, y_test = antoineData() 

# 4. Build model from Keras 
model = Sequential() # Sequential model is a linear stack of layers 
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]))) # adds LSTM layer
model.add(Dense(1)) # adds a dense layer
model.compile(loss='mae', optimizer='adam') # sets the loss as mean absolute error and the optimiser as ADAM

# 5. Fit RNN
history = model.fit(x_train, y_train, epochs=50, batch_size=72, validation_data=(x_test, y_test), verbose=2, shuffle=False) # fits

# 6. Plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
y_hat = model.predict(x_test)
# calculate the error (can modify this for accuracy instead if needed using skl)
RMSE = sqrt(mean_squared_error(y_test, y_hat))