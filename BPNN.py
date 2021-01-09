from keras.datasets import boston_housing
from keras import models
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

def file_open():
    data = pd.read_csv(r'D:\ML-2020.1.4\database-master\database-master\Data base for ML prediction-new.csv')
    data_list = list(data)
    # drop_name_unname = 'Unnamed: 28'
    # data = file_read.drop(drop_name_unname, axis=1)
    drop_name_unnames = 'Unnamed: 0'
    data = data.drop(drop_name_unnames, axis=1)
    data_list = list(data)
    y = data['Eads(H)']
    x = data.drop(['Eads(H)'], axis=1, inplace=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
    train_y = train_y.values
    test_y = test_y.values
    return train_x, train_y, test_x, test_y



def bulid_model(train_x, train_y, test_x, test_y):
    train_x = train_x
    train_y = train_y
    model = keras.Sequential((
        layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu',
                     input_shape=[len(list(train_x))]),
        layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
        layers.Dense(1)
    ))
    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse']
                  )
    return model

def map_plot(test_x,test_y,pre_y):
    test_x_m = test_x
    test_y_m = test_y
    pre_y = pre_y
    plt.plot(test_x_m,pre_y,'r-')
    plt.scatter(test_x_m,test_y)
    plt.show()

def guji(pre_y, test_y):
    pre = pre_y
    pre_pd = pd.DataFrame(pre)
    test_y_g = test_y
    mae = mean_squared_error(test_y_g, pre)
    r2 = r2_score(test_y_g, pre)
    error = round((mae+r2)/2, 2)
    print('r2=', r2)
    print('error=', error)
    return error
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

if __name__ =="__main__":
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    train_x, train_y, test_x, test_y = file_open()
    model = bulid_model(train_x, train_y, test_x, test_y)
    model.summary()
    model_fit = model.fit(train_x,train_y,epochs=1000,validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    pre_y = model.predict(test_x)
    guji(pre_y,test_y)
    loss1, mae1, mse1 = model.evaluate(test_x, test_y, verbose=2)
    # test_x, test_y, pre_y = map_plot(test_x, test_y, pre_y)

