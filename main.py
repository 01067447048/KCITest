import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

pd1 = pd.read_csv('./samples/train1.csv')
pd2 = pd.read_csv('./samples/train2.csv')
pd3 = pd.read_csv('./samples/train3.csv')

train_df = pd.concat([pd1, pd2, pd3], axis=0, ignore_index=True)
test_df = pd.read_csv('./samples/test1.csv')
print('Number of rows and columns: ', train_df.shape, test_df.shape)

train_df_without_time = train_df.drop('time', axis=1).drop('attack_P1', axis=1).drop('attack_P2', axis=1).drop('attack_P3', axis=1)
test_df_without_time = test_df.drop('time', axis=1).drop('attack_P1', axis=1).drop('attack_P2', axis=1).drop('attack_P3', axis=1)

print('With out Time & attack Number of rows and columns: ', train_df_without_time.shape, test_df_without_time.shape)

train_set, y_train = train_df_without_time.drop('attack', axis=1).values, train_df_without_time['attack'].values
test_set, y_test = test_df_without_time.drop('attack', axis=1).values, test_df_without_time['attack'].values

feature = train_set.shape[1]

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_set)
test_set_scaled = sc.fit_transform(test_set)

TIME_STEPS = 60

# for i in range(TIME_STEPS, training_set_scaled.shape[0]):
#     X_train.append(training_set_scaled[i - 60:i, 0])
#     y_train.append(training_set_scaled[i, 0])
#
# for i in range(TIME_STEPS, test_set_scaled.shape[0]):
#     X_test.append(test_set_scaled[i - 60:i, 0])
#     y_test.append(test_set_scaled[i, 0])
#
# X_train, y_train = np.array(X_train), np.array(y_train)
# X_test, y_test = np.array(X_test), np.array(y_test)

def create_sequences(X, y, timesteps=TIME_STEPS):
    output_X = []
    output_y = []
    for i in range(len(X) - timesteps - 1):
        t = []
        for j in range(1, timesteps + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + timesteps + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)

X_train, y_train = create_sequences(training_set_scaled, y_train)
X_test, y_test = create_sequences(test_set_scaled, y_test)

print(X_train.shape)
print(y_train.shape)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], feature))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], feature))
#
# print(X_train.shape)
# print(y_train.shape)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[2]), activation='relu'))
model.add(LSTM(32, return_sequences=True, activation='relu'))
model.add(LSTM(8, return_sequences=True, activation='relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(4, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[2])))
# model.add(LSTM(16))
# model.add(Dropout(0.2))
model.add(Dense(units=X_train.shape[2]))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, X_train, epochs=100, batch_size=3600,
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],  validation_split=0.33)

print(history.history['accuracy'])
print(history.history['loss'])

test_result = model.predict(X_test)

def vis(history, name):
    plt.title(f"{name.upper()}")
    plt.xlabel('epochs')
    plt.ylabel(f"{name.lower()}")
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}", None)
    epochs = range(1, len(value) + 1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None:
        plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2), fontsize=10, ncol=1)


def plot_history(history):
    key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    plt.figure(figsize=(12, 4))
    for idx, key in enumerate(key_value):
        plt.subplot(1, len(key_value), idx + 1)
        vis(history, key)
    plt.tight_layout()
    plt.show()

# plot_history(history)

def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return flattened_X

test_result = flatten(test_result)
print(test_result.shape)
print(test_result)
test_result = sc.inverse_transform(test_result)
f = open('./result_relu.txt', 'w')
f.write(str(test_result.shape))
f.write('\n')
f.write(str(test_result))
f.close()

# plt.plot(test_df.loc[:, 'time'], test_result, color = 'blue', label = 'Model Data')
# plt.show()