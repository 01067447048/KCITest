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
from sklearn import metrics
from keras.callbacks import EarlyStopping

pd1 = pd.read_csv('./samples/train1.csv')
pd2 = pd.read_csv('./samples/train2.csv')
pd3 = pd.read_csv('./samples/train3.csv')

train_df = pd.concat([pd1, pd3], axis=0, ignore_index=True)
valid_df = pd2
# train_df = pd.read_csv('./samples/train1.csv')
test_df = pd.read_csv('./samples/test1.csv')
print('Number of rows and columns: ', train_df.shape, test_df.shape)

train_df_without_time = train_df.drop('time', axis=1).drop('attack_P1', axis=1).drop('attack_P2', axis=1).drop('attack_P3', axis=1)
valid_df_without_time = valid_df.drop('time', axis=1).drop('attack_P1', axis=1).drop('attack_P2', axis=1).drop('attack_P3', axis=1)
test_df_without_time = test_df.drop('time', axis=1).drop('attack_P1', axis=1).drop('attack_P2', axis=1).drop('attack_P3', axis=1)

print('With out Time & attack Number of rows and columns: ', train_df_without_time.shape, test_df_without_time.shape)

train_set, y_train = train_df_without_time.drop('attack', axis=1).values, train_df_without_time['attack'].values
valid_set, y_valid = valid_df_without_time.drop('attack', axis=1).values, valid_df_without_time['attack'].values
test_set, y_test = test_df_without_time.drop('attack', axis=1).values, test_df_without_time['attack'].values

feature = train_set.shape[1]

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_set)
valid_set_scaled = sc.fit_transform(valid_set)
test_set_scaled = sc.fit_transform(test_set)

TIME_STEPS = 5

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
X_valid, y_valid = create_sequences(valid_set_scaled, y_valid)
X_test, y_test = create_sequences(test_set_scaled, y_test)

print(X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Bidirectional(LSTM(1024, activation='relu', return_sequences=True), input_shape=(TIME_STEPS, X_train.shape[2])))
model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True)))
model.add(Bidirectional(LSTM(16, activation='relu', return_sequences=False)))
model.add(RepeatVector(TIME_STEPS))
model.add(Bidirectional(LSTM(16, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(1024, return_sequences=True)))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, X_train, epochs=100, batch_size=5600,
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],  validation_split=0.33)

print(history.history['accuracy'])
print(history.history['loss'])

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

valid_result = model.predict(X_valid)

mse = np.mean(np.power(flatten(X_valid) - flatten(valid_result), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error':mse,
                         'True_class':list(y_valid)})

print(valid_df.shape)
print(error_df.shape)
print(mse.shape)
print(error_df.head(10))

precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])
print(f'precision : {precision_rt} / recall : {recall_rt} / threshold : {threshold_rt}')

# plt.figure(figsize=(8,5))
# plt.plot(threshold_rt, precision_rt[1:], label='Precision')
# plt.plot(threshold_rt, recall_rt[1:], label='Recall')
# plt.xlabel('Threshold'); plt.ylabel('Precision/Recall')
# plt.legend()
# plt.show()

index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p==r]
# index_cnt = index_cnt[0]
print(index_cnt)
# print('precision: ',precision_rt[index_cnt],', recall: ',recall_rt[index_cnt])

# fixed Threshold
# threshold_fixed = threshold_rt[index_cnt]
# print('threshold: ',threshold_fixed)

# plt.plot(test_df.loc[:, 'time'], test_result, color = 'blue', label = 'Model Data')
# plt.show()