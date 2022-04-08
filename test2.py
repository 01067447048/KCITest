from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import metrics

np.random.seed(1)
tf.random.set_seed(1)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

# Data
def get_column_category(columns, type):
    category_col = ['time']
    for column in columns:
        if type == 1:
            if column.find('P1') != -1:
                category_col.append(column)

        elif type == 2:
            if column.find('P2') != -1:
                category_col.append(column)

        elif type == 3:
            if column.find('P3') != -1:
                category_col.append(column)

    return category_col


train_df = pd.read_csv('./samples/train1.csv')
test_df = pd.read_csv('./samples/test1.csv')
train_columns = np.array(train_df.columns)
train_columns = get_column_category(train_columns, 1)
test_columns = np.array(test_df.columns)
test_columns = get_column_category(test_columns, 1)

train_df = train_df[train_columns]
train_df['time'] = pd.to_datetime(train_df['time'])
test_df = test_df[test_columns]
test_df['time'] = pd.to_datetime(test_df['time'])

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df['time'], y=train_df['attack_P1'], name='attack'))
fig.add_trace(go.Scatter(x=test_df['time'], y=test_df['attack_P1'], name='attack'))
# fig.show()

train, test = train_df.loc[:], test_df.loc[:]
print(train.shape, test.shape)
print(train.head(5), test.head(5))

train_input_x, train_input_y = train_df.drop('time', axis=1).drop('attack_P1', axis=1), train_df['attack_P1']
test_input_x, test_input_y = test_df.drop('time', axis=1).drop('attack_P1', axis=1), train_df['attack_P1']

n_features = train_input_x.shape[1]  # Feature 의 개수
print(n_features)
print(train_input_x.shape)
print(train_input_y.shape)

TIME_STEPS = 60
def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)

train_x, train_y = create_sequences(train_input_x, train_input_y)
test_x, test_y = create_sequences(test_input_x, test_input_y)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return flattened_X


def scale(X, local_scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = local_scaler.transform(X[i, :, :])

    return X

scaler = StandardScaler().fit(flatten(train_x))

scaled_train_x = scale(train_x, scaler)
scaled_test_x = scale(test_x, scaler)

print(scaled_train_x.shape, scaled_test_x.shape)

model = Sequential()
# Encoder (128 64)
model.add(LSTM(16, activation='relu', input_shape=(TIME_STEPS, n_features), return_sequences=True))
model.add(LSTM(4, activation='relu', return_sequences=False))
model.add(RepeatVector(TIME_STEPS))
# Decoder (64, 128)
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.summary()

# model = Sequential()
# model.add(LSTM(128, input_shape=(TIME_STEPS, n_features)))
# model.add(Dropout(rate=0.2))
# model.add(RepeatVector(TIME_STEPS))
# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(rate=0.2))
# model.add(TimeDistributed(Dense(1)))
# model.summary()

# Model compile
model.compile(optimizer='adam', loss='mae')

history = model.fit(scaled_train_x, train_y, epochs=100, batch_size=3600, validation_split=0.3,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')], shuffle=False)


model.evaluate(scaled_test_x, test_y)

X_train_pred = model.predict(scaled_train_x, verbose=0)
print(X_train_pred.shape)
print(X_train_pred)

# train_mae_loss = np.mean(np.abs(X_train_pred - scaled_train_x), axis=1)
#
# def get_threshold(mae_loss):
#     threshold = []
#     mae_loss = mae_loss.reshape(mae_loss.shape[1], mae_loss.shape[0])
#     for i in range(mae_loss.shape[0]):
#         print(f'len : {len(mae_loss[i])} / {mae_loss[i]}')
#         a = np.max(mae_loss[i])
#         a = float(a * 0.85)
#         print(f'threshold[{i}] : {a}')
#         threshold.append(a)
#
#     return threshold
#
#
# # threshold = np.max(train_mae_loss)
# threshold = get_threshold(train_mae_loss)
# print(f'Reconstruction error threshold: {threshold}')
# print(f'Reconstruction error threshold: {len(threshold)}')
#
# X_test_pred = model.predict(scaled_test_x, verbose=0)
# test_mae_loss = np.mean(np.abs(X_test_pred - scaled_test_x), axis=1)
# test_mae_loss = test_mae_loss.reshape(test_mae_loss.shape[1], test_mae_loss.shape[0])
# print(f'test_mae_loss : {test_mae_loss} / test_mae_loss_shape : {test_mae_loss.shape}')
#
# test_score_df = pd.DataFrame(test[TIME_STEPS:])
# # test_score_df = pd.DataFrame(scaled_test_x[TIME_STEPS:])
# print(test_score_df.head(5))
# print(test_score_df.shape)
#
# sr2 = pd.Series(threshold, name='threshold')
#
# anomalies_list = []
#
# for i in range(test_input_x.shape[1]):
#     if test_score_df.columns[i+1] == 'time' or test_score_df.columns[i+1] == 'attack_P1':
#         continue
#     loss = test_score_df.columns[i+1] + '_loss'
#     threshold_col = test_score_df.columns[i+1] + '_threshold'
#     anomalies_col = test_score_df.columns[i+1] + '_anomaly'
#     anomalies_list.append(anomalies_col)
#     # test_score_df[loss] = test_mae_loss[i]
#     # test_score_df[threshold_col] = threshold[i]
#     test_score_df[anomalies_col] = test_mae_loss[i] > threshold[i]
#
#
# # test_score_df['threshold'] = threshold
# # test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
# # test_score_df['attack'] = test[TIME_STEPS:]['attack_P1']
# # result = pd.concat([test_score_df, sr1, sr2, sr3], axis=1)
# print(test_score_df.head(5))
# test_score_df.to_csv('./test_score_df.csv')
#
# anomalies_df = test_score_df.loc[:, anomalies_list]
# print(anomalies_df.head(5))
# print(anomalies_df.shape[0])
# print(anomalies_df.iloc[0].values)
# print(anomalies_df.iloc[0].values[1])
#
# anomalies = []
# anomalies2 = []
# f = open('./anomalies.txt', 'w')
# for i in range(anomalies_df.shape[0]):
#     count = 0
#     tmp = anomalies_df.iloc[i].values
#     for j in range(len(tmp)):
#         if tmp[j] == True:
#             count = count + 1
#             # f.write(str(i))
#             anomalies.append(i)
#         if j == len(tmp):
#             if count > 3:
#                 f.write(str(i))
#                 anomalies2.append(i)
#
# print('------------------------------------------')
# print(anomalies)
# print(len(anomalies))
# print(anomalies2)
# print(len(anomalies2))
#
# def get_acc(anomaly):
#     attack_P1 = test_df['attack_P1'].values
#     cnt = 0
#     if len(anomaly) == 0:
#         return 0
#
#     for i in range(len(anomaly)):
#         if attack_P1[i] == 0:
#             attack = False
#         else:
#             attack = True
#
#         if anomaly[i] == attack:
#             cnt = cnt + 1
#
#     return float(cnt / len(anomaly))
#
# print(get_acc(anomalies))
# print(get_acc(anomalies2))



# anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
# print(f'anomalies.shape : {anomalies.shape}')