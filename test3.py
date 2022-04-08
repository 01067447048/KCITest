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
    category_col.append('attack')
    return category_col

train_df = pd.read_csv('./samples/train1.csv')
test_df = pd.read_csv('./samples/test1.csv')
# print(train_df.head(5))
train_df_without_time = train_df.drop('time', axis=1).drop('attack_P1', axis=1).drop('attack_P2', axis=1).drop('attack_P3', axis=1)
test_df_without_time = test_df.drop('time', axis=1).drop('attack_P1', axis=1).drop('attack_P2', axis=1).drop('attack_P3', axis=1)

for i in train_df_without_time.columns:
    if i.find('P1') == -1 and i.find('attack') == -1:
        train_df_without_time = train_df_without_time.drop(i, axis=1)

for i in test_df_without_time.columns:
    if i.find('P1') == -1 and i.find('attack') == -1:
        test_df_without_time = test_df_without_time.drop(i, axis=1)

dup_col1 = train_df_without_time.columns
dup_col2 = test_df_without_time.columns
# print(train_df_without_time.head(5))
std_scaler = StandardScaler()
fitted = std_scaler.fit(train_df_without_time)
train_df_without_time = std_scaler.transform(train_df_without_time)
train_df_without_time = pd.DataFrame(train_df_without_time, columns=dup_col1)
fitted2 = std_scaler.fit(test_df_without_time)
test_df_without_time = std_scaler.transform(test_df_without_time)
test_df_without_time = pd.DataFrame(test_df_without_time, columns=dup_col2)
# print(train_df_without_time.head(5))
# train_df_without_time.hist(bins=50, figsize=(20, 15))
# test_df_without_time.hist(bins=50, figsize=(20, 15))
# train_df_without_time.plot()
# test_df_without_time.plot()
train_df_list = []
test_df_list = []
# print(pd.DataFrame(train_df_without_time['attack']).join(train_df_without_time['P1_B2004']))
for i in train_df_without_time.columns:
    if i.find('attack') == -1:
        train_df_list.append(
            pd.DataFrame(train_df_without_time['attack']).join(train_df_without_time[i])
        )

for i in test_df_without_time.columns:
    if i.find('attack') == -1:
        test_df_list.append(
            pd.DataFrame(test_df_without_time['attack']).join(test_df_without_time[i])
        )

for i in test_df_list:
    i.plot()
plt.show()

# train_columns = np.array(train_df.columns)
# train_columns = get_column_category(train_columns, 1)
# test_columns = np.array(test_df.columns)
# test_columns = get_column_category(test_columns, 1)
# train_fig = go.Figure()
# # test_fig = go.Figure()
# for i in train_df_without_time.columns:
#     train_fig.add_trace(go.Scatter(x=train_df['time'], y=train_df_without_time[i], name=str(i)))
#
# # for i in test_df.columns:
# #     test_fig.add_trace(go.Scatter(x=test_df['time'], y=test_df[i], name=str(i)))
# # fig.add_trace(go.Scatter(x=train_df['time'], y=train_df['attack_P1'], name='attack'))
# # fig.add_trace(go.Scatter(x=test_df['time'], y=test_df['attack_P1'], name='attack'))
# train_fig.update_layout(showlegend=True, title='Attack')
# train_fig.show()

#
# test_dic = {
#     'A': [False, False, False, False],
#     'B': [False, False, False, False],
#     'C': [True, True, False, False],
#     'D': [False, True, False, True],
#     'result': [True, True, False, True]
# }
#
# test_df = pd.DataFrame(test_dic)
# test_df2 = test_df.drop('result', axis=1)
# result_df = test_df['result'].values
# print(test_df)
# print(test_df2)
# print(result_df)
#
# a = []
#
# for i in range(test_df.shape[0]):
#     tmp = list(test_df.iloc[i])
#     print(tmp)
#     for j in range(len(tmp)):
#         if tmp[j] is True:
#             a.append(i)
#             break
#
#
# print(a)
# cnt = 0
# for i in range(len(a)):
#     if a[i] == result_df[i]:
#        cnt = cnt + 1
#
# print(float(cnt / len(a)))


# train_df = pd.read_csv('./samples/train1.csv')
# test_df = pd.read_csv('./samples/test1.csv')
# train_columns = np.array(train_df.columns)
# train_columns = get_column_category(train_columns, 1)
# test_columns = np.array(test_df.columns)
# test_columns = get_column_category(test_columns, 1)
#
# train_df = train_df[train_columns]
# train_df['time'] = pd.to_datetime(train_df['time'])
# test_df = test_df[test_columns]
# test_df['time'] = pd.to_datetime(test_df['time'])
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=train_df['time'], y=train_df['attack_P1'], name='attack'))
# fig.add_trace(go.Scatter(x=test_df['time'], y=test_df['attack_P1'], name='attack'))
# # fig.show()
#
# train, test = train_df.loc[:], test_df.loc[:]


# train, test = train_df.loc[:], test_df.loc[:, ['time']]
# print(train.shape, test.shape)
# print(train.head(5), test.head(5))

# train_input_x, train_input_y = train_df.drop('time', axis=1).drop('attack_P1', axis=1), train_df['attack_P1']
# test_input_x, test_input_y = test_df.drop('time', axis=1).drop('attack_P1', axis=1), train_df['attack_P1']
#
# n_features = train_input_x.shape[1]  # Feature 의 개수
# print(n_features)
# print(train_input_x.shape)
# print(train_input_y.shape)
#
# TIME_STEPS = 60
# def create_sequences(X, y, time_steps=TIME_STEPS):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         Xs.append(X.iloc[i:(i + time_steps)].values)
#         ys.append(y.iloc[i + time_steps])
#
#     return np.array(Xs), np.array(ys)
#
# train_x, train_y = create_sequences(train_input_x, train_input_y)
# test_x, test_y = create_sequences(test_input_x, test_input_y)
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
#
# def flatten(X):
#     flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
#     for i in range(X.shape[0]):
#         flattened_X[i] = X[i, (X.shape[1] - 1), :]
#     return flattened_X
#
#
# def scale(X, local_scaler):
#     for i in range(X.shape[0]):
#         X[i, :, :] = local_scaler.transform(X[i, :, :])
#
#     return X
#
# scaler = StandardScaler().fit(flatten(train_x))
#
# scaled_train_x = scale(train_x, scaler)
# scaled_test_x = scale(test_x, scaler)
#
# print(scaled_train_x.shape, scaled_test_x.shape)
#
# model = Sequential()
# # Encoder (128 64)
# model.add(LSTM(16, activation='relu', input_shape=(TIME_STEPS, n_features), return_sequences=True))
# model.add(LSTM(4, activation='relu', return_sequences=False))
# model.add(RepeatVector(TIME_STEPS))
# # Decoder (64, 128)
# model.add(LSTM(4, activation='relu', return_sequences=True))
# model.add(LSTM(16, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(1)))
#
# model.summary()
#
# # Model compile
# model.compile(optimizer='adam', loss='mae')
#
# history = model.fit(scaled_train_x, train_y, epochs=100, batch_size=3600, validation_split=0.25,
#                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)
#
#
# model.evaluate(scaled_test_x, test_y)
#
# X_train_pred = model.predict(scaled_train_x, verbose=0)
#
# train_mae_loss = np.mean(np.abs(X_train_pred - scaled_train_x), axis=1)
# threshold = np.max(train_mae_loss)
# print(f'Reconstruction error threshold: {threshold}')
#
# X_test_pred = model.predict(scaled_test_x, verbose=0)
# test_mae_loss = np.mean(np.abs(X_test_pred - scaled_test_x), axis=1)
# print(f'test_mae_loss : {test_mae_loss}')