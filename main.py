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
import process as pr
import process as pr2
import process as pr3
import process as pr4

pd1 = pd.read_csv('./samples/2nd_data/train1.csv')
pd2 = pd.read_csv('./samples/2nd_data/train2.csv')
pd3 = pd.read_csv('./samples/2nd_data/train3.csv')

# Data Shape : (421201, 88)
# Columns : timestamp / P1 Data / P2 Data / P3 Data / P4 Data / Attack
train_df = pd.concat([pd1, pd2, pd3], axis=0, ignore_index=True)
valid_df = pd.read_csv('./samples/2nd_data/test2.csv')
test_df = pd.read_csv('./samples/2nd_data/test1.csv')

train_df = train_df.drop('timestamp', axis=1) # Shape (421201, 87)
valid_df = valid_df.drop('timestamp', axis=1)
test_df = test_df.drop('timestamp', axis=1)

train_df = train_df ** 2
valid_df = valid_df ** 2
test_df = test_df ** 2

# print(test_df.head(5))

def get_column_category(columns, type):
    category_col = []
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

        elif type == 4:
            if column.find('P4') != -1:
                category_col.append(column)

    category_col.append('Attack')
    return category_col

# print(train_df.columns)

p1_train_col = get_column_category(np.array(train_df.columns), 1)
p2_train_col = get_column_category(np.array(train_df.columns), 2)
p3_train_col = get_column_category(np.array(train_df.columns), 3)
p4_train_col = get_column_category(np.array(train_df.columns), 4)

p1_train_df = train_df[p1_train_col]
p2_train_df = train_df[p2_train_col]
p3_train_df = train_df[p3_train_col]
p4_train_df = train_df[p4_train_col]
# print(p1_train_df.head(3))
# print(p2_train_df.head(3))
# print(p3_train_df.head(3))
# print(p4_train_df.head(3))

p1_valid_df = valid_df[p1_train_col]
p2_valid_df = valid_df[p2_train_col]
p3_valid_df = valid_df[p3_train_col]
p4_valid_df = valid_df[p4_train_col]

p1_test_df = test_df[p1_train_col]
p2_test_df = test_df[p2_train_col]
p3_test_df = test_df[p3_train_col]
p4_test_df = test_df[p4_train_col]

p1_process = pr.Process(p1_train_df, p1_valid_df, p1_test_df)
p1_X_train, p1_y_train, p1_X_valid, p1_y_valid, p1_X_test, p1_y_test = p1_process.create_sequences_data()
p2_process = pr2.Process(p2_train_df, p2_valid_df, p2_test_df)
p2_X_train, p2_y_train, p2_X_valid, p2_y_valid, p2_X_test, p2_y_test = p2_process.create_sequences_data()
p3_process = pr3.Process(p3_train_df, p3_valid_df, p3_test_df)
p3_X_train, p3_y_train, p3_X_valid, p3_y_valid, p3_X_test, p3_y_test = p3_process.create_sequences_data()
p4_process = pr4.Process(p4_train_df, p4_valid_df, p4_test_df)
p4_X_train, p4_y_train, p4_X_valid, p4_y_valid, p4_X_test, p4_y_test = p4_process.create_sequences_data()

p1_process.train_model(p1_X_train, p1_y_train, 'P1')
p1_process.predict_process(p1_X_valid, p1_y_valid, p1_X_test, p1_y_test, 'P1')

p2_process.train_model(p2_X_train, p2_y_train, 'P2')
p2_process.predict_process(p2_X_valid, p2_y_valid, p2_X_test, p2_y_test, 'P2')

p3_process.train_model(p3_X_train, p3_y_train, 'P3')
p3_process.predict_process(p3_X_valid, p3_y_valid, p3_X_test, p3_y_test, 'P3')

p4_process.train_model(p4_X_train, p4_y_train, 'P4')
p4_process.predict_process(p4_X_valid, p4_y_valid, p4_X_test, p4_y_test, 'P4')

# train_set, y_train = train_df.drop('Attack', axis=1).values, train_df['Attack'].values # Shape (421201, 86) (421201, )
# valid_set, y_valid = valid_df.drop('Attack', axis=1).values, valid_df['Attack'].values
# test_set, y_test = test_df.drop('Attack', axis=1).values, test_df['Attack'].values

# feature = train_set.shape[1]
#
# sc = MinMaxScaler(feature_range=(0, 1))
# training_set_scaled = sc.fit_transform(train_set)
# valid_set_scaled = sc.fit_transform(valid_set)
# test_set_scaled = sc.fit_transform(test_set)
#
# TIME_STEPS = 15
#
# def create_sequences(X, y, timesteps=TIME_STEPS):
#     output_X = []
#     output_y = []
#     for i in range(len(X) - timesteps - 1):
#         t = []
#         for j in range(1, timesteps + 1):
#             # Gather the past records upto the lookback period
#             t.append(X[[(i + j + 1)], :])
#         output_X.append(t)
#         output_y.append(y[i + timesteps + 1])
#     return np.squeeze(np.array(output_X)), np.array(output_y)
#
# X_train, y_train = create_sequences(training_set_scaled, y_train)
# X_valid, y_valid = create_sequences(valid_set_scaled, y_valid)
# X_test, y_test = create_sequences(test_set_scaled, y_test)
#
# print(X_train.shape)
# print(y_train.shape)
#
# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[2]), activation='relu'))
# model.add(LSTM(32, return_sequences=True, activation='relu'))
# model.add(Dense(units=X_train.shape[2]))
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# model.summary()
#
# history = model.fit(X_train, X_train, epochs=100, batch_size=5600,
#           callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],  validation_split=0.33)
#
# print(history.history['accuracy'])
# print(history.history['loss'])
#
# def vis(history, name):
#     plt.title(f"{name.upper()}")
#     plt.xlabel('epochs')
#     plt.ylabel(f"{name.lower()}")
#     value = history.history.get(name)
#     val_value = history.history.get(f"val_{name}", None)
#     epochs = range(1, len(value) + 1)
#     plt.plot(epochs, value, 'b-', label=f'training {name}')
#     if val_value is not None:
#         plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2), fontsize=10, ncol=1)
#
#
# def plot_history(history):
#     key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
#     plt.figure(figsize=(12, 4))
#     for idx, key in enumerate(key_value):
#         plt.subplot(1, len(key_value), idx + 1)
#         vis(history, key)
#     plt.tight_layout()
#     plt.show()
#
# # plot_history(history)
#
# def flatten(X):
#     flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
#     for i in range(X.shape[0]):
#         flattened_X[i] = X[i, (X.shape[1] - 1), :]
#     return flattened_X
#
# valid_result = model.predict(X_valid)
#
# mse = np.mean(np.power(flatten(X_valid) - flatten(valid_result), 2), axis=1)
#
# error_df = pd.DataFrame({'Reconstruction_error': mse,
#                          'True_class': y_valid})
#
# precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])
# print(f'precision : {precision_rt} / recall : {recall_rt} / threshold : {threshold_rt}')
#
# plt.figure(figsize=(8,5))
# plt.plot(threshold_rt, precision_rt[1:], label='Precision')
# plt.plot(threshold_rt, recall_rt[1:], label='Recall')
# plt.xlabel('Threshold')
# plt.ylabel('Precision/Recall')
# plt.legend()
# plt.savefig('./result/threshold_precision_recall.png')
#
# index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p==r]
# index_cnt = index_cnt[0]
# print(index_cnt)
# print('precision: ', precision_rt[index_cnt],', recall: ', recall_rt[index_cnt])
#
# # fixed Threshold
# threshold_fixed = threshold_rt[index_cnt]
# print('threshold: ', threshold_fixed)
# #plt.show()
#
# test_result = model.predict(X_test)
# mse = np.mean(np.power(flatten(X_test) - flatten(test_result), 2), axis=1)
#
# error_df = pd.DataFrame({
#     'Reconstruction_error': mse,
#     'True_class': y_test
# })
#
# groups = error_df.groupby('True_class')
# fig, ax = plt.subplots()
#
# for name, group in groups:
#     ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
#             label='Attack' if name == 1 else 'Normal')
# ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors='r', zorder=100, label='Threshold')
# ax.legend()
# plt.savefig('./result/Reconstruction error for different classes')