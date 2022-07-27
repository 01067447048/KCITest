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
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.callbacks import EarlyStopping

TIME_STEPS = 15


def get_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        recall_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_recall_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * recall_mask
        class_recall = K.cast(K.sum(class_recall_tensor), 'float32') / K.cast(K.maximum(K.sum(recall_mask), 1), 'float32')
        return class_recall
    return recall


def get_precision(interesting_class_id):
    def prec(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        precision_mask = K.cast(K.equal(class_id_pred, interesting_class_id), 'int32')
        class_prec_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * precision_mask
        class_prec = K.cast(K.sum(class_prec_tensor), 'float32') / K.cast(K.maximum(K.sum(precision_mask), 1), 'float32')
        return class_prec
    return prec


def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score

class Process:

    def __init__(self, train_df, valid_df, test_df):
        self.training_set_scaled = None
        self.valid_set_scaled = None
        self.test_set_scaled = None
        self.train_set = train_df.values
        self.y_train = train_df['Attack'].values
        self.valid_set = valid_df.values
        self.y_valid = valid_df['Attack'].values
        self.test_set = test_df.values
        self.y_test = test_df['Attack'].values
        self.feature = self.train_set.shape[1]
        self.model = None


    def data_scaler(self):
        sc = MinMaxScaler(feature_range=(0, 1))
        self.training_set_scaled = sc.fit_transform(self.train_set)
        self.valid_set_scaled = sc.fit_transform(self.valid_set)
        self.test_set_scaled = sc.fit_transform(self.test_set)

    def create_sequences(self, X, y, timesteps=TIME_STEPS):
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

    # 호출 해야 할 함수 1
    def create_sequences_data(self):
        self.data_scaler()
        X_train, y_train = self.create_sequences(self.training_set_scaled, self.y_train)
        X_valid, y_valid = self.create_sequences(self.valid_set_scaled, self.y_valid)
        X_test, y_test = self.create_sequences(self.test_set_scaled, self.y_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    # 호출 해야 할 함수 2
    def train_model(self, X, y, file_name):
        # metrics_list = ['accuracy']
        #
        # for id in self.train_col:
        #     print(id)
        #     metrics_list.append(f'get_precision({id})')
        #     metrics_list.append(f'get_recall({id})')
        #
        # f = open('./met.txt', 'w')
        # for met in metrics_list:
        #     f.write(str(met) + '\n')
        # f.close()
        # print(metrics_list)

        self.model = Sequential()
        self.model.add(LSTM(256, return_sequences=True, input_shape=(TIME_STEPS, X.shape[2]), activation='relu'))
        # self.model.add(LSTM(32, return_sequences=True, activation='relu'))
        self.model.add(Dense(units=X.shape[2]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', precision, recall, f1score])
        # self.model.summary()

        history = self.model.fit(X, X, epochs=100, batch_size=900,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                            validation_split=0.33)

        self.save_acc_data(history, file_name)
        self.save_precision_data(history, file_name)
        self.save_recall_data(history, file_name)

        return history

    def save_acc_data(self, history, file_name):
        plt.plot(history.history['accuracy'], label='acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(loc='lower right')
        plt.savefig(f'./result/{file_name}/acc_data.png')
        plt.clf()

    def save_precision_data(self, history, file_name):
        plt.plot(history.history['precision'], label='precision')
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.legend(loc='lower right')
        plt.savefig(f'./result/{file_name}/precision_data.png')
        plt.clf()


    def save_recall_data(self, history, file_name):
        plt.plot(history.history['recall'], label='recall')
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.legend(loc='lower right')
        plt.savefig(f'./result/{file_name}/recall_data.png')
        plt.clf()


    def flatten(self, X):
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]
        return flattened_X

    # 호출 해야 할 함수. 3
    def predict_process(self, X_valid, y_valid, X_test, y_test, file_name):
        valid_result = self.model.predict(X_valid)

        mse = np.mean(np.power(self.flatten(X_valid) - self.flatten(valid_result), 2), axis=1)

        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': y_valid})

        error_df.to_csv(f'./result/{file_name}/validate_mse.csv')

        precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'],
                                                                               error_df['Reconstruction_error'])
        print(f'precision : {precision_rt} / recall : {recall_rt} / threshold : {threshold_rt}')

        plt.figure(figsize=(8, 5))
        plt.plot(threshold_rt, precision_rt[1:], label='Precision')
        plt.plot(threshold_rt, recall_rt[1:], label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        file = f'./result/{file_name}/threshold_precision_recall.png'
        plt.savefig(file)
        plt.clf()

        index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p == r]
        index_cnt = index_cnt[0]
        print(index_cnt)
        print('precision: ', precision_rt[index_cnt], ', recall: ', recall_rt[index_cnt])

        # fixed Threshold
        threshold_fixed = threshold_rt[index_cnt]
        print('threshold: ', threshold_fixed)
        # plt.show()

        test_result = self.model.predict(X_test)
        mse = np.mean(np.power(self.flatten(X_test) - self.flatten(test_result), 2), axis=1)

        error_df = pd.DataFrame({
            'Reconstruction_error': mse,
            'True_class': y_test
        })
        error_df.to_csv(f'./result/{file_name}/test_mse.csv')

        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label='Attack' if name == 1 else 'Normal')
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors='r', zorder=100, label='Threshold')
        ax.legend()
        file = f'./result/{file_name}/Reconstruction error for different classes'
        plt.savefig(file)
        plt.clf()