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
print(train_columns)
print(test_columns)

train_df = train_df[train_columns]
train_df['time'] = pd.to_datetime(train_df['time'])
test_df = test_df[test_columns]
test_df['time'] = pd.to_datetime(test_df['time'])
# print(train_df['time'].min(), train_df['time'].max())
# print(train_df.head(5))
# print(test_df['time'].min(), test_df['time'].max())
# print(test_df.head(5))

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df['time'], y=train_df['attack_P1'], name='attack'))
fig.add_trace(go.Scatter(x=test_df['time'], y=test_df['attack_P1'], name='attack'))
# fig.show()

# Data setting
train, test = train_df.loc[:], test_df.loc[:]
print(train.shape, test.shape)
print(train.head(5), test.head(5))

train_input_x, train_input_y = train_df.drop('time', axis=1).drop('attack_P1', axis=1).values, train_df['attack_P1'].values
test_input_x, test_input_y = test_df.drop('time', axis=1).drop('attack_P1', axis=1).values, train_df['attack_P1'].values

n_features = train_input_x.shape[1]  # Feature 의 개수
print(n_features)

# Transform to Series Data
TIME_STEPS = 60
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

train_x, train_y = create_sequences(train_input_x, train_input_y)
test_x, test_y = create_sequences(test_input_x, test_input_y)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.33)
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)

x_train_y0 = train_x[train_y == 0]
x_train_y1 = train_x[train_y == 1]

x_valid_y0 = valid_x[valid_y == 0]
x_valid_y1 = valid_x[valid_y == 1]

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

x_train_y0_scaled = scale(x_train_y0, scaler)
x_valid_scaled = scale(valid_x, scaler)
x_valid_y0_scaled = scale(x_valid_y0, scaler)
x_test_scaled = scale(test_x, scaler)

# Create Model
model = Sequential()
# Encoder (64 > 16)
model.add(LSTM(64, activation='relu', input_shape=(TIME_STEPS, n_features), return_sequences=True))
model.add(LSTM(16, activation='relu', return_sequences=False))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(TIME_STEPS))
# Decoder (16 > 64)
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(n_features)))
model.summary()

# Model compile
model.compile(optimizer='adam', loss='mae')

#Model fit
history = model.fit(x_train_y0_scaled, x_train_y0_scaled, epochs=100, batch_size=1800, validation_data=(x_valid_y0_scaled, x_valid_y0_scaled),
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')], shuffle=False)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='vaild loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.show()

valid_x_predictions = model.predict(x_valid_scaled)
mse = np.mean(np.power(flatten(x_valid_scaled) - flatten(valid_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error':mse,
                         'True_class':list(valid_y)})
print(error_df.head(10))

precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])

print(precision_rt)
print(recall_rt)
print(threshold_rt)

plt.figure(figsize=(8,5))
plt.plot(threshold_rt, precision_rt[1:], label='Precision')
plt.plot(threshold_rt, recall_rt[1:], label='Recall')
plt.xlabel('Threshold'); plt.ylabel('Precision/Recall')
plt.legend()
# plt.show()

index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p==r][0]
print('precision: ',precision_rt[index_cnt],', recall: ',recall_rt[index_cnt])

# fixed Threshold
threshold_fixed = threshold_rt[index_cnt]
print('threshold: ',threshold_fixed)

test_x_predictions = model.predict(x_test_scaled)
mse = np.mean(np.power(flatten(x_test_scaled) - flatten(test_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': test_y.tolist()})

groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
# plt.show()

# classification by threshold
pred_y = [1 if e > threshold_fixed else 0 for e in error_df['Reconstruction_error'].values]

conf_matrix = metrics.confusion_matrix(error_df['True_class'], pred_y)
# plt.figure(figsize=(7, 7))
# sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Class'); plt.ylabel('True Class')
# plt.show()

false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(error_df['True_class'], error_df['Reconstruction_error'])
roc_auc = metrics.auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()