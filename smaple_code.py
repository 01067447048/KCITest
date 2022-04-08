from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
import pandas as pd


np.random.seed(1)
tf.random.set_seed(1)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

# 데이터 확인.
df = pd.read_csv('JNJ.csv')
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'].min(), df['Date'].max())

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close price'))
# fig.show()


# 데이터 분할.
train, test = df.loc[df['Date'] <= '2020-12-31'], df.loc[df['Date'] > '2020-12-31']
print(train.shape, test.shape)

# 데이터 표준화.
scaler = StandardScaler()
scaler = scaler.fit(train[['Close']])

train['Close'] = scaler.transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])
print(train['Close'], test['Close'])

# Create Sequences
TIME_STEPS = 30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])

    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])

print(f'Training shape: {X_train.shape}, {y_train.shape}')
print(f'Test shape: {X_test.shape}, {y_test.shape}')
#
# # Create Model
# # 30개의 시간 단계와 하나의 기능이 있는 입력 시퀀스를 예상하고 30개의 시간 단계와 하나의 기능이 있는 시퀀스를 출력하는 재구성 LSTM Autoencoder 아키텍처를 정의합니다.
# # RepeatVector()입력을 30번 반복합니다.
# # 설정 return_sequences=True하면 출력이 계속 시퀀스가 ​​됩니다.
# # TimeDistributed(Dense(X_train.shape[2]))출력을 얻기 위해 끝에 추가됩니다. 여기서 X_train.shape[2]는 입력 데이터의 기능 수입니다.
#
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

# # Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=52, validation_split=0.25,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

# # Model Plot
# # plt.plot(history.history['loss'], label='Training loss')
# # plt.plot(history.history['val_loss'], label='Validation loss')
# # plt.legend()
# # plt.show()
#
# # Model evaluate
model.evaluate(X_test, y_test)
#
# # Anomaly detection
# # 훈련 데이터에서 MAE 손실을 찾습니다.
# # 훈련 데이터의 최대 MAE 손실 값을 reconstruction error threshold 로 설정
# # 테스트 세트의 데이터 포인트에 대한 재구성 손실이 이 reconstruction error threshold 값보다 크면 이 데이터 포인트를 비정상으로 분류합니다.
#
X_train_pred = model.predict(X_train, verbose=0)

# plt.hist(train_mae_loss, bins=50)
# plt.xlabel('Train MAE loss')
# plt.ylabel('Number of Samples')
# plt.show()

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
# np.sort(train_mae_loss)
threshold = np.max(train_mae_loss)
print(f'Reconstruction error threshold: {threshold}')

X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
print(test_mae_loss.shape)
#
# # plt.hist(test_mae_loss, bins=50)
# # plt.xlabel('Test MAE loss')
# # plt.ylabel('Number of samples')
# # plt.show()
#
test_score_df = pd.DataFrame(test[TIME_STEPS:])
print(test_score_df.head(5))
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['Close'] = test[TIME_STEPS:]['Close']
print(test_score_df.head(5))

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['loss'], name='Test loss'))
# fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['threshold'], name='Threshold'))
# fig.show()
#
anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
print(f'anomalies.shape : {anomalies.shape}')
#
#
# fig = go.Figure()
# test_score_df_close = np.array(test_score_df['Close']).reshape(-1, test_score_df['Close'].shape[0])
# anomalies_close = np.array(anomalies['Close']).reshape(-1, anomalies['Close'].shape[0])
# test_score_df_close = scaler.inverse_transform(test_score_df_close).ravel()
# anomalies_close = scaler.inverse_transform(anomalies_close).ravel()
# fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df_close, name='Close price'))
# fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies_close, mode='markers', name='Anomaly'))
# fig.update_layout(showlegend=True, title='Detected anomalies')
# fig.show()