# Bibliotecas usadas

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import LSTM, Dense

# Importei o dataset

dados = pd.read_csv(
    '/home/renato/Projetos_Python/Machine_Learning_Cafe/coffee-price-prediction/data/processed/coffee.csv')

# Transformando as datas no tipo 'datetime'

dados['Date'] = pd.to_datetime(dados['Date'], format='%Y-%m-%d')

# Traduzindo o nome das colunas

valores_traduzidos = {
    'Date': 'Data',
    'Open': 'Abertura',
    'High': 'Maior_valor_dia',
    'Low': 'Menor_valor_dia',
    'Close': 'Fechamento_dia',
    'Volume': 'Volume',
    'Currency': 'Moeda',
}

dados = dados.rename(columns=valores_traduzidos)

# Deletando colunas que não são importantes para a predição

dados = dados.drop(['Volume', 'Moeda'], axis=1)

# Construindo o modelo de previsão do Keras - PT1

window_size = 20
prediction_size = 30

X = []
y = []

for i in range(len(dados)-window_size-prediction_size+1):
    X.append(dados.iloc[i:i+window_size]["Fechamento_dia"].values)
    y.append(dados.iloc[window_size+i+prediction_size-1]["Fechamento_dia"])

X = np.array(X)
y = np.array(y)


X = X.reshape(len(X), window_size, 1)

# Separação Treino e Teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# Construindo o modelo de previsão do Keras  - PT2

units = 150

activation = "relu"

input_shape = (window_size, 1)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss = "mean_squared_error"


def r2_keras(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(
        y_true - tf.keras.backend.mean(y_true)))
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))


tf.random.set_seed(42)
model = tf.keras.models.Sequential(
    [
        # -- LSTM ---------------------------------------------
        tf.keras.layers.LSTM(
            units,
            activation=activation,
            input_shape=input_shape,
            kernel_initializer=GlorotNormal(seed=42),
        ),

        # -- Dense ---------------------------------------------
        tf.keras.layers.Dense(1, activation="linear",
                              kernel_initializer=GlorotNormal(seed=42))
    ]
)

tf.random.set_seed(42)
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=[r2_keras],
)

# Treinando o modelo

history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.1,
    verbose=1,
)

# Verificando taxa de acerto do modelo

y_train_pred = model.predict(X_train).flatten()
y_test_pred = model.predict(X_test).flatten()

metrics = {
    "MAE_train": mean_absolute_error(y_train, y_train_pred),
    "MAE_test": mean_absolute_error(y_test, y_test_pred),
    "R2_train": r2_score(y_train, y_train_pred),
    "R2_test": r2_score(y_test, y_test_pred),
}

print(metrics)
