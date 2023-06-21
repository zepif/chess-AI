import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import numpy as np
import chess
import chess.engine
import random
import os
from state import Board

state = Board()

board = state.random_board()
score = state.stockfish(board, 15)
state.board = board
state.split_dims()

arr1 = np.empty(0)
arr2 = np.empty(0)
arr1 = np.append(arr1, state.board3d)
arr2 = np.append(arr2, score)

np.savez('dataset.npz', b=arr1, v=arr2)

data = np.load('dataset.npz', allow_pickle=True, encoding='latin1')

print(data['b'].shape)

sz = 0  # 10^6

new_data = {}
new_data['b'] = data['b']
print(new_data['b'])
new_data['v'] = data['v']


for i in range(sz):
    board = state.random_board()
    score = state.stockfish(board, 15)
    state.board = board
    state.split_dims()

    new_data['b'] = np.append(new_data['b'], state.board3d)
    new_data['v'] = np.append(new_data['v'], score)

print(new_data['b'])
print(new_data['b'].dtype)
print(new_data['b'].shape)
print(new_data['v'])
np.savez('dataset.npz', b=new_data['b'], v=new_data['v'])

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))

    x = board3d
    for i in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)
    # good idea, which could be added later
    # https://en.wikipedia.org/wiki/Evaluation_function#Neural_networks
    return models.Model(inputs=board3d, outputs=x)


def build_model_residual(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))

    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format=None, activation='relu')(board3d)
    for i in range(conv_depth):
        previous = x
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format=None, activation='sigmoid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format=None, activation='sigmoid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, previous])
        x = layers.Activation('relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, 'sigmoid')(x)

    return models.Model(inputs=board3d, outputs=x)


# model = build_model(32, 4)
model = build_model_residual(32, 4)
utils.plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)


def get_dataset():
    b, v = new_data['b'], new_data['v']
    v = np.asarray(v, dtype=np.float32)
    
    valid_indices = []
    processed_v = []

    for i, value in enumerate(v):
        if value is not None:
            valid_indices.append(i)
            processed_v.append(value)

    b = b[valid_indices]
    #b = b.reshape((14, 8, 8))   Reshape input to match the model's input shape
    v = np.asarray(processed_v, dtype=np.float32)
    v = v / np.abs(v).max() / 2 + 0.5

    return b, v


X_train, y_train = get_dataset()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_train data type:", X_train.dtype)
print("y_train data type:", y_train.dtype)

print("X_train:")
print(X_train)

print("y_train:")
print(y_train)

model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
model.summary()
model.fit(X_train, y_train,
          batch_size=2048,
          epochs=1000,
          verbose=1,
          validation_split=0.1,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                     callbacks.EarlyStopping(monitor='loss', min_delta=0.01)])
model.save('model.h5')
