import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import numpy as np
import os

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

#model = build_model(32, 4)
model = build_model_residual(32, 4)
utils.plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)

def get_dataset():
    container = np.load('dataset.npz', allow_pickle=True)
    b, v = container['b'], container['v']
    b = b.reshape((129, 14, 8, 8))
    v = np.asarray(v, dtype=np.float32)
    print(b.shape)
    valid_indices = []
    processed_v = []
    
    for i, value in enumerate(v):
        if value is not None:
            valid_indices.append(i)
            processed_v.append(value)
    
    b = b[valid_indices]
    #print(b)
    #b = b.reshape(14, 8, 8)
    v = v / np.abs(v).max() / 2 + 0.5

    return b, v

X_train, y_train = get_dataset()
"""
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_train data type:", X_train.dtype)
print("y_train data type:", y_train.dtype)

print("X_train:")
print(X_train)

print("y_train:")
print(y_train)
"""

model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
model.summary()
model.fit(X_train, y_train, 
          batch_size=2048,
          epochs=1000,
          verbose=1,
          validation_split=0.1,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                     callbacks.EarlyStopping(monitor='loss', min_delta=0.1)],)
model.save('model.h5')


