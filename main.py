import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
import tensorflow as tf

RAW_DATA_PATH = 'input/data/'

MINITRAIN_PATH = 'input/data/train/'
MINIVAL_PATH = 'input/data/val/'
MINITEST_PATH = 'input/data/test/'

META_PATH = 'input/metadata/'
IMG_SIZE = 256
N_CHANNELS = 3

# get epochs from command line
# introduce arguments as follows: python main.py 10 32
import sys
N_EPOCHS = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])

# N_EPOCHS = 10
# BATCH_SIZE = 32


print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

train_data = np.load(os.path.join(MINITRAIN_PATH, 'train.npz'))
X_train, y_train = train_data['X'], train_data['y']
output_shape = np.unique(y_train).shape[0]
print(X_train.shape, y_train.shape)

val_data = np.load(os.path.join(MINIVAL_PATH, 'val.npz'))
X_val, y_val = val_data['X'], val_data['y']
print(X_val.shape, y_val.shape)

test_data = np.load(os.path.join(MINITEST_PATH, 'test.npz'))
X_test, y_test = test_data['X'], test_data['y']
print(X_test.shape, y_test.shape)

layers = [
    keras.Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS)),
    keras.layers.Rescaling(1./255.),
    keras.layers.Conv2D(16,(3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),

    keras.layers.Dropout(0.8),
    keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),

    keras.layers.Dropout(0.8),
    keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.8),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(output_shape, activation='softmax')
]

model = keras.Sequential(layers)
'''
sparse_categorical_crossentropy para cuando las categorias vienen en una unica columna
categorical_crossentropy para cuando las categorias vienen en formato dummy
'''
model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

df_hist = model.history.history
df_hist = pd.DataFrame(df_hist)

fig, ax = plt.subplots(1,2, figsize=(15,5))
df_hist[['accuracy', 'val_accuracy']].plot(ax=ax[0])
df_hist[['loss', 'val_loss']].plot(ax=ax[1])
plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight");
