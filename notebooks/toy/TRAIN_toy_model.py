# %%
"""
# Train Toy Model
"""

# %%
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
import tensorflow as tf

RAW_DATA_PATH = 'input/data/'

MINITRAIN_PATH = 'input/toy/train/'
MINIVAL_PATH = 'input/toy/val/'
MINITEST_PATH = 'input/toy/test/'

META_PATH = 'input/metadata/'
IMG_SIZE = 256
N_CHANNELS = 3

print(keras.__version__)
print(tf. __version__)

print(os. getcwd())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

# %%
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

# %%
from tensorflow import keras
from keras.regularizers import l2,l1
layers = [
    keras.Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS)),
    keras.layers.experimental.preprocessing.Rescaling(1./255.),
    # keras.layers.Dropout(0.8),
    keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (4,4)),

    # keras.layers.Dropout(0.8),
    keras.layers.Conv2D(64,(3,3), activation = 'relu',kernel_regularizer=l2(0.05)),
    keras.layers.Dropout(0.5),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    
    keras.layers.Conv2D(128,(3,3), activation = 'relu',kernel_regularizer=l2(0.05)),
    keras.layers.Dropout(0.5),
    keras.layers.MaxPooling2D(pool_size = (2,2)),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),


    keras.layers.Dense(output_shape, activation='softmax')
]

model = keras.Sequential(layers)
'''
sparse_categorical_crossentropy para cuando las categorias vienen en una unica columna
categorical_crossentropy para cuando las categorias vienen en formato dummy
'''
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# %%
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_val, y_val))

# %%
df_hist = model.history.history
df_hist = pd.DataFrame(df_hist)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('toy_accuracy.pdf')
plt.close()
